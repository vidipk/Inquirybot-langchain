import logging

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CHAT_MODEL_CHOICES,
    DATA_PATH,
    DEFAULT_CHAT_MODEL,
    REFUSAL_MESSAGE,
    RETRIEVAL_K,
    SIMILARITY_SCORE_THRESHOLD,
)
from database import (
    get_vector_store,
    ingest_pdf_paths,
    list_pdf_files,
    rebuild_vector_database,
    retrieve_relevant_documents,
)
from utils import (
    append_lead,
    citation_payload,
    format_documents_for_prompt,
    save_uploaded_pdf,
    setup_logging,
)


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


ANSWER_SYSTEM_PROMPT = """You are InquiryBot, a careful business document assistant.

Answer the user's question using only the provided context. If the context does not
contain the answer, reply with the configured refusal message.

When the documents describe iSparx or its business profile, tailor the answer to
business information such as services, portfolio, contact details, process, and
commercial intent. Keep the answer concise and useful."""


def complete_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def answer_question(question: str, model: str) -> tuple[str, list[dict]]:
    vector_store = get_vector_store()
    retrieved = retrieve_relevant_documents(
        question,
        vector_store=vector_store,
        k=RETRIEVAL_K,
        score_threshold=SIMILARITY_SCORE_THRESHOLD,
    )

    if not retrieved:
        return REFUSAL_MESSAGE, []

    docs = [doc for doc, _score in retrieved]
    citations = [citation_payload(doc, score) for doc, score in retrieved]
    user_prompt = (
        f"Refusal message:\n{REFUSAL_MESSAGE}\n\n"
        f"Context:\n{format_documents_for_prompt(docs)}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    answer = complete_chat(model, ANSWER_SYSTEM_PROMPT, user_prompt)
    return answer, citations


def generate_email_draft(
    question: str,
    answer: str,
    model: str,
    name: str,
    email: str,
    intent: str,
) -> str:
    user_prompt = (
        "Draft a concise professional email based on the user's inquiry and the "
        "document-grounded answer.\n\n"
        f"Lead name: {name or 'Prospective customer'}\n"
        f"Lead email: {email or 'Not provided'}\n"
        f"Intent: {intent or 'General inquiry'}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Email draft:"
    )
    return complete_chat(model, "You write clear business emails.", user_prompt)


def display_citations(citations: list[dict]) -> None:
    if not citations:
        return

    st.markdown("**Sources**")
    for index, citation in enumerate(citations, start=1):
        label = f"{index}. {citation['document_name']}"
        if citation.get("page_number"):
            label += f" - page {citation['page_number']}"
        if citation.get("score") is not None:
            label += f" - relevance {citation['score']:.2f}"

        with st.expander(label):
            st.write(citation["chunk_text"])


st.set_page_config(page_title="InquiryBot RAG", layout="wide")
st.title("InquiryBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_ingested" not in st.session_state:
    st.session_state.uploaded_files_ingested = set()


with st.sidebar:
    st.header("Settings")

    selected_model = st.selectbox(
        "OpenAI chat model",
        CHAT_MODEL_CHOICES,
        index=CHAT_MODEL_CHOICES.index(DEFAULT_CHAT_MODEL)
        if DEFAULT_CHAT_MODEL in CHAT_MODEL_CHOICES
        else 0,
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            key = f"{uploaded_file.name}:{uploaded_file.size}"
            if key in st.session_state.uploaded_files_ingested:
                continue

            try:
                saved_path = save_uploaded_pdf(uploaded_file)
                with st.spinner(f"Ingesting {saved_path.name}..."):
                    chunk_count = ingest_pdf_paths([saved_path], replace_existing=True)
                st.session_state.uploaded_files_ingested.add(key)
                st.success(f"Ingested {chunk_count} chunks from {saved_path.name}")
            except Exception as exc:
                logger.exception("Upload ingestion failed")
                st.error(
                    f"Could not ingest {uploaded_file.name}. The document was saved, "
                    "but embedding/indexing failed. Check logs/app.log for details."
                )
                st.caption(str(exc))

    if st.button("Ingest PDFs from data/"):
        try:
            pdfs = list_pdf_files(DATA_PATH)
            if not pdfs:
                st.warning("No PDF files found in data/.")
            else:
                with st.spinner("Ingesting local PDFs..."):
                    chunk_count = ingest_pdf_paths(pdfs, replace_existing=True)
                st.success(f"Ingested {chunk_count} chunks from {len(pdfs)} PDF(s).")
        except Exception as exc:
            logger.exception("Local ingestion failed")
            st.error(
                "Could not ingest local PDFs. The app now inserts chunks in batches, "
                "so if this continues, check the PDF text extraction and OpenAI API "
                "error details in logs/app.log."
            )
            st.caption(str(exc))

    if st.button("Clear and rebuild vector database"):
        try:
            with st.spinner("Rebuilding Chroma database from data/..."):
                chunk_count = rebuild_vector_database()
            st.success(f"Rebuilt vector database with {chunk_count} chunks.")
        except Exception as exc:
            logger.exception("Vector database rebuild failed")
            st.error(
                "Could not rebuild the vector database safely. The app clears the "
                "Chroma collection without deleting chroma_db files; if this continues, "
                "restart the Streamlit app to release any remaining Chroma handles."
            )
            st.caption(str(exc))

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Lead capture")
    lead_name = st.text_input("Name")
    lead_email = st.text_input("Email")
    lead_intent = st.selectbox(
        "Intent",
        ["General inquiry", "Services", "Portfolio", "Pricing", "Contact", "Support"],
    )
    create_email_draft = st.checkbox("Generate email draft for each answer")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            display_citations(message["citations"])
        if message.get("email_draft"):
            st.markdown("**Email draft**")
            st.code(message["email_draft"], language="markdown")


if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                answer, citations = answer_question(question, selected_model)

            st.markdown(answer)
            display_citations(citations)

            email_draft = None
            if create_email_draft and answer != REFUSAL_MESSAGE:
                with st.spinner("Drafting email..."):
                    email_draft = generate_email_draft(
                        question=question,
                        answer=answer,
                        model=selected_model,
                        name=lead_name,
                        email=lead_email,
                        intent=lead_intent,
                    )
                st.markdown("**Email draft**")
                st.code(email_draft, language="markdown")

            append_lead(lead_name, lead_email, lead_intent, question)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "email_draft": email_draft,
                }
            )
        except Exception as exc:
            logger.exception("Query failed")
            st.error(f"Could not answer the question: {exc}")
            st.error("Check your OPENAI_API_KEY, model access, and Chroma database.")
