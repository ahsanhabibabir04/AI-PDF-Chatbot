
import hashlib

import ollama
import pymupdf
import streamlit as st

from retrieval import (
    split_text,
    build_search_index,
    search_chunks,
)

MODEL_NAME = "qwen2.5:3b"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-weight: 750;
            letter-spacing: -1px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        [data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        .hero-card {
            padding: 1.4rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 18px;
            margin-bottom: 1.5rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.8;
            margin-bottom: 0;
        }

        .footer-text {
            font-size: 0.85rem;
            opacity: 0.75;
        }

        .status-note {
            font-size: 0.9rem;
            opacity: 0.85;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

defaults = {
    "messages": [],
    "pdf_key": None,
    "pending_question": None,
    "pdf_chunks": [],
    "search_model": None,
    "search_index": None,
    "pdf_name": None,
    "pdf_pages": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📚 AI PDF Chatbot")
    st.caption("Your personal document assistant")

    st.divider()
    st.subheader("📄 Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Choose a text-based PDF to start chatting.",
    )

    if uploaded_file is not None:
        st.info(
            f"Selected file: **{uploaded_file.name}**"
        )
        st.caption(
            f"File size: {uploaded_file.size / 1024:.1f} KB"
        )

    st.divider()
    st.subheader("🤖 AI Information")

    st.markdown(
        f"""
        **Language model**

        `{MODEL_NAME}`

        **Embedding model**

        `all-MiniLM-L6-v2`

        **Search engine**

        FAISS semantic search
        """
    )

    st.divider()
    st.caption(
        "AI answers are generated using your local Ollama model."
    )
    st.caption("Scanned PDFs may require OCR.")


# --------------------------------------------------
# Main header
# --------------------------------------------------

st.title("Chat with your PDF")

st.markdown(
    """
    Ask questions, explore key information, and get answers
    grounded in your document with PDF page references.
    """
)

st.divider()


# --------------------------------------------------
# Welcome screen
# --------------------------------------------------

if uploaded_file is None:
    st.markdown(
        """
        <div class="hero-card">
            <h3>👋 Welcome to your AI document assistant</h3>
            <p class="hero-subtitle">
                Upload a PDF using the sidebar to get started.
                Then ask questions in natural language.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔎 Semantic Search")
        st.caption(
            "Find relevant passages based on meaning, "
            "not just exact keywords."
        )

    with col2:
        st.markdown("### 💬 Follow-up Chat")
        st.caption(
            "Ask follow-up questions using recent "
            "conversation context."
        )

    with col3:
        st.markdown("### 📌 Page Citations")
        st.caption(
            "Inspect the PDF passages used to generate "
            "each answer."
        )


# --------------------------------------------------
# PDF processing and chat
# --------------------------------------------------

else:
    pdf_bytes = uploaded_file.getvalue()
    current_pdf_key = hashlib.sha256(pdf_bytes).hexdigest()

    # Process a new or changed PDF
    if st.session_state.pdf_key != current_pdf_key:
        # Reset old document state before processing
        st.session_state.pdf_key = None
        st.session_state.pdf_chunks = []
        st.session_state.search_model = None
        st.session_state.search_index = None
        st.session_state.pdf_name = None
        st.session_state.pdf_pages = 0
        st.session_state.messages = []
        st.session_state.pending_question = None

        try:
            with st.status(
                "Processing your PDF...",
                expanded=True,
            ) as status:
                st.write("📄 Reading PDF pages and extracting text...")

                with pymupdf.open(
                    stream=pdf_bytes,
                    filetype="pdf",
                ) as pdf:
                    total_pdf_pages = len(pdf)
                    page_texts = []

                    progress = st.progress(0)

                    for page_number, page in enumerate(
                        pdf,
                        start=1,
                    ):
                        page_text = page.get_text().strip()

                        if page_text:
                            page_texts.append(
                                {
                                    "page": page_number,
                                    "text": page_text,
                                }
                            )

                        progress.progress(
                            page_number / max(total_pdf_pages, 1)
                        )

                progress.empty()

                if not page_texts:
                    status.update(
                        label="No extractable text found",
                        state="error",
                        expanded=True,
                    )
                    st.warning(
                        "This PDF contains no extractable text. "
                        "It may be scanned or image-only. "
                        "OCR support is needed for this type of PDF."
                    )

                else:
                    st.write("✂️ Splitting document into searchable chunks...")

                    chunks = []

                    for item in page_texts:
                        for chunk_text in split_text(item["text"]):
                            chunks.append(
                                {
                                    "page": item["page"],
                                    "text": chunk_text,
                                }
                            )

                    if not chunks:
                        status.update(
                            label="Could not create text chunks",
                            state="error",
                        )
                        st.warning(
                            "Text was extracted, but no searchable "
                            "chunks could be created."
                        )

                    else:
                        st.write(
                            f"🔎 Building semantic index for "
                            f"{len(chunks)} text chunks..."
                        )

                        model, index, _ = build_search_index(
                            [chunk["text"] for chunk in chunks]
                        )

                        st.session_state.pdf_key = current_pdf_key
                        st.session_state.pdf_chunks = chunks
                        st.session_state.search_model = model
                        st.session_state.search_index = index
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.pdf_pages = len(page_texts)

                        status.update(
                            label="PDF processed successfully!",
                            state="complete",
                            expanded=False,
                        )

        except Exception as error:
            st.error(
                "Could not process this PDF. "
                f"Details: {error}"
            )

    # Continue only if processing succeeded for this PDF
    if st.session_state.pdf_key == current_pdf_key:
        chunks = st.session_state.pdf_chunks
        model = st.session_state.search_model
        index = st.session_state.search_index

        st.success(
            f"✅ **{st.session_state.pdf_name}** is ready to use."
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("PDF Text Pages", st.session_state.pdf_pages)

        with metric2:
            st.metric("Text Chunks", len(chunks))

        with metric3:
            st.metric(
                "Chat Messages",
                len(st.session_state.messages),
            )

        with st.expander("📄 Document information"):
            st.write(f"**Filename:** {st.session_state.pdf_name}")
            st.write(f"**Text pages:** {st.session_state.pdf_pages}")
            st.write(f"**Searchable chunks:** {len(chunks)}")

        # Suggested questions
        st.markdown("#### 💡 Try asking")

        suggested_questions = [
            "What is the main topic of this document?",
            "Summarize the key points.",
            "What are the most important findings?",
        ]

        suggestion_columns = st.columns(3)

        for column, suggested_question in zip(
            suggestion_columns,
            suggested_questions,
        ):
            with column:
                if st.button(
                    suggested_question,
                    use_container_width=True,
                    key=f"suggestion_{suggested_question}",
                ):
                    st.session_state.pending_question = suggested_question
                    st.rerun()

        # Chat controls
        control_col1, control_col2 = st.columns([5, 1])

        with control_col1:
            st.subheader("💬 Conversation")

        with control_col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.pending_question = None
                st.rerun()

        # Display previous conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📌 View source passages"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**PDF page {source['page']}**"
                            )
                            st.write(source["text"])

        # Read a typed or suggested question
        typed_question = st.chat_input(
            "Ask anything about your PDF..."
        )

        question = (
            typed_question
            or st.session_state.pending_question
        )

        if question:
            st.session_state.pending_question = None

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.status(
                        "Finding an answer...",
                        expanded=True,
                    ) as answer_status:
                        answer_status.write(
                            "🔎 Searching relevant PDF passages..."
                        )

                        results = search_chunks(
                            question,
                            model,
                            index,
                            [chunk["text"] for chunk in chunks],
                            top_k=3,
                        )

                        sources = []

                        for result in results:
                            matching_pages = sorted(
                                {
                                    item["page"]
                                    for item in chunks
                                    if item["text"] == result["text"]
                                }
                            )

                            page_number = (
                                matching_pages[0]
                                if matching_pages
                                else "Unknown"
                            )

                            sources.append(
                                {
                                    "page": page_number,
                                    "text": result["text"],
                                }
                            )

                        if not sources:
                            answer = (
                                "I could not find relevant information "
                                "in the provided PDF text."
                            )

                        else:
                            context = "\n\n".join(
                                (
                                    f"[PDF page {source['page']}]\n"
                                    f"{source['text']}"
                                )
                                for source in sources
                            )

                            recent_history = (
                                st.session_state.messages[-7:-1]
                            )

                            history_text = "\n".join(
                                (
                                    f"{item['role'].upper()}: "
                                    f"{item['content']}"
                                )
                                for item in recent_history
                            )

                            prompt = f"""
You are a careful PDF question-answering assistant.

RULES:
1. Use the supplied PDF passages as the authority for factual answers.
2. Use recent conversation only to understand follow-up questions.
3. Do not invent facts or unsupported conclusions.
4. If the passages do not provide enough information, say so clearly.
5. Cite supporting page numbers as [Page N].
6. Keep the answer clear and well structured.

RECENT CONVERSATION:
{history_text or "No previous conversation."}

PDF SOURCE TEXT:
{context}

CURRENT QUESTION:
{question}

Write your answer below.
"""

                            answer_status.write(
                                "🤖 Generating an answer with Ollama..."
                            )

                            response = ollama.chat(
                                model=MODEL_NAME,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt,
                                    }
                                ],
                            )

                            answer = response["message"]["content"]

                        answer_status.update(
                            label="Answer ready!",
                            state="complete",
                            expanded=False,
                        )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📌 View source passages"):
                            for source in sources:
                                st.markdown(
                                    f"**PDF page {source['page']}**"
                                )
                                st.write(source["text"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    st.error(
                        "Could not generate an answer. "
                        "Check that Ollama is running and the model "
                        f"`{MODEL_NAME}` is available. Details: {error}"
                    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption("📚 AI PDF Chatbot")
    st.caption("Built with Streamlit, FAISS and Ollama")

with footer_col2:
    st.caption("Educational project")
    st.caption("AI-generated answers may contain errors.")
