
"""SupportAI - Customer Support RAG Assistant."""

import os
from typing import Any, Dict

import requests
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SupportAI | Customer Support",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud: configure API_BASE_URL in App Settings > Secrets.
# Local development: falls back to localhost.
try:
    secrets_api_url = st.secrets.get("API_BASE_URL", "")
except Exception:
    secrets_api_url = ""

API_BASE_URL = (
    secrets_api_url
    or os.getenv("API_BASE_URL", "http://localhost:8000")
).rstrip("/")

HEALTH_URL = f"{API_BASE_URL}/api/health"
QUERY_URL = f"{API_BASE_URL}/api/query"
PROCESS_URL = f"{API_BASE_URL}/api/process_document"
st.sidebar.caption(f"Backend URL: {API_BASE_URL}")

# ============================================================
# 2. PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: #f7f8fc !important;
        color: #26334d !important;
    }

    [data-testid="stHeader"] {
        background: #f7f8fc !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e8ebf3;
    }

    [data-testid="stSidebar"] * {
        color: #34415c;
    }

    .brand {
        font-size: 29px;
        font-weight: 800;
        color: #283a78 !important;
        letter-spacing: -1px;
    }

    .brand span {
        color: #7164dc !important;
    }

    .brand-caption {
        color: #929bb0 !important;
        font-size: 10px;
        letter-spacing: 1.3px;
        margin-bottom: 25px;
    }

    .hero-box {
        background: linear-gradient(120deg, #edf0ff, #f4efff, #faf5ff);
        border: 1px solid #e3e6ff;
        border-radius: 20px;
        padding: 32px;
        margin: 8px 0 27px;
    }

    .hero-kicker {
        color: #6258c9 !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }

    .hero-heading {
        color: #26366f !important;
        font-size: 31px;
        font-weight: 800;
        margin: 10px 0;
        line-height: 1.3;
    }

    .hero-description {
        color: #657491 !important;
        font-size: 14px;
        line-height: 1.7;
    }

    h1, h2, h3 {
        color: #26366f !important;
        font-weight: 750 !important;
    }

    p, label {
        color: #34415c;
    }

    .muted {
        color: #8290a8 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .online {
        display: inline-block;
        color: #16734b !important;
        background: #e5f7ee;
        padding: 7px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .offline {
        display: inline-block;
        color: #b54747 !important;
        background: #fff0ef;
        padding: 7px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    div.stButton > button {
        background: #ffffff !important;
        color: #34415c !important;
        border: 1px solid #dfe4ef !important;
        border-radius: 11px !important;
        min-height: 44px !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        background: #efefff !important;
        color: #4941b6 !important;
        border-color: #8178e4 !important;
    }

    div.stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    div[data-testid="stFormSubmitButton"] button {
        background: #5952ca !important;
        color: #ffffff !important;
        border: 1px solid #5952ca !important;
        border-radius: 11px !important;
        min-height: 46px !important;
        font-weight: 700 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #443bb5 !important;
        color: #ffffff !important;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #26334d !important;
        border: 1px solid #dce2ee !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        caret-color: #514bc8 !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #929bb0 !important;
        opacity: 1 !important;
    }

    button[data-baseweb="tab"] {
        color: #596780 !important;
        font-weight: 650 !important;
        background: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #5148c5 !important;
        border-bottom: 3px solid #655bd5 !important;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #aebbe4 !important;
        border-radius: 15px !important;
        padding: 16px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: #34415c !important;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e6eaf2 !important;
        border-radius: 15px !important;
        padding: 17px !important;
        margin-bottom: 13px !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
        color: #26334d !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessage"] li {
        margin-bottom: 8px !important;
        line-height: 1.7 !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e7eaf2 !important;
        border-radius: 12px !important;
    }

    .footer {
        text-align: center;
        color: #929bb0 !important;
        font-size: 12px;
        padding: 12px;
    }

    hr {
        border-color: #e8ebf3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. API FUNCTIONS
# ============================================================

def check_api() -> bool:
    """Check whether the FastAPI backend is available."""
    try:
        response = requests.get(HEALTH_URL, timeout=8)
        return response.ok
    except requests.RequestException:
        return False


def query_rag(query: str, k: int) -> Dict[str, Any]:
    """Send a question to the RAG backend."""
    try:
        response = requests.post(
            QUERY_URL,
            json={
                "query": query,
                "k": k,
            },
            timeout=120,
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        return {"error": str(exc)}


def process_document(uploaded_file) -> Dict[str, Any]:
    """Upload a document directly to the FastAPI backend."""
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }

        response = requests.post(
            PROCESS_URL,
            files=files,
            timeout=300,
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if response.ok:
            return {
                "success": True,
                "data": data,
            }

        return {
            "success": False,
            "error": data.get(
                "detail",
                f"Backend returned HTTP {response.status_code}",
            ),
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Document processing timed out. Please try again.",
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# 4. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">Support<span>AI</span></div>
        <div class="brand-caption">DOCUMENT INTELLIGENCE PLATFORM</div>
        """,
        unsafe_allow_html=True,
    )

    api_connected = check_api()

    if api_connected:
        st.markdown(
            '<span class="online">● API Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="offline">● API Disconnected</span>',
            unsafe_allow_html=True,
        )
        st.caption("Check your FastAPI backend URL and service.")

    if st.button("Refresh connection", use_container_width=True):
        st.rerun()

    st.divider()

    st.markdown("**RETRIEVAL SETTINGS**")

    retrieval_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=False,
    )

    show_validation = st.checkbox(
        "Show validation details",
        value=False,
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.caption("Powered by LangGraph · FAISS · Gemini")


# ============================================================
# 6. HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-kicker">
            YOUR KNOWLEDGE, ONE CONVERSATION AWAY
        </div>
        <div class="hero-heading">
            Customer Support Assistant
        </div>
        <div class="hero-description">
            Ask questions about your support documents.
            Get clear answers grounded in your knowledge base.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. MAIN TABS
# ============================================================

chat_tab, knowledge_tab = st.tabs(
    ["💬 Ask Assistant", "📁 Knowledge Base"]
)


# ============================================================
# 8. CHAT ASSISTANT
# ============================================================

with chat_tab:

    st.subheader("How can I help you today?")

    st.markdown(
        '<div class="muted">'
        'Ask anything about your uploaded customer-support documents.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("**Suggested questions**")

    suggestions = [
        "What is the refund policy?",
        "How can I reset my password?",
        "How do I contact customer support?",
    ]

    columns = st.columns(3)

    for index, suggestion in enumerate(suggestions):
        with columns[index]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion

    st.write("")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Question input
    with st.form("question_form", clear_on_submit=True):

        user_question = st.text_area(
            "Your question",
            placeholder="Type your question here...",
            height=100,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Ask Assistant  →",
            type="primary",
            use_container_width=True,
        )

    question_to_process = None

    if st.session_state.pending_question:
        question_to_process = st.session_state.pending_question
        st.session_state.pending_question = None

    elif submitted:
        question_to_process = user_question.strip()

    # Process question
    if question_to_process:

        if not question_to_process.strip():
            st.warning("Please enter a question.")

        elif not api_connected:
            st.error(
                "FastAPI is disconnected. "
                "Please check the backend service."
            )

        else:

            st.session_state.messages.append({
                "role": "user",
                "content": question_to_process,
            })

            with st.spinner("Searching your knowledge base..."):

                result = query_rag(
                    question_to_process,
                    retrieval_k,
                )

            if result.get("error"):

                st.session_state.messages.pop()

                st.error(
                    "Unable to generate an answer: "
                    + result["error"]
                )

            else:

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer.",
                )

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(item) for item in answer
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(answer),
                })

                st.session_state.last_result = result

            st.rerun()

    # Retrieved sources
    result = st.session_state.last_result

    if result and show_sources:

        st.divider()
        st.markdown("**Retrieved Sources**")

        chunks = result.get("retrieved_chunks", [])

        if chunks:

            for index, chunk in enumerate(chunks, start=1):

                metadata = chunk.get("metadata") or {}

                filename = metadata.get(
                    "file_name",
                    metadata.get("source", "Document"),
                )

                page = metadata.get(
                    "page",
                    metadata.get("page_number", "N/A"),
                )

                score = chunk.get("similarity_score")

                with st.expander(
                    f"Source {index} · {filename} · Page {page}"
                ):

                    if isinstance(score, (int, float)):
                        st.caption(
                            f"Retrieval score: {score:.4f}"
                        )

                    st.markdown(
                        chunk.get("content", "")
                    )

        else:
            st.info("No sources were returned.")

    # Validation details
    if result and show_validation:

        st.divider()
        st.markdown("**Answer Validation**")

        confidence = result.get("confidence_score")

        if isinstance(confidence, (int, float)):
            st.metric(
                "Validation score",
                f"{confidence:.2%}",
            )

        validation = result.get("validation", {})

        if validation:
            st.json(validation)
        else:
            st.info("No validation details were returned.")


# ============================================================
# 9. KNOWLEDGE BASE / DOCUMENT UPLOAD
# ============================================================

with knowledge_tab:

    st.subheader("Manage your documents")

    st.markdown(
        '<div class="muted">'
        'Upload PDF or DOCX documents to expand your assistant knowledge.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # The uploader remains enabled even if the API is offline.
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX document",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        help="Supported formats: PDF and DOCX.",
        key="document_uploader",
    )

    if uploaded_file:

        content = uploaded_file.getvalue()

        file_col, size_col = st.columns([3, 1])

        with file_col:
            st.markdown("**Selected document**")
            st.write(uploaded_file.name)

        with size_col:
            st.markdown("**File size**")
            st.write(f"{len(content) / 1024:.2f} KB")

        if not api_connected:

            st.warning(
                "Your document is selected, but the backend is "
                "currently unavailable. Try again when the API is online."
            )

        else:

            if st.button(
                "Upload and process document  →",
                type="primary",
                use_container_width=True,
                key="process_document_button",
            ):

                with st.spinner(
                    "Uploading, extracting, chunking and indexing your document..."
                ):

                    result = process_document(uploaded_file)

                if result["success"]:

                    message = result["data"].get(
                        "message",
                        "Document processed successfully.",
                    )

                    st.success(message)
                    st.balloons()

                else:

                    st.error(
                        "Document processing failed: "
                        + str(result["error"])
                    )

    else:

        st.info(
            "Select a PDF or DOCX file above to get started."
        )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        SupportAI · Intelligent Customer Support
        <br>
        Built with Streamlit, LangGraph, FAISS and Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
