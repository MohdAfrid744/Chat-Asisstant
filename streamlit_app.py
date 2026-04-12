import streamlit as st
import requests

# =========================
# CONFIG
# =========================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Chat Assistant",
    layout="wide"
)

st.title("🤖 RAG Chat Assistant")

# =========================
# SESSION STATE INIT
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False


# =========================
# SIDEBAR
# =========================

st.sidebar.title("📄 Document Panel")


# -------------------------
# New Chat Button
# -------------------------

if st.sidebar.button("🆕 New Chat"):

    st.session_state.messages = []

    try:
        requests.get(
            f"{API_URL}/clear_memory"
        )
    except:
        pass

    st.sidebar.success(
        "New chat started."
    )


# -------------------------
# File Upload
# -------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    files = {
        "file": uploaded_file
    }

    with st.spinner(
        "Processing document..."
    ):

        response = requests.post(
            f"{API_URL}/ingest",
            files=files
        )

    if response.status_code == 200:

        st.session_state.uploaded = True

        st.sidebar.success(
            response.json()["message"]
        )

    else:

        st.sidebar.error(
            "Upload failed."
        )


# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================
# USER INPUT
# =========================

if prompt := st.chat_input(
    "Ask your question..."
):

    if not st.session_state.uploaded:

        st.warning(
            "Please upload a document first."
        )

    else:

        # Add user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)


        # Assistant response

        with st.chat_message("assistant"):

            message_placeholder = st.empty()

            full_response = ""

            try:

                response = requests.get(
                    f"{API_URL}/query",
                    params={
                        "query": prompt
                    },
                    stream=True
                )

                for chunk in response.iter_content(
                    chunk_size=1024
                ):

                    if chunk:

                        text = chunk.decode(
                            "utf-8"
                        )

                        full_response += text

                        message_placeholder.markdown(
                            full_response + "▌"
                        )

                message_placeholder.markdown(
                    full_response
                )

            except Exception as e:

                full_response = (
                    "⚠️ Error occurred."
                )

                message_placeholder.markdown(
                    full_response
                )


        # Save assistant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response
            }
        )
