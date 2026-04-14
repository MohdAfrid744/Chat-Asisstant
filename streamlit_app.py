# streamlit_app.py

import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAG Chat Assistant",
    layout="wide"
)

st.title("🤖 RAG Chat Assistant")


# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "status_message" not in st.session_state:
    st.session_state.status_message = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# =========================
# FILE UPLOAD
# =========================

st.sidebar.title("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload File",
    type=["pdf", "txt", "md"]
)


# Only upload once per file

if (
    uploaded_file
    and uploaded_file.name
    != st.session_state.uploaded_filename
):

    files = {"file": uploaded_file}

    with st.spinner("Processing document..."):

        try:

            response = requests.post(
                f"{API_URL}/ingest",
                files=files
            )

            if response.status_code == 200:

                data = response.json()

                time_taken = data.get(
                    "processing_time",
                    0
                )

                st.session_state.status_message = (
                    f"✅ Document ready "
                    f"(Processed in {time_taken}s)"
                )

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )

            else:

                st.session_state.status_message = (
                    "⚠️ Upload failed."
                )

        except Exception:

            st.session_state.status_message = (
                "⚠️ Backend not reachable."
            )


# =========================
# STATUS MESSAGE
# =========================

if st.session_state.status_message:

    st.success(
        st.session_state.status_message
    )


# =========================
# NEW CHAT BUTTON
# =========================

if st.sidebar.button("🆕 New Chat"):

    st.session_state.messages = []

    st.session_state.status_message = None


# =========================
# DISPLAY CHAT
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

prompt = st.chat_input(
    "Ask your question..."
)


if prompt:

    # Remove message immediately

    st.session_state.status_message = None


    # Add user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    with st.chat_message("user"):

        st.markdown(prompt)


    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            response = requests.get(
                f"{API_URL}/query",
                params={
                    "query": prompt
                }
            )


            if response.status_code == 200:

                try:

                    result = response.json()

                    full_response = (
                        f"{result.get('source','')}\n\n"
                        f"{result.get('response','')}"
                    )

                except Exception:

                    full_response = (
                        "⚠️ Invalid response format."
                    )

            else:

                full_response = (
                    "⚠️ Query failed."
                )

        except Exception:

            full_response = (
                "⚠️ Backend not reachable."
            )


        placeholder.markdown(
            full_response
        )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )
