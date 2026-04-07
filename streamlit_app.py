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


if "uploaded_file_name" not in st.session_state:

    st.session_state.uploaded_file_name = None


# =========================
# SIDEBAR
# =========================

st.sidebar.title("📄 Upload Document")


uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# Upload only if new file

if uploaded_file:

    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        files = {
            "file": uploaded_file
        }

        with st.spinner(
            "Uploading document..."
        ):

            response = requests.post(
                f"{API_URL}/ingest",
                files=files
            )

        if response.status_code == 200:

            st.sidebar.success(
                response.json()["message"]
            )

            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )

            # Reset chat for new doc

            st.session_state.messages = []


# =========================
# NEW CHAT BUTTON
# =========================

if st.sidebar.button("🧹 New Chat"):

    st.session_state.messages = []

    st.sidebar.success(
        "Chat cleared!"
    )


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

if prompt := st.chat_input(
    "Ask your question..."
):

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    with st.chat_message("user"):

        st.markdown(prompt)


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


        except Exception:

            full_response = (
                "❌ Error occurred."
            )

            message_placeholder.markdown(
                full_response
            )


    # Save assistant message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )