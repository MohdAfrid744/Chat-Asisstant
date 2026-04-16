import streamlit as st
import requests
import os


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

if "doc_ready" not in st.session_state:

    st.session_state.doc_ready = False


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📂 Controls")


    # NEW CHAT

    if st.button("🆕 New Chat"):

        try:

            requests.post(
                "http://localhost:8000/clear"
            )

        except:
            pass

        st.session_state.messages = []

        st.session_state.doc_ready = False

        st.rerun()


    st.divider()


    # FILE UPLOAD

    uploaded_file = st.file_uploader(

        "📄 Upload Document",

        type=["pdf", "txt", "md"]

    )


    if uploaded_file:

        os.makedirs("data", exist_ok=True)

        file_path = f"data/{uploaded_file.name}"

        with open(file_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )


        with st.spinner("Processing..."):

            try:

                response = requests.post(

                    "http://localhost:8000/ingest",

                    json={

                        "file_path": file_path

                    },

                    timeout=600

                )

                result = response.json()


                if result["status"] == "duplicate":

                    st.warning(

                        "⚠️ Document already processed."

                    )

                    st.session_state.doc_ready = True


                else:

                    st.success(

                        f"✅ Document ready "
                        f"(Processed in {result['time']}s)"

                    )

                    st.session_state.doc_ready = True


            except:

                st.error(

                    "⚠️ Failed to process document."

                )


    st.divider()


    # STATUS

    if st.session_state.doc_ready:

        st.success("✅ Ready to Query")

    else:

        st.warning("⚠️ Upload a document first")


# =========================
# CHAT DISPLAY
# =========================

for msg in st.session_state.messages:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )


# =========================
# CHAT INPUT
# =========================

query = st.chat_input(
    "Ask your question..."
)


if query:

    st.chat_message("user").write(query)

    st.session_state.messages.append({

        "role": "user",

        "content": query

    })


    try:

        response = requests.get(

            "http://localhost:8000/query",

            params={"query": query},

            timeout=300

        )

        result = response.json()

        answer = result.get(

            "response",

            "No response."

        )

    except:

        answer = "⚠️ Backend error."


    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })
