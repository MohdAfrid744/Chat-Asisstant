from collections import deque


# =========================
# CHAT MEMORY
# =========================

# Store last N messages

chat_memory = deque(
    maxlen=10
)


# =========================
# ADD MEMORY
# =========================

def add_to_memory(
    query,
    response
):

    chat_memory.append(
        {
            "role": "user",
            "content": query
        }
    )

    chat_memory.append(
        {
            "role": "assistant",
            "content": response
        }
    )


# =========================
# GET MEMORY
# =========================

def get_memory():

    return list(chat_memory)


# =========================
# CLEAR MEMORY (Optional)
# =========================

def clear_memory():

    chat_memory.clear()