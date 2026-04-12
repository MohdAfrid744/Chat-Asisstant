from collections import deque

# =========================
# CONFIG
# =========================

MAX_MEMORY = 5

# Stores last N conversations
chat_memory = deque(maxlen=MAX_MEMORY)


# =========================
# ADD MEMORY
# =========================

def add_to_memory(query, response):

    # Prevent empty entries

    if not query or not response:
        return

    chat_memory.append({
        "query": str(query),
        "response": str(response)
    })


# =========================
# GET MEMORY
# =========================

def get_memory():

    return list(chat_memory)


# =========================
# CLEAR MEMORY
# =========================

def clear_memory():

    chat_memory.clear()

    print("🧠 Memory cleared.")
