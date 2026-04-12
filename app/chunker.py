# =========================
# SMART DYNAMIC CHUNKING
# =========================

def dynamic_chunk_params(text_length):

    # Very small documents

    if text_length < 1000:

        return text_length, 0


    # Small documents

    elif text_length < 5000:

        return 300, 50


    # Medium documents

    elif text_length < 20000:

        return 500, 75


    # Large documents

    else:

        return 800, 100


def chunk_text(text):

    if not text:

        return []


    text_length = len(text)


    chunk_size, overlap = dynamic_chunk_params(
        text_length
    )


    print(
        f"Dynamic chunk_size={chunk_size}, overlap={overlap}"
    )


    chunks = []

    start = 0


    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)


        if overlap == 0:

            break


        start += (
            chunk_size - overlap
        )


    return chunks
