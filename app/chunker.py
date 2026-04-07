def chunk_text(
    text,
    chunk_size=500,
    overlap=50
):

    # Safety check

    if not text:

        return []


    chunks = []

    start = 0

    text_length = len(text)


    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]


        # Clean chunk

        chunk = chunk.strip()


        if chunk:

            chunks.append(chunk)


        start += (
            chunk_size - overlap
        )


    return chunks