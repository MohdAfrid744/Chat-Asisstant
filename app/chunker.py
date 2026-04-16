import re


# =========================
# HEADING-AWARE CHUNKING
# =========================

def split_by_headings(text):

    """
    Split markdown/text by headings.

    Supports:
    # Heading
    ## Heading
    ### Heading
    """

    pattern = r"(#+\s.*)"

    parts = re.split(pattern, text)

    chunks = []

    current_chunk = ""

    for part in parts:

        if re.match(pattern, part):

            if current_chunk.strip():

                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = part

        else:

            current_chunk += "\n" + part


    if current_chunk.strip():

        chunks.append(
            current_chunk.strip()
        )

    return chunks


# =========================
# DYNAMIC FALLBACK CHUNKING
# =========================

def fallback_chunking(
    text,
    chunk_size=500,
    overlap=80
):

    chunks = []

    start = 0

    length = len(text)

    while start < length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# MAIN CHUNK FUNCTION
# =========================

def chunk_text(text):

    """
    Smart chunking:

    1. Try heading-aware
    2. Fallback to dynamic
    """

    heading_chunks = split_by_headings(text)

    if len(heading_chunks) > 1:

        print("Using heading-aware chunking")

        return heading_chunks


    print("Using fallback chunking")

    return fallback_chunking(text)
