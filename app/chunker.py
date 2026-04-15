def chunk_text(
    text,
    chunk_size,
    overlap
):

    """
    Splits text into overlapping chunks.

    Parameters:
    text : str
    chunk_size : int
    overlap : int

    Returns:
    List[str]
    """

    chunks = []

    start = 0

    text_length = len(text)

    step = chunk_size - overlap

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += step

    return chunks
