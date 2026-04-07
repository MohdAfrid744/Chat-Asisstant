import hashlib
import os


HASH_FILE = "data/doc_hash.txt"


# =========================
# FILE HASHING (SAFE)
# =========================

def calculate_file_hash(file_path):

    hasher = hashlib.md5()

    with open(file_path, "rb") as f:

        # Read file in chunks

        while chunk := f.read(4096):

            hasher.update(chunk)

    return hasher.hexdigest()


# =========================
# CHECK IF NEW DOCUMENT
# =========================

def is_new_document(file_path):

    os.makedirs(
        "data",
        exist_ok=True
    )

    new_hash = calculate_file_hash(
        file_path
    )


    if os.path.exists(HASH_FILE):

        with open(
            HASH_FILE,
            "r"
        ) as f:

            old_hash = f.read()


        if old_hash == new_hash:

            return False


    with open(
        HASH_FILE,
        "w"
    ) as f:

        f.write(new_hash)


    return True