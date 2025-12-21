import re


def clean_description(text: str) -> str:
    """
    Normalize and clean incident narratives for TF–IDF:
    - lowercase
    - normalize cloud storage terms (google drive, gdrive, box, dropbox)
    - normalize URLs, IPs, file paths, and long encoded blobs
    - keep only alphanumeric characters and spaces
    - drop standalone numbers
    """

    # Lowercase
    text = text.lower()

    # Normalize common cloud storage terms
    text = re.sub(r"\bgoogle\s+drive\b", " googledrive ", text)
    text = re.sub(r"\bgdrive\b", " googledrive ", text)
    text = re.sub(r"\bbox\.com\b", " box ", text)
    text = re.sub(r"\bdropbox\b", " dropbox ", text)

    # Normalize URLs
    text = re.sub(r"http\S+|www\.\S+", " url ", text)

    # Normalize IPv4 addresses
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ipaddr ", text)

    # Normalize Windows-style file paths (e.g., C:\Users\...)
    text = re.sub(r"[a-z]:\\[^\s]+", " filepath ", text)

    # Normalize Unix-style paths (e.g., /var/www/html/index.php)
    text = re.sub(r"/[^\s]+", " filepath ", text)

    # Normalize long base64 / encoded-looking blobs
    text = re.sub(r"[a-z0-9+/=]{20,}", " encoded ", text)

    # Strip non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # Drop standalone numbers (years, ids, etc.)
    text = re.sub(r"\b\d+\b", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
