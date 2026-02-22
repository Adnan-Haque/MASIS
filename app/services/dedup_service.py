import hashlib

def compute_hash(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()