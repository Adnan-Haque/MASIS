from loader import extract_text_stream, normalize_content

with open("sample.json", "rb") as f:
    file_bytes = f.read()

for raw_text in extract_text_stream("sample.json", file_bytes):
    print("RAW TEXT:")
    print(raw_text)
    print("-----")

    normalized = normalize_content(raw_text)

    print("NORMALIZED CHUNKS:")
    for chunk in normalized:
        print(chunk)
        print("-----")