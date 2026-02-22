from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

def smart_chunk(units, max_length=1000):
    """
    Split only large units.
    Preserve structured units.
    """

    final_chunks = []

    for unit in units:
        if len(unit) <= max_length:
            final_chunks.append(unit)
        else:
            split_units = splitter.split_text(unit)
            final_chunks.extend(split_units)

    return final_chunks