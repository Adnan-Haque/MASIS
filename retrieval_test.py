from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

COLLECTION_NAME = "masis_documents"
WORKSPACE_ID = "test2"

client = QdrantClient(host="localhost", port=6333)

# =====================================================
# 1️⃣ Count Total Chunks
# =====================================================

count_result = client.count(
    collection_name=COLLECTION_NAME,
    count_filter=Filter(
        must=[
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=WORKSPACE_ID)
            )
        ]
    )
)

total_chunks = count_result.count
print("\n===== WORKSPACE SUMMARY =====")
print("Workspace ID:", WORKSPACE_ID)
print("Total Chunks:", total_chunks)

# =====================================================
# 2️⃣ Scroll All Points
# =====================================================

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=WORKSPACE_ID)
            )
        ]
    ),
    limit=10000,  # increase if needed
    with_payload=True,
    with_vectors=False
)

# =====================================================
# 3️⃣ Analyze Per Document
# =====================================================

doc_chunk_count = defaultdict(int)
file_names = set()

for point in points:
    payload = point.payload
    file_name = payload.get("file_name", "UNKNOWN")

    doc_chunk_count[file_name] += 1
    file_names.add(file_name)

print("\nTotal Documents:", len(file_names))
print("\nChunks Per Document:")
print("--------------------------------")

for file_name, chunk_count in doc_chunk_count.items():
    print(f"{file_name} → {chunk_count} chunks")

# =====================================================
# 4️⃣ Sample Chunk Preview
# =====================================================

print("\nSample Chunk Preview:")
print("--------------------------------")

for i, point in enumerate(points[:3]):
    payload = point.payload
    text_preview = payload.get("text", "")[:200]
    print(f"\nDocument: {payload.get('file_name')}")
    print("Chunk Index:", payload.get("chunk_index"))
    print("Text Preview:", text_preview)

print("\n===== END SUMMARY =====")