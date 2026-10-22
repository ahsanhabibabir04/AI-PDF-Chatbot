
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_search_index(chunks):
    """Create embeddings and a FAISS index for text chunks."""
    if not chunks:
        return None, None, None

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks


def search_chunks(question, model, index, chunks, top_k=3):
    """Return the most relevant chunks for a question."""
    if model is None or index is None or not chunks:
        return []

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": chunks[int(idx)],
                "score": float(score),
            })

    return results
