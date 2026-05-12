import chromadb
import uuid

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

chroma_client = chromadb.Client()

resume_collection = chroma_client.get_or_create_collection(
    name="resume_collection"
)

jd_collection = chroma_client.get_or_create_collection(
    name="jd_collection"
)


def store_resume(
    chunks,
    candidate_name
):

    existing = resume_collection.get(
        where={"candidate": candidate_name}
    )

    if existing["ids"]:
        return

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = []

    for idx in range(len(chunks)):

        metadatas.append({
            "candidate": candidate_name,
            "chunk": idx
        })

    resume_collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def store_jd(
    chunks,
    candidate_name
):

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = []

    for idx in range(len(chunks)):

        metadatas.append({
            "candidate": candidate_name,
            "chunk": idx
        })

    jd_collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def search_resumes(query):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = resume_collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    return results


def search_jds(query):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = jd_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results