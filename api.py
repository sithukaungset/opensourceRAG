from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import List
import time
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle

# Initialize FastAPI
app = FastAPI()

# Database and model configuration
db_name = "postgres"
host = "localhost"
password = "geniusraver27"
port = "5432"
user = "postgres"

# Initialize Postgres Connection for Vector Store
conn = psycopg2.connect(
    dbname=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

# Setup for Storing Responses
DATABASE_URL = f"postgresql://sithu:{password}@{host}:{port}/{db_name}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class ChatbotResponse(Base):
    __tablename__ = "chatbot_responses"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    response = Column(Text, nullable=False)
    latency = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Setup Vector Store and Embeddings
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Setup LLM Model
model_path = 'models/llama-2-7b-chat.Q2_K.gguf'
llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    document_url: str

# Ingest Document
@app.post("/ingest")
async def ingest_document(doc_request: DocumentRequest):
    doc_path = f"./data/{Path(doc_request.document_url).name}"

    loader = PyMuPDFReader()
    documents = loader.load(file_path=doc_path)

    text_parser = SentenceSplitter(chunk_size=1024)
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        node.embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        nodes.append(node)

    vector_store.add(nodes)

    return {"status": "success", "message": "Document ingested successfully."}

# Query Document and Measure Latency
@app.post("/query")
async def query_pipeline(query_request: QueryRequest):
    start_time = time.time()

    query_embedding = embed_model.get_query_embedding(query_request.query)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=2,
        mode="default",
    )
    query_result = vector_store.query(vector_store_query)

    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score = query_result.similarities[index] if query_result.similarities else None
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    retriever = VectorDBRetriever(vector_store, embed_model, query_mode="default", similarity_top_k=2)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    response = query_engine.query(query_request.query)

    end_time = time.time()
    latency = end_time - start_time

    # Save response and latency to the database
    db = SessionLocal()
    db_response = ChatbotResponse(
        query=query_request.query,
        response=str(response),
        latency=latency
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    db.close()

    return {
        "response": str(response),
        "source": response.source_nodes[0].get_content(),
        "latency": latency
    }

# VectorDBRetriever Class
class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: HuggingFaceEmbedding,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = query_result.similarities[index] if query_result.similarities else None
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
