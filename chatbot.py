from flask import Flask, request, jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore 
from llama_index.readers.file import PyMuPDFReader

import time
import psycopg2
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Set up model and vector store
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

llm = LlamaCPP(
    #model_url=model_url,
    model_path= 'models\llama-2-7b-chat.Q2_K.gguf',
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# Connect to Postgres database
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="geniusraver27",
    host="localhost",
    port="5432"
)
conn.autocommit = True

# Initialize vector store
vector_store = PGVectorStore.from_params(
    database="postgres",
    host="localhost",
    password="geniusraver27",
    port="5432",
    user="postgres",
    table_name="llama2_paper",
    embed_dim=384,  # OpenAI embedding dimension
)

# Define retriever class
class VectorDBRetriever(BaseRetriever):
    def __init__(self, vector_store, embed_model, query_mode="default", similarity_top_k=2):
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)
        nodes_with_scores = [
            NodeWithScore(node=node, score=(query_result.similarities[index] if query_result.similarities else None))
            for index, node in enumerate(query_result.nodes)
        ]
        return nodes_with_scores

retriever = VectorDBRetriever(vector_store, embed_model, query_mode="default", similarity_top_k=2)
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

# Define routes
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_str = data.get('query')
    start_time = time.time()
    response = query_engine.query(query_str)
    latency = time.time() - start_time
    result = {
        "query": query_str,
        "response": str(response),
        "latency": latency,
        "source_content": response.source_nodes[0].get_content() if response.source_nodes else None
    }
    # Store result in the database
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO responses (query, response, latency) VALUES (%s, %s, %s)",
            (query_str, result["response"], latency)
        )
    return jsonify(result)

@app.route('/load_data', methods=['POST'])
def load_data():
    # Example endpoint to load and process data
    loader = PyMuPDFReader()
    file_path = Path("data/llama2.pdf")
    documents = loader.load(file_path=file_path)
    
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
    return jsonify({"status": "Data loaded and processed successfully."})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)