from flask import Flask, request, jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import LlamaPostgresVectorStore
from langchain.llms import LlamaLLM
from langchain.memory import ConversationBufferMemory
import time
import psycopg2
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Set up model and vector store using LangChain components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# Connect to Postgres database
conn = psycopg2.connect(
    dbname="vector_db",
    user="jerry",
    password="password",
    host="localhost",
    port="5432"
)
conn.autocommit = True

# Initialize vector store with LangChain wrapper
vector_store = LlamaPostgresVectorStore.from_params(
    database="vector_db",
    host="localhost",
    password="password",
    port="5432",
    user="jerry",
    table_name="llama2_paper",
    embed_dim=384,
)

# Define LangChain components
prompt_template = PromptTemplate(template="Given the following query: {query}, provide a detailed response.")
memory = ConversationBufferMemory()

llm_chain = LLMChain(
    llm=LlamaLLM(llm),
    prompt=prompt_template,
    memory=memory
)

# Define routes
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_str = data.get('query')
    
    # Start latency timer
    start_time = time.time()
    
    # Create vector store query
    query_embedding = embed_model.get_query_embedding(query_str)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=2,
        mode="default"
    )
    
    # Query the vector store and retrieve relevant nodes
    query_result = vector_store.query(vector_store_query)
    nodes_content = "\n".join([node.get_content() for node in query_result.nodes])
    
    # Generate a response using the LLM chain
    response = llm_chain.run({"query": nodes_content})
    
    # Measure latency
    latency = time.time() - start_time
    
    # Prepare the result
    result = {
        "query": query_str,
        "response": response,
        "latency": latency,
        "source_content": nodes_content
    }
    
    # Store result in the database
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO responses (query, response, latency) VALUES (%s, %s, %s)",
            (query_str, response, latency)
        )
    
    return jsonify(result)

@app.route('/load_data', methods=['POST'])
def load_data():
    # Example endpoint to load and process data
    loader = PyMuPDFReader()
    documents = loader.load(file_path=Path("data/llama2.pdf"))
    
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