from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Gemini(model="models/gemini-3.6-flash")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("legal_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    system_prompt=(
        "You are a legal assistant. Answer ONLY using the provided context "
        "from the uploaded legal documents. If the answer isn't in the "
        "documents, say so clearly instead of guessing. Cite the relevant "
        "clause or section when possible."
    ),
)

while True:
    question = input("\nAsk a question about your documents (or 'quit'): ")
    if question.lower() == "quit":
        break
    response = query_engine.query(question)
    print("\nAnswer:", response)