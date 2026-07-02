from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.config.setting import PINECONE_INDEX_NAME
from src.rag.embedding_model import get_embedding_model

_vector_db = None

# creating vectorstore 
def create_vectorstore(indexname,embedding_model):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=indexname,
        embedding=embedding_model
    )
    return vectorstore

# get vector_db
def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = create_vectorstore(PINECONE_INDEX_NAME, embedding_model=get_embedding_model())
    return _vector_db