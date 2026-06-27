from src.config.setting import SEARCH_KWARGS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.rag.vector_store import get_vector_db
from src.rag.data_loader import get_chunks

_retriever = None

# create retriver
def create_retriever(search_kwargs,vectordb,chunks):
    
    pinecone_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    bm25_retriever.k = 3
    
    
    # hybrid retiever
    hybrid_retiever = EnsembleRetriever(
        retrievers=[pinecone_retriever,bm25_retriever],
        weights=[0.8,0.2]
    )
    
    return hybrid_retiever

# get retriever
def get_retriever():
    global _retriever
    
    if _retriever is None:
        _retriever = create_retriever(SEARCH_KWARGS, vectordb=get_vector_db(), chunks=get_chunks())
    return _retriever