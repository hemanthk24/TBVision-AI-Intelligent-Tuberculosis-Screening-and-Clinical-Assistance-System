from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.config.setting import PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION, DATA_DIR
from src.rag.data_loader import data_loader_and_splitter
from src.rag.embedding_model import load_embedding_model
from src.utils.logger import logger


# data ingestion
def data_ingestion(indexname,chunks,embedding_model):
    pc = Pinecone()
    if not pc.has_index(indexname):
        pc.create_index(
            name=indexname,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    docsearch = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=indexname,
        embedding=embedding_model,
    )
    
    print("Data Succesfully uploded to pinecone")
    

# data ingestion to the pinecone
if __name__ == "__main__":
    embedding = load_embedding_model()
    chunks = data_loader_and_splitter(DATA_DIR)
    print("------- Created chunks of data--------")
    data_ingestion(PINECONE_INDEX_NAME, chunks=chunks, embedding_model=embedding)
    print("-------Completed Data Ingestion-------")
    