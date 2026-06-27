from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src.config.setting import DATA_DIR


_chunks = None

# loading and splitting function
def data_loader_and_splitter(data_path):
    
    loader = PyPDFLoader(data_path)
    docs = loader.load()
    
    for doc in docs:
        doc.metadata = {
            'source':'Handbook of Tuberculosis',
            'Editors': 'Jacques H. Grosset and Richard E. Chaisson',
            'page': doc.metadata['page'],
            'total_pages': doc.metadata['total_pages']
            }
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
        )
    
    chunks = splitter.split_documents(docs)
    
    return chunks


# get chunks
def get_chunks():
    global _chunks
    
    if _chunks is None:
        _chunks = data_loader_and_splitter(DATA_DIR)
    return _chunks

    