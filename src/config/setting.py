from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone
PINECONE_INDEX_NAME = "tb-chunks"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Model
TB_MODEL_PATH = (
    "src/CV_model/tb_weights_fixed.weights.h5"
)

# Retrieval
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 5

# Thresholds
TB_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.80

# Outputs
GRADCAM_OUTPUT_DIR = (
    "outputs/grad_cam_outputs"
)

# LLM
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0