from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Base dir
BASE_DIR = Path(__file__).resolve().parents[2]

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone
PINECONE_INDEX_NAME = "tb-chunks"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Models
TB_MODEL_WEIGHTS_PATH = BASE_DIR / "src" / "cv_model" / "tb_weights_fixed.weights.h5"
LLM_MODEL = "gpt-4o-mini"

# Image
IMG_SIZE = (224, 224)
MODEL_INPUT_SHAPE = (224, 224, 3)
# Retrieval
TOP_K_RETRIEVAL = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Thresholds
TB_THRESHOLD = 0.35
LOW_CONFIDENCE_THRESHOLD = 0.45

# Outputs
GRADCAM_OUTPUT_DIR = BASE_DIR / "outputs" / "grad_cam_outputs"

# LLM
TEMPERATURE = 0