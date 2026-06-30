from src.cv_model.resnet_model import cv_model
from src.cv_model.prediction import predict_xray
from src.graph.state import DiagnosticState
from src.utils.logger import logger

## 1. --------- Prediction node ----------
def prediction_node(state: DiagnosticState)->dict:
    image_path = state['image_path']
    model = cv_model()
    predictions = predict_xray(model=model,image_path=image_path)
    return {
        'prediction':predictions['prediction'],
        'confidence':predictions['confidence']
    }
    