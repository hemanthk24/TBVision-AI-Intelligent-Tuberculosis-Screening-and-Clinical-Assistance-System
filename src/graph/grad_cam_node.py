from src.grad_cam.make_grad_cam_tool import gradcam_tool
from src.cv_model.resnet_model import cv_model
from src.graph.state import DiagnosticState


# 2. ------- Grad-CAM tool ---------
def grad_cam_node(state: DiagnosticState)->dict:
    image_path = state['image_path']
    model = cv_model()
    grad_cam_output = gradcam_tool(image_path=image_path,model=model)
    return {
        'grad_cam_output':grad_cam_output['highlighted_region']
    }
    
