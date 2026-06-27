from src.cv_model.resnet_model import build_model
from src.utils.logger import logger
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from pathlib import Path
from src.config.setting import IMG_SIZE, MODEL_INPUT_SHAPE, TB_MODEL_WEIGHTS_PATH
import numpy as np

# loading Model
model = build_model()
model.load_weights(TB_MODEL_WEIGHTS_PATH)

def crop_center_region_np(image):
    h, w = image.shape[:2]

    top_crop = int(0.05 * h)
    bottom_crop = int(0.05 * h)
    left_crop = int(0.08 * w)
    right_crop = int(0.08 * w)

    cropped = image[
        top_crop:h - bottom_crop,
        left_crop:w - right_crop,
        :
    ]

    cropped = cv2.resize(cropped, IMG_SIZE)
    return cropped

def apply_clahe_np(image):
    image = np.uint8(np.clip(image, 0, 255))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_img.astype(np.float32)

def preprocess_xray(image_path):
    try:
        image_path = str(image_path)
        # read image
        image = cv2.imread(image_path)

        # BGR > RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize
        image = cv2.resize(image, IMG_SIZE)
        # crop
        image = crop_center_region_np(image)
        # CLAHE
        image = apply_clahe_np(image)
        # ResNet preprocessing
        image = preprocess_input(image)
        # add batch dimension
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        print("Exception Occured: ",e)
    return image


def predict_xray(model,image_path):
    class_names = ['Normal', 'Tuberculosis']
    x = preprocess_xray(image_path=image_path)
    prob = model.predict(x)[0][0]
    
    if prob >= 0.35:
        pred_class = 1
        pred_label = class_names[1]
    else:
        pred_class = 0
        pred_label = class_names[0]
    
    confidence = prob if pred_class == 1 else 1 - prob
        
    return {
        "probability_tuberculosis": float(prob),
        "prediction" : pred_label,
        "confidence": float(confidence)
    }
    
    
if __name__ == "__main__":
    image_path = Path("test_images\\img_normal1.jpg")
    predicted = predict_xray(model=model,image_path=image_path)
    print("Prediction of the image :",predicted)

