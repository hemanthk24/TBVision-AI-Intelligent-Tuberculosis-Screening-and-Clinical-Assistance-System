import numpy as np
from src.cv_model.resnet_model import build_model
from src.cv_model.prediction import crop_center_region_np, apply_clahe_np
from src.config.setting import IMG_SIZE
from src.cv_model.resnet_model import cv_model
import cv2
import os
from pathlib import Path
from src.utils.logger import logger
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def make_gradcam_heatmap(img_array, model, pred_index=None):

    # correct layers
    base_model = model.layers[1]

    gap = model.layers[2]
    dense1 = model.layers[3]
    dropout1 = model.layers[4]
    dense2 = model.layers[5]
    dropout2 = model.layers[6]
    out_layer = model.layers[7]

    with tf.GradientTape() as tape:

        # feature maps from ResNet50
        conv_outputs = base_model(img_array)

        tape.watch(conv_outputs)

        # classifier head
        x = gap(conv_outputs)

        x = dense1(x)

        x = dropout1(
            x,
            training=False
        )

        x = dense2(x)

        x = dropout2(
            x,
            training=False
        )

        preds = out_layer(x)

        # binary classifier
        class_channel = preds[:, 0]

    # gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # average gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # weighted activation maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # normalize
    heatmap /= (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def detect_highlighted_region(heatmap):
    h, w = heatmap.shape

    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    if y < h / 3:
        vertical = "upper"
    elif y < 2 * h / 3:
        vertical = "middle"
    else:
        vertical = "lower"

    if x < w / 3:
        horizontal = "left"
    elif x > 2 * w / 3:
        horizontal = "right"
    else:
        horizontal = "central"

    return f"{vertical} {horizontal} lung region"

def preprocess_image(image_path):
    try:
        image_path = str(image_path)

        # read image
        image = cv2.imread(image_path)

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize
        image = cv2.resize(image, IMG_SIZE)

        # save original for Grad-CAM overlay
        original_image = image.copy()

        # crop
        image = crop_center_region_np(image)

        # CLAHE
        image = apply_clahe_np(image)

        # ResNet preprocessing
        image = preprocess_input(image)

        # add batch dimension
        image = np.expand_dims(image, axis=0)

        return original_image, image

    except Exception as e:
        print("Exception Occured:", e)
        return None, None
    
def gradcam_tool(image_path,model):
    # create output directory
    os.makedirs("grad_cam_outputs", exist_ok=True)

        # preprocess image
    original_img, img_array = preprocess_image(image_path)

        # generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

        # resize heatmap
    heatmap_resized = cv2.resize(
            heatmap,
            (original_img.shape[1], original_img.shape[0])
        )

        # convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # overlay
    overlay = cv2.addWeighted(
            original_img.astype(np.uint8),
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # save image
    output_path = (
            f"grad_cam_outputs/"
            f"{Path(image_path).stem}_gradcam.png"
        )

    cv2.imwrite(output_path, overlay)

        # highlighted region
    highlighted_region = detect_highlighted_region(heatmap)

    return {
            "grad_cam": output_path,
            "highlighted_region": highlighted_region
        }
    
    
    
if __name__ == "__main__":
    image_path = Path("test_images\\img_normal1.jpg")
    model = cv_model()
    output = gradcam_tool(image_path=image_path,model=model)
    print("Grad_cam_output :",output)