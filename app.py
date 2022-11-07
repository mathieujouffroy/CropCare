import json
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from train_framework.metrics import f1_m
from train_framework.models import LayerScale
from train_framework.interpretability import display_gradcam

### GRADIO INTERFACE WITH PREDICTION CAM / GRADCAM / SALIENCY MAPS
dependencies = {
    'f1_m': f1_m,
}

with open("resources/label_maps/diseases_label_map.json") as f:
    CLASS_INDEX = json.load(f)

@tf.function
def resize_img(img, shape):
    img = tf.image.resize(img, shape)
    return img


def classify_image(image, label, model, remove_bg=False):
    print(f"model: {model}")
    print(label)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    if model == 'EfficientNetV2B3':
        image = resize_img(image, (128, 128))
        clf =  tf.keras.models.load_model("resources/best_models/cnn/pret_EfficientNetV2B3_dropout/model-best.h5",custom_objects=dependencies)
    elif model == 'ConvNext':
        image = resize_img(image, (224, 224))
        clf =  tf.keras.models.load_model("resources/best_models/transformers/ConvNext/model-best.h5",custom_objects={'f1_m': f1_m, 'LayerScale':LayerScale})

    elif model == 'DenseNet201':
        image = resize_img(image, (128, 128))
        print(image.shape)
        clf = tf.keras.models.load_model("resources/best_models/cnn/pret_DenseNet201/model-best.h5", custom_objects=dependencies)

    #elif model == 'ResNet50V2':
    #    clf = tf.keras.models.load_model("resources/best_models/pret_ResNet50V2/model-best.h5",custom_objects=dependencies)
    #    input = resize_img(input, (128, 128))
    input = tf.expand_dims(image, 0)
    y_probs = clf.predict(input).flatten()
    y_pred = np.argmax(y_probs, axis=-1)
    confidences = {CLASS_INDEX[str(i)]: float(y_probs[i]) for i in range(38)}
    superimposed_img, heatmap = display_gradcam(clf, image, alpha=0.6)
    return confidences, superimposed_img


demo = gr.Interface(
            fn=classify_image,
            inputs=[
              gr.Image(shape=(256, 256)),
              gr.Textbox(label='Target Label'),
              gr.Dropdown(choices=['EfficientNetV2B3', 'ConvNexT', 'ResNet50V2', 'DenseNet201'], type="value", label='Model'),
              gr.Checkbox(label="Remove Background"),
              ],
            outputs=[
                gr.Label(num_top_classes=3),
                gr.outputs.Image(type="pil", label="Grad CAM"),
            ],
            examples=[
              ["resources/test_imgs/Apple___Black_rot/image (14).JPG", 'Apple Black Rot'],
              ["resources/test_imgs/Apple___Black_rot/image (29).JPG", 'Apple Black Rot'],
              ["resources/test_imgs/Blueberry___healthy/image (23).JPG", 'Blueberry healthy'],
              ["resources/test_imgs/Blueberry___healthy/image (34).JPG", 'Blueberry healthy'],

              ["resources/test_imgs/Cherry___Powdery_Mildew/image (30).JPG", 'Cherry Powdery Mildew'],
              ["resources/test_imgs/Cherry___Powdery_Mildew/image (77).JPG", 'Cherry Powdery Mildew'],
              ["resources/test_imgs/Corn___Northern_Leaf_Blight/image (17).JPG", 'Corn Northern Leaf Blight'],
              ["resources/test_imgs/Corn___Northern_Leaf_Blight/image (37).JPG", 'Corn Northern Leaf Blight'],
              ["resources/test_imgs/Grape___Black_rot/image (28).JPG", 'Grape Black Rot'],
              ],
            #interpretation="shap",
            #num_shap=3,
            title='Crop Disease Detection',
            description="Crop Disease Prediction with Grad CAM",
        )

if __name__ == "__main__":
    demo.launch()
