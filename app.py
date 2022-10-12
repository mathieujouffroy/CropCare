import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from train_framework.metrics import f1_m
from train_framework.models import LayerScale
import json 
from tensorflow import keras 

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


def classify_image(input, label, model):
    print(f"model: {model}")
    print(label)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = tf.convert_to_tensor(input)
    input = tf.expand_dims(input, 0)
    input = tf.cast(input, tf.float32)

    if model == 'EfficientNetV2B3':
        input = resize_img(input, (128, 128))
        clf =  tf.keras.models.load_model("resources/best_models/cnn/pret_EfficientNetV2B3_dropout/model-best.h5",custom_objects=dependencies)
    elif model == 'ConvNext':
        input = resize_img(input, (224, 224))
        clf =  tf.keras.models.load_model("resources/best_models/transformers/ConvNext/model-best.h5",custom_objects={'f1_m': f1_m, 'LayerScale':LayerScale})
        
    #elif model == 'DenseNet201':
    #    clf = tf.keras.models.load_model("resources/best_models/pret_DenseNet201/model-best.h5", custom_objects=dependencies)
    #    input = resize_img(input, (128, 128))
    #elif model == 'ResNet50V2':
    #    clf = tf.keras.models.load_model("resources/best_models/pret_ResNet50V2/model-best.h5",custom_objects=dependencies)
    #    input = resize_img(input, (128, 128))

    print(input.shape)
    y_probs = clf.predict(input)
    y_pred = np.argmax(y_probs, axis=-1)
    pred_label_names = [CLASS_INDEX[str(y)] for y in y_pred]
    pred = pred_label_names[0]
    return pred


demo = gr.Interface(
            fn=classify_image, 
            inputs=[
              gr.Image(shape=(256, 256)),
              gr.Textbox(value='Apple Healthy', label='label'),
              gr.Dropdown(choices=['EfficientNetV2B3', 'ConvNext', 'ResNet50V2', 'DenseNet201'], type="value", label='model'),
              #gr.Checkbox(label="Remove Background"),
              ],
            outputs=gr.Label(num_top_classes=3),
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
            title='Crop Disease Detection'
        )

if __name__ == "__main__":
    demo.launch()



