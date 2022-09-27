import gradio as gr
import cv2
import torch
import numpy as np
import tensorflow as tf
import requests
from metrics import f1_m
import json 
from tensorflow import keras 
#from tensorflow.keras.applications.convnext import LayerScale

### GRADIO INTERFACE WITH PREDICTION CAM / GRADCAM / SALIENCY MAPS
dependencies = {
    'f1_m': f1_m,
}

class LayerScale(tf.keras.layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

with open("../resources/label_maps/diseases_label_map.json") as f:
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
        clf =  tf.keras.models.load_model("Best_models/pret_EfficientNetV2B3_dropout/model-best.h5",custom_objects=dependencies)
        input = resize_img(input, (128, 128))
    elif model == 'ConvNext':
        clf =  tf.keras.models.load_model("Best_models/ConvNext/model-best.h5",custom_objects={'f1_m': f1_m, 'LayerScale':LayerScale})
        input = resize_img(input, (224, 224))
    elif model == 'DenseNet201':
        #input = tf.keras.applications.mobilenet_v2.preprocess_input(input)
        clf = tf.keras.models.load_model("Best_models/pret_DenseNet201/model-best.h5", custom_objects=dependencies)
        input = resize_img(input, (128, 128))
    elif model == 'ResNet50V2':
        clf = tf.keras.models.load_model("Best_models/pret_ResNet50V2/model-best.h5",custom_objects=dependencies)
        input = resize_img(input, (128, 128))

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
			  ["Img_test_app/Apple_Black_rot/image (14).JPG", 'Apple Black Rot'],
			  ["Img_test_app/Apple_Black_rot/image (29).JPG", 'Apple Black Rot'],
			  ["Img_test_app/Blueberry_healthy/image (23).JPG", 'Blueberry healthy'],
			  ["Img_test_app/Blueberry_healthy/image (34).JPG", 'Blueberry healthy'],

			  ["Img_test_app/Cherry_Powdery_Mildew/image (30).JPG", 'Cherry Powdery Mildew'],
			  ["Img_test_app/Cherry_Powdery_Mildew/image (77).JPG", 'Cherry Powdery Mildew'],
			  ["Img_test_app/Corn_Northern_Leaf_Blight/image (17).JPG", 'Corn Northern Leaf Blight'],
			  ["Img_test_app/Corn_Northern_Leaf_Blight/image (37).JPG", 'Corn Northern Leaf Blight'],
			  ["Img_test_app/Grape_Black_rot/image (28).JPG", 'Grape Black Rot'],
              ],
            #interpretation="shap",
            #num_shap=3,
            title='Crop Disease Detection'
        )

if __name__ == "__main__":
    demo.launch()



