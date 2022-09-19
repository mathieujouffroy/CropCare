import gradio as gr
import cv2
import torch
import numpy as np
import tensorflow as tf
import requests
from metrics import f1_m
import json 
from tensorflow import keras 

### GRADIO INTERFACE WITH PREDICTION CAM / GRADCAM / SALIENCY MAPS
dependencies = {
    'f1_m': f1_m,
}

with open("../resources/label_maps/diseases_label_map.json") as f:
    CLASS_INDEX = json.load(f)

@tf.function
def resize_img(img):
    img = tf.image.resize(img, (128, 128))
    return img

#@tf.function
#def encode_categorical(label, n_classes):
#    label = tf.one_hot(label, n_classes,  dtype='uint8')
#    return label


def classify_image(input, label, model):
    print(f"model: {model}")
    print(label)
    #image = cv2.imread("../Img_test_app/Bluenerry_healthy/image (23).JPG")
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = tf.convert_to_tensor(input)
    input = tf.expand_dims(input, 0)  # Create batch axis
    input = tf.cast(input, tf.float32)
    input = resize_img(input)
    #label = encode_categorical(label, 38)
    print(input.shape)
    
    #if model == 'CVT':
    #    #input = tf.keras.applications.mobilenet_v2.preprocess_input(input)
    #    clf = tf.keras.applications.MobileNetV2()
    #elif model == 'VIT':
    #    #input = tf.keras.applications.mobilenet_v2.preprocess_input(input)
    #    clf = tf.keras.applications.MobileNetV2()
    if model == 'EfficientNetV2B3':
        clf =  tf.keras.models.load_model("wandb/run-20220915_175411-1tbtjv3t/files/model-best.h5",custom_objects=dependencies)
    elif model == 'InceptionResnetV2':
        clf =  tf.keras.models.load_model("Best_models/InceptionResNetV2_dropout_/model-best.h5",custom_objects=dependencies)
    elif model == 'InceptionV3':
        clf =  tf.keras.models.load_model("../resources/experiments/comp-top-k/InceptionV3_dropout_16:09:2022_07:31:58",custom_objects=dependencies)
    elif model == 'DenseNet201':
        #input = tf.keras.applications.mobilenet_v2.preprocess_input(input)
        clf = tf.keras.models.load_model("wandb/run-20220916_150125-2azy63ub/files/model-best.h5", custom_objects=dependencies)
    elif model == 'ResNet50V2':
        clf = tf.keras.models.load_model("wandb/run-20220916_133658-1w7qi5vq/files/model.h5",custom_objects=dependencies)


    #y_probs = clf(input)
    clf = clf(input, training=False)
    print(clf)
    y_probs = clf(input)
    #y_pred = np.argmax(y_probs, axis=-1)
    #pred_label_names = [CLASS_INDEX[str(y)] for y in y_pred]
    #pred = pred_label_names[0]
    #return pred
    print(y_probs)
    prediction = y_probs.flatten()
    confidences = {CLASS_INDEX[i]: float(prediction[i]) for i in range(38)}
    return confidences

demo = gr.Interface(
            fn=classify_image, 
            inputs=[
              gr.Image(shape=(128, 128)),
			  gr.Textbox(value='Apple Healthy', label='label'),
              gr.Dropdown(choices=['EfficientNetV2B3','InceptionV3','InceptionResnetV2', 'ResNet50V2'], type="value", label='model'),
              #gr.Checkbox(label="Remove Background"),
              ],
            outputs=gr.Label(num_top_classes=3),
            examples=[
			  ["../Img_test_app/Apple_Black_rot/image (14).JPG", 'Apple Black Rot'],
			  ["../Img_test_app/Apple_Black_rot/image (29).JPG", 'Apple Black Rot'],
			  ["../Img_test_app/Blueberry_healthy/image (23).JPG", 'Blueberry healthy'],
			  ["../Img_test_app/Blueberry_healthy/image (34).JPG", 'Blueberry healthy'],

			  ["../Img_test_app/Cherry_Powdery_Mildew/image (30).JPG", 'Cherry Powdery Mildew'],
			  ["../Img_test_app/Cherry_Powdery_Mildew/image (77).JPG", 'Cherry Powdery Mildew'],
			  ["../Img_test_app/Corn_Northern_Leaf_Blight/image (17).JPG", 'Corn Northern Leaf Blight'],
			  ["../Img_test_app/Corn_Northern_Leaf_Blight/image (37).JPG", 'Corn Northern Leaf Blight'],
			  ["../Img_test_app/Grape_Black_rot/image (28).JPG", 'Grape Black Rot'],
			  ["../Img_test_app/Grape_Black_rot/image (128).JPG", 'Grape Black Rot'],
			  ["../Img_test_app/Grape_Black_rot/image (136).JPG", 'Grape Black Rot'],
              ],
            #interpretation="shap",
            #num_shap=3,
            title='Crop Disease Detection'
        )

if __name__ == "__main__":
    demo.launch()



