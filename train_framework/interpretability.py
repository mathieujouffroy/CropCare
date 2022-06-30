import wandb
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display


def make_gradcam_heatmap(model, img_array, pred_index=None):
    """
    GRADient-weighted Class Activation Mapping (Grad-CAM)

    We let the gradients of any target concept score (logits for any class of interest) flow
    into the final convolutional layer. We can then compute an importance score based on
    the gradients and produce a coarse localization map highlighting the important regions
    in the image for predicting that concept.
    """
    img_array = img_array[np.newaxis, :]

    last_conv_layer_name = "last_conv"
    model.layers[-1].activation = None
    print(model.get_layer(last_conv_layer_name).output)
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # watch the conv_output_values
        tape.watch(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print(f"Grads shape : {grads.shape}")

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print("pooled_grads shape : ", pooled_grads.shape)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(args, model, x_test, n_img, model_metrics_dir, alpha=0.4):
    print(f"Displaying Grad-CAM for {n_img} images")
    print(f"X test shape is : {x_test.shape}")

    img_ids = random.sample(range(x_test.shape[0]), n_img)

    if args.wandb:
        TABLE_NAME = "gradcam_visualization"
        columns = ["image", "heat_map", "img + gradcam"]
        grad_cam_table = wandb.Table(columns=columns)

    for id in img_ids:
        img = x_test[id]
        heatmap = make_gradcam_heatmap(model, img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        #superimposed_img.save(f"{model_metrics_dir}/img_{id}.jpg")

        # Display Grad CAM
        plt.imshow(superimposed_img)
        plt.show()

        if args.wandb:
            row = [wandb.Image(img), #,caption=np.argmax(predictions[i])),
                    wandb.Image(heatmap),
                    wandb.Image(superimposed_img)]
            grad_cam_table.add_data(*row)

    if args.wandb:
        wandb.run.log({TABLE_NAME : grad_cam_table})
