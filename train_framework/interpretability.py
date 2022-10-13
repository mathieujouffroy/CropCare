import wandb
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from train_framework.preprocess_tensor import preprocess_image


def get_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def make_gradcam_heatmap(args, model, m_name, img_array, pred_index=None):
    """
    GRADient-weighted Class Activation Mapping (Grad-CAM)

    We let the gradients of any target concept score (logits for any class of interest) flow
    into the final convolutional layer. We can then compute an importance score based on
    the gradients and produce a coarse localization map highlighting the important regions
    in the image for predicting that concept.
    """

    img_array = img_array[np.newaxis, :]
    model.layers[-1].activation = None
    last_4d = get_target_layer(model)
    if "Functional" == last_4d.__class__.__name__:
        last_conv = get_target_layer(last_4d)
        print(f"last conv: {last_conv.name}")
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_4d.inbound_nodes[0].output_tensors, model.output]
        )
    else:
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.input], [model.get_layer(last_conv).output, model.output]
        )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        img_array = tf.cast(img_array, tf.float32)
        tape.watch(img_array)
        last_conv_layer_output, preds = grad_model(img_array)
        # watch the conv_output_values
        tape.watch(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(args, model, m_name, x_test, n_img, model_metrics_dir, alpha=0.4):
    print(f"X test shape is : {x_test.shape}")
    print(f"Displaying Grad-CAM for {m_name} on {n_img} images")

    img_ids = random.sample(range(x_test.shape[0]), n_img)
    print(model.summary())

    if args.wandb:
        TABLE_NAME = "gradcam_visualization"
        columns = ["image", "heat_map", "img + gradcam"]
        grad_cam_table = wandb.Table(columns=columns)

    model.layers[-1].activation = None
    for id in img_ids:
        #img = tf.cast(img, tf.float32)
        img = x_test[id]
        #img = preprocess_image(img, args.mean_arr, args.std_arr, mode)
        heatmap = make_gradcam_heatmap(args, model, m_name, img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Display Grad CAM
        plt.imshow(superimposed_img)
        plt.show()

        # Save the superimposed image
        superimposed_img.save(f"{model_metrics_dir}/img_{id}.jpg")

        if args.wandb:
            row = [wandb.Image(img), #,caption=np.argmax(predictions[i])),
                    wandb.Image(heatmap),
                    wandb.Image(superimposed_img)]
            grad_cam_table.add_data(*row)

    if args.wandb:
        wandb.run.log({TABLE_NAME : grad_cam_table})