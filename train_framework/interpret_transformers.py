# https://github.com/sayakpaul/probing-vits
# https://github.com/jacobgil/vit-explain/blob/main/vit_explain.py
# https://jacobgil.github.io/deeplearning/vision-transformer-explainability
# https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf


#image = dataset["test"]["image"][0]
#feature_extractor = ViTFeatureExtractor.from_pretrained(
#    "google/vit-base-patch16-224")
#model = TFViTForImageClassification.from_pretrained(
#    "google/vit-base-patch16-224")
#
#inputs = feature_extractor(image, return_tensors="tf")
#logits = model(**inputs).logits
#
#
#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]
#
#feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
#model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#
#inputs = feature_extractor(image, return_tensors="tf")
#outputs = model(**inputs)
#
