
# CropCare
Crop Disease Detection

## CLI
Command Line Interface used to generate the dataset and plot class distribution.
<br>
<br>
You have the possibility to choose between 4 types of labels:<br>
- plants
- diseases
- general_diseases
- healthy (binary)
<br>
You can also remove the background in the images with different techniques.<br>
<br>

```bash
python cli/cli.py path_images_folder
```

## Train_framework
Script to train different Computer Vision models on a given classification task (multiclass or binary).<br>
The training configuration parameters are in the file configs/train_config.yml.<br>
The evaluation configuration parameters are in the file configs/infer_config.yml.<br>
The models you can train are in the file resources/models_to_eval.json
<br>
<br>
Training script:
<br>
```bash
python run_training.py --config configs/train_config.yml
```
<br>
Inference script:

```bash
python run_infererence.py --configs configs/infer_config.yml
```
<br>
Gradio App (Demo):

```bash
python app.py
```

Report on:<br>
 <a href="https://wandb.ai/mjouffro/cropdis-models-comp/reports/Plant-Disease-Classification--VmlldzoyMjc1OTQ5">
    <img src="https://camo.githubusercontent.com/5c70f08219d50671f896067e1024b0db9dfca119304d0d977cbf273565be32fc/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d576569676874732b2532362b42696173657326636f6c6f723d323232323232266c6f676f3d576569676874732b2532362b426961736573266c6f676f436f6c6f723d464642453030266c6162656c3d" alt="Weight&Biases Badge"/>

  </a>

<!--
-->
