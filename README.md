# Vi2PC
Plant Disease Classification

## CLI
Used to generate the dataset.
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
cd cli
python cli.py path_to_parent_folder_containing_images
```

## Train_framework
Script to train different CV models on given classification task (multiclass or binary).<>br
The configuration parameters are in the train_config.yml file.
<br>
<br>
```bash
cd train_framework
python train.py --config ../train_config.yml
```
