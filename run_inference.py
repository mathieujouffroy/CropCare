import math
import wandb
import os
import json
import logging
import tensorflow as tf
from datasets import load_from_disk
from transformers import DefaultDataCollator
from train_framework.utils import set_seed, set_logging, parse_args
from train_framework.prep_data_train import load_split_hdf5
from train_framework.preprocess_tensor import prep_ds_input
from train_framework.metrics import compute_training_metrics, f1_m
from train_framework.models import LayerScale

logger = logging.getLogger(__name__)

dependencies = {
    'f1_m': f1_m,
}

def evaluate_models(args, model_dict, test_dataset):
    """
    Evaluate the models on the test dataset.
    """

    for name, model in model_dict.items():
        args.model_dir = os.path.join(args.output_dir, f"{name}")
        print(args.model_dir)
        print(name)
        if args.wandb:
            dir_name = args.output_dir.split('/')[-1]
            project_name = f"cropdis-{dir_name}"
            MODEL_NAME = name
            cfg = {
                "dataset": args.dataset,
                "nbr_classes": args.n_classes,
                "class_names": args.class_names,
                "class_type": args.class_type,
                "test_set_len": args.len_test,
                "batch_size": args.batch_size,
                "nbr_test_batches": args.nbr_test_batch,
            }
            print(MODEL_NAME)
            run = wandb.init(project=project_name,
                         job_type="infer", name=MODEL_NAME, config=cfg, reinit=True)
            #wandb.init() returns a run object, and you can also access the run object via wandb.run:
            assert run is wandb.run

        logger.info("\n")
        logger.info(f"  ***** Evaluating {name} Validation set *****")
        compute_training_metrics(args, model, name, test_dataset, m_type='eval_test')


        if args.wandb:
            wandb.run.finish()
            print("\n\n--- FINISH WANDB RUN ---\n")

def main():
    args = parse_args()
    # SET LOGGING
    set_logging(args, 'infer')
    # SET SEED
    set_seed(args)

    if args.class_type == 'healthy':
        args.n_classes = 2
        args.class_names = ['healthy', 'not_healthy']
    else:
        if args.transformer:
            args.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            args.loss = tf.keras.losses.CategoricalCrossentropy()

        if args.class_type == 'disease':
            args.n_classes = 38
            args.label_map_path = 'resources/label_maps/diseases_label_map.json'
        elif args.class_type == 'plants':
            args.n_classes = 14
            args.label_map_path = 'resources/label_maps/plants_label_map.json'
        else:
            args.n_classes = 14
            args.label_map_path = 'resources/label_maps/general_diseases_label_map.json'
        with open(args.label_map_path) as f:
            id2label = json.load(f)
        args.class_names = [str(v) for k,v in id2label.items()]
        print(f"  Class names = {args.class_names}")

    # Load the dataset
    X_test, y_test = load_split_hdf5(args.dataset, 'test')
    # Set parameters
    args.len_test = len(X_test)
    args.nbr_test_batch = int(math.ceil(args.len_test / args.batch_size))


    ## Create Dataset
    if args.transformer:
        img_size = (224, 224)
        test_set = load_from_disk(f'{args.fe_dataset}/test')
        data_collator = DefaultDataCollator(return_tensors="tf")
        print(test_set.features["labels"].names)
        test_set = test_set.to_tf_dataset(
                    columns=['pixel_values'],
                    label_cols=["labels"],
                    shuffle=True,
                    batch_size=32,
                    collate_fn=data_collator)
    else:
        if args.xp_dir == 'resources/best_models/transformers':
            img_size = (224, 224)
        else:
            img_size = (128, 128)
        test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    test_set = prep_ds_input(args, test_set, args.len_test, img_size)
    for elem, label in test_set.take(1):
        print(elem)
        img = elem[0].numpy()
        #print(f"element shape is {elem.shape}, type is {elem.dtype}")
        print(f"image shape is {img.shape}, type: {img.dtype}")
        print(f"label shape is {label.shape} type: {label.dtype}")

    model_dict = dict()
    for model_dir in os.listdir(args.xp_dir):
        # Load the trained model saved to disk
        print(f"Model is : {model_dir}")

        if os.path.isdir(f'{args.xp_dir}/{model_dir}'):
            print(f'{args.xp_dir}/{model_dir}')
            if model_dir == 'ConvNext':
                model = tf.keras.models.load_model(
                        f"{args.xp_dir}/{model_dir}/model-best.h5",custom_objects={'f1_m': f1_m, 'LayerScale':LayerScale})
            else:
                model = tf.keras.models.load_model(
                        f"{args.xp_dir}/{model_dir}/model-best.h5",custom_objects=dependencies)
            print(model.summary())
            model_dict[model_dir] = model

    evaluate_models(args, model_dict, test_set)

if __name__ == '__main__':
    main()