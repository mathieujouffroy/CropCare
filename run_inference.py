import math
import wandb
import os
import gc
import json
import logging
import tensorflow as tf
from datasets import load_from_disk
from transformers import DefaultDataCollator, AdamWeightDecay
from train_framework.utils import set_seed, set_logging, parse_args
from train_framework.prep_data_train import load_split_hdf5
from train_framework.preprocess_tensor import prep_ds_input
from train_framework.metrics import compute_training_metrics, f1_m, matt_coeff, precision_m, recall_m
from train_framework.models import LayerScale
from train_framework.custom_inception_model import CopyChannels

logger = logging.getLogger(__name__)

dependencies = {
    'f1_m': f1_m,
    "matt_coeff": matt_coeff,
    "precision_m": precision_m,
    "recall_m": recall_m,
}


def evaluate_models(args, model_dict, test_dataset):
    """
    Evaluate the models on the test dataset.
    """

    for m_name, model in model_dict.items():
        args.model_dir = os.path.join(args.output_dir, f"{m_name}")
        if args.wandb:
            dir_name = args.output_dir.split('/')[-1]
            project_name = f"cropdis-{dir_name}"
            cfg = {
                "nbr_classes": args.n_classes,
                "class_names": args.class_names,
                "class_type": args.class_type,
                "test_set_len": args.len_test,
                "batch_size": args.batch_size,
                "nbr_test_batches": args.nbr_test_batch,
            }
            run = wandb.init(project=project_name,
                             job_type="infer", name=m_name, config=cfg, reinit=True)
            #wandb.init() returns a run object, and you can also access the run object via wandb.run:
            assert run is wandb.run

        logger.info("\n")
        logger.info(f"  ***** Evaluating {m_name} Validation set *****")
        print(model.summary())
        compute_training_metrics(args, model, m_name, test_dataset)

        if args.wandb:
            wandb.run.finish()
            print("\n\n--- FINISH WANDB RUN ---\n")


def main():
    args = parse_args()
    # SET LOGGING
    set_logging(args, 'infer')
    # SET SEED
    set_seed(args)

    args.input_shape = (224, 224, 3)
    args.transformer = False

    if args.xp_dir == 'resources/best_models/cnn/128':
        args.input_shape = (128, 128, 3)
        ds_path = 'resources/datasets/augm_disease_60343_ds_128.h5'

    elif args.xp_dir == 'resources/best_models/cnn/224':
        ds_path = 'resources/datasets/augm_disease_60343_ds_224.h5'

    elif args.xp_dir == 'resources/best_models/lab/128':
        args.input_shape = (128, 128, 3)
        ds_path = 'resources/datasets/augm_disease_60343_ds_224.h5'

    elif args.xp_dir == 'resources/best_models/lab/224':
        ds_path = 'resources/datasets/augm_lab_disease_60343_ds_224.h5'

    elif args.xp_dir == 'resources/best_models/transformers/VIT':
        ds_path = "../block_storage/transformers/vit"
        args.transformer = True

    elif args.xp_dir == 'resources/best_models/transformers/ConvNexT':
        ds_path = "../block_storage/transformers/convnext"
        args.transformer = True

    elif args.xp_dir == 'resources/best_models/transformers/Swin':
        ds_path = "../block_storage/transformers/swin"
        args.transformer = True

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
        args.class_names = [str(v) for k, v in id2label.items()]


    ## Create Dataset
    if args.transformer:
        args.input_shape = (3, 224, 224)

        test_set = load_from_disk(f'{ds_path}/test')
        args.len_test = test_set.num_rows
        data_collator = DefaultDataCollator(return_tensors="tf")
        test_set = test_set.to_tf_dataset(
            columns=['pixel_values'],
            label_cols=["labels"],
            shuffle=True,
            batch_size=32,
            collate_fn=data_collator)
    else:
        # Load the dataset
        X_test, y_test = load_split_hdf5(ds_path, 'test')
        # Set parameters
        args.len_test = len(X_test)
        test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        del X_test, y_test
        gc.collect()

    args.nbr_test_batch = int(math.ceil(args.len_test / args.batch_size))
    test_set = prep_ds_input(
        args, test_set, args.len_test, args.input_shape[0:2])
    for elem, label in test_set.take(1):
        img = elem[0].numpy()
        logger.info(f"batch shape is {elem.shape}, type is {elem.dtype}")
        logger.info(f"image shape is {img.shape}, type: {img.dtype}")
        logger.info(f"label shape is {label.shape} type: {label.dtype}")

    logger.info(f"  ---- Evaluation Parameters ----\n\n{args}\n\n")
    logger.info(f"  ***** Running Evaluation *****")
    logger.info(f"  test_set = {test_set}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Nbr training examples = {args.len_test}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Nbr of test batch = {args.nbr_test_batch}")
    logger.info(f"  Class names = {args.class_names}")

    model_dict = dict()
    if args.transformer:
        model_dir = args.xp_dir.split('/')[-1]
        logger.info(f"Model : {model_dir}")
        model = tf.keras.models.load_model(
            f"{args.xp_dir}/model-best.h5", custom_objects={"AdamWeightDecay": AdamWeightDecay})
        model_dict[model_dir] = model
    else:
        for model_dir in os.listdir(args.xp_dir):
            # Load the trained model saved to disk
            logger.info(f"Model : {model_dir}")

            if os.path.isdir(f'{args.xp_dir}/{model_dir}'):
                if model_dir == 'ConvNexT_Keras':
                    model = tf.keras.models.load_model(
                        f"{args.xp_dir}/{model_dir}/model-best.h5", custom_objects={'f1_m': f1_m, 'LayerScale': LayerScale})
                elif "lab" in model_dir:
                    model = tf.keras.models.load_model(
                        f"{args.xp_dir}/{model_dir}/model-best.h5", custom_objects={'f1_m': f1_m, 'CopyChannels': CopyChannels})
                else:
                    model = tf.keras.models.load_model(
                        f"{args.xp_dir}/{model_dir}/model-best.h5", custom_objects=dependencies)
                model_dict[model_dir] = model

    evaluate_models(args, model_dict, test_set)


if __name__ == '__main__':
    main()
