import math
import wandb
import os
import gc
import json
import logging
import pandas as pd
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
    score_dict = dict()
    for m_name, model in model_dict.items():
        if args.transformer:
            args.model_dir = args.output_dir
        else:
            args.model_dir = os.path.join(args.output_dir, f"{m_name}")
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)

        if args.wandb:
            dir_name = args.output_dir.split('/')[1]
            m_type = args.output_dir.split('/')[2]
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
        if args.transformer:
            print(model)
            y_probs = model.predict(test_dataset, batch_size=32)
            # y_probs shape is (9052, 38)
            y_preds = y_probs.argmax(axis=-1)
            # y_preds shape is (9052, 38)
            results = model.evaluate(test_dataset, batch_size=32)
            f1_sc = None
            roc_sc = None
        else:
            results, f1_sc, roc_sc = compute_training_metrics(
                args, model, m_name, test_dataset)

        logger.info(f"Result of evaluation:")
        logger.info(f"  {results}")
        logger.info(f"  loss: {results[0]}")
        logger.info(f"  acc: {results[1]}")

        metrics_d = dict(
            {
                'loss': results[0],
                'accuracy': results[1],
                'f1': f1_sc,
                'AUC': roc_sc,
            }
        )
        score_dict[m_name] = metrics_d
        if args.wandb:
            wandb.run.finish()

    score_df = pd.DataFrame.from_dict(score_dict, orient='index').reset_index()
    score_df = score_df.rename(columns={'index': 'model'})
    logger.info(f"scores:\n{score_df}")

    if args.wandb:
        run = wandb.init(project=project_name,
                         job_type="infer", name=f"{m_type}_evals", config=cfg, reinit=True)
        assert run is wandb.run
        score_tb = wandb.Table(data=score_df, columns=score_df.columns)
        wandb.run.log({f'{m_type}_Evaluations': score_tb})
        wandb.run.finish()


def main():
    args = parse_args()
    # SET LOGGING
    args.output_dir = os.path.join(args.output_dir, args.xp_dir.split('/')[2])
    set_logging(args, 'infer')
    # SET SEED
    set_seed(args)

    args.input_shape = (224, 224, 3)
    args.transformer = False

    if args.xp_dir == 'resources/best_models/cnn':
        args.input_shape = (128, 128, 3)
        ds_path = 'resources/datasets/augm_disease_60343_ds_128.h5'

    elif args.xp_dir == 'resources/best_models/keras_transformers':
        ds_path = 'resources/datasets/augm_disease_60343_ds_224.h5'

    elif args.xp_dir == 'resources/best_models/lab':
        args.input_shape = (128, 128, 3)
        ds_path = 'resources/datasets/augm_lab_disease_60343_ds_224.h5'

    elif args.xp_dir == 'resources/best_models/seg':
        args.input_shape = (128, 128, 3)
        ds_path = 'resources/datasets/segm_disease_60343_ds_128.h5'

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

    logger.info(f"  ---- Evaluation Parameters ----\n\n{args}\n\n")
    logger.info(f"  ***** Running Evaluation *****")
    logger.info(f"  test_set = {test_set}")
    for elem, label in test_set.take(1):
        img = elem[0].numpy()
        logger.info(f"  batch shape is {elem.shape}, type is {elem.dtype}")
        logger.info(f"  image shape is {img.shape}, type: {img.dtype}")
        logger.info(f"  label shape is {label.shape} type: {label.dtype}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Nbr training examples = {args.len_test}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Nbr of test batch = {args.nbr_test_batch}")
    logger.info(f"  Class names = {args.class_names}")

    model_dict = dict()
    if args.transformer:
        model_dir = args.xp_dir.split('/')[-1]
        logger.info(f"Model : {model_dir}")
        model_path = f"{args.xp_dir}/model-best.h5"
        size = os.path.getsize(model_path)
        logger.info(f"Model Size: {size:.2f} MB")
        model = tf.keras.models.load_model(
            model_path, custom_objects={"AdamWeightDecay": AdamWeightDecay})
        model_dict[model_dir] = model
    else:
        for model_dir in os.listdir(args.xp_dir):
            # Load the trained model saved to disk
            logger.info(f"Model : {model_dir}")

            if os.path.isdir(f'{args.xp_dir}/{model_dir}'):
                model_path = f"{args.xp_dir}/{model_dir}/model-best.h5"
                size = os.path.getsize(model_path)
                logger.info(f"Model Size: {size:.2f} MB")
                if model_dir == 'ConvNexTSmall':
                    model = tf.keras.models.load_model(
                        model_path, custom_objects={'f1_m': f1_m, 'LayerScale': LayerScale})
                elif args.xp_dir.split('/')[-1] == "lab":
                    model = tf.keras.models.load_model(
                        model_path, custom_objects={'f1_m': f1_m, 'CopyChannels': CopyChannels})
                else:
                    model = tf.keras.models.load_model(
                        model_path, custom_objects=dependencies)
                model_dict[model_dir] = model

    evaluate_models(args, model_dict, test_set)


if __name__ == '__main__':
    main()
