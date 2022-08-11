import math
from utils import *
from prep_data_train import *
from preprocess_tensor import *
from metrics import compute_training_metrics, f1_m
import wandb

logger = logging.getLogger(__name__)

dependencies = {
    'f1_m': f1_m,
}

def evaluate_models(args, model_dict, test_dataset):
    """
    Evaluate the model on the test dataset.

    Parameters:
        args():
        model_tupl():
        trained_model():
        test_dataset():
    """
    ## optional globals to modify
    ## set to a custom name to help keep your experiments organized
    #RUN_NAME = ""
    ## change this if you'd like start a new set of comparable Tables
    ## (only Tables logged to the same key can be compared)
    #TEST_TABLE_NAME = "test_result"
    #MODEL_NAME = model_tupl[0]
    #run = wandb.init(project="plant-disease-classification", job_type="inference")
    #model_at = run.use_artifact(MODEL_NAME + ":latest")
    #model_dir = model_at.download()
    #print("model: ", model_dir)
    #model = tf.keras.models.load_model(model_dir)
    #model.compile(optimizer=args.optimizer, loss=args.loss, metrics=args.metrics)
    # download latest version of test data
    #test_data_at = run.use_artifact(TEST_DATA_AT + ":latest")
    #test_dir = test_data_at.download()
    #test_dir += "/test/"

    for name, model in model_dict.items():
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
            #run = wandb.init(project=project_name,
            #             job_type="infer", name=MODEL_NAME, config=cfg, reinit=True)
            ##wandb.init() returns a run object, and you can also access the run object via wandb.run:
            #assert run is wandb.run

        logger.info("\n")
        logger.info(f"  ***** Evaluating {name} Validation set *****")
        #compute_training_metrics(args, model, test_dataset, m_type='eval_test')
        
        #wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)
        
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
            args.label_map_path = '../resources/label_maps/diseases_label_map.json'
        elif args.class_type == 'plants':
            args.n_classes = 14
            args.label_map_path = '../resources/label_maps/plants_label_map.json'
        else:
            args.n_classes = 14
            args.label_map_path = '../resources/label_maps/general_diseases_label_map.json'
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
        test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    test_set = prep_ds_input(args, test_set, args.len_test)
    for elem, label in test_set.take(1):
        img = elem[0].numpy()
        print(f"element shape is {elem.shape}, type is {elem.dtype}")
        print(f"image shape is {img.shape}, type: {img.dtype}")
        print(f"label shape is {label.shape} type: {label.dtype}")

    model_dict = dict()
    for model_dir in os.listdir(args.xp_dir):
        # Load the trained model saved to disk
        model = tf.keras.models.load_model(
                    f"{args.xp_dir}/{model_dir}/files/model-best.h5",
                    custom_objects=dependencies)
        print(model)
        model_dict[model_dir] = model
 
    #evaluate_models(args, model_dict, test_set)

if __name__ == '__main__':
    main()
