from utils import *
from prep_data_train import *
from preprocess_tensor import *
from metrics import *
import wandb

logger = logging.getLogger(__name__)


def evaluate_models(args, models, test_dataset):
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

    for model_dir in os.listdir(args.xp_dir):
        # Load the trained model saved to disk
        model = tf.keras.models.load_model(
            f"{args.xp_dir}/{model_dir}/best-checkpoint-f1/saved_model.pb'")

        if args.wandb:
            MODEL_NAME = model
            cfg = {
                "dataset": args.dataset,
                "nbr_train_epochs": args.n_epochs,
                "nbr_classes": args.n_classes,
                #"class_names": args.class_names,
                "class_type": args.class_type,
                #"class_weights": class_weights,
                "test_set_len": args.len_test,
                "batch_size": args.batch_size,
                "nbr_test_batches": args.nbr_test_batch,
            }
            run = wandb.init(project=f"Evaluate-{args.class_type}-best-models",
                         job_type="infer", name=MODEL_NAME, config=cfg)
            #wandb.init() returns a run object, and you can also access the run object via wandb.run:
            assert run is wandb.run

        compute_training_metrics(args, model, test_dataset, m_type='eval_test')
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

    # Load the dataset
    X_test, y_test = load_split_hdf5(args.dataset, 'test')

    if args.class_type == 'disease':
        args.label_map_path = '../resources/diseases_label_map.json'
    elif args.class_type == 'plants':
        args.label_map_path = '../resources/plants_label_map.json'
    else:
        args.label_map_path = '../resources/general_diseases_label_map.json'

    args.len_test = len(X_test)
    test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_set = prep_ds_input(args, test_set, 'test')
    best_models = ''
    evaluate_models(args, best_models, test_set)


if __name__ == '__main__':
    main()
