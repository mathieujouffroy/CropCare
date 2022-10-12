import os
import yaml
import logging
import datetime
import argparse
import random
import numpy as np
import tensorflow as tf
import wandb

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class YamlNamespace(argparse.Namespace):
    """Namespace from a nested dict returned by yaml.load()"""

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x)
                        if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b)
                        if isinstance(b, dict) else b)


def set_logging(args, type="train"):
    "Defines the file in which we will write our training logs"

    date = datetime.datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
    if type == 'train':
        file_name = f"RUN_{date}.log"
    elif type == 'infer':
        file_name = f"INFER_{date}.log"
    
    if type == 'train':
        log_dir = os.path.join(args.output_dir, file_name)
    else:
        log_dir = file_name
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir)]
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def wandb_cfg(args, n_training_steps):
    # enforced max for this is ceil(NUM_VAL/batch_size)
    NUM_LOG_BATCHES = 32

    # SETUP WANDB
    config_dict = {
        "dataset": args.dataset,
        "nbr_train_epochs": args.n_epochs,

        "nbr_classes": args.n_classes,
        #"class_names": args.class_names,
        "class_type": args.class_type,
        #"class_weights": class_weights,

        "train_set_len": args.len_train,
        "valid_set_len": args.len_valid,
        "batch_size": args.batch_size,
        "nbr_train_batch": args.nbr_train_batch,
        "n_train_steps": n_training_steps,

        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "loss": args.loss,
    }

    max_log_batches = 1
    # change to max to log ALL the available images to a Table
    config_dict["num_log_batches"] = min(max_log_batches, NUM_LOG_BATCHES)

    return config_dict


def set_wandb_project_run(args, run_name):
    """ Initialize wandb directory to keep track of our models. """
    
    dir_name = args.output_dir.split('/')[-1]
    project_name = f"cropdis-{dir_name}"
    cfg = wandb_cfg(args, args.n_training_steps)
    run = wandb.init(project=project_name,
                     job_type="train", name=run_name, config=cfg, reinit=True)
    assert run is wandb.run


def parse_args():
    """ Parse training paremeters from config YAML file. """

    parser = argparse.ArgumentParser(
        description='Train a model for plant disease classification.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="The YAML config file")
    cli_args = parser.parse_args()

    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)

    assert os.path.isfile(config.dataset)

    return config
