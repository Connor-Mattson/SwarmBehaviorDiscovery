import time
import os
from absl import flags
from absl import app
from absl import logging
from src.constants import ROBOT_TYPES
from src.training.hil_training import human_label_training

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 500, "The number of epochs to terminate training at", lower_bound=0)
flags.DEFINE_integer("seed", None, "The initialization seed for Numpy and Pytorch")
flags.DEFINE_float("lr", 3e-2, "Learning Rate", lower_bound=0.0)
flags.DEFINE_string("cp", None, "Model Checkpoint")
flags.DEFINE_string("data_path", None, "A path to the generated dataset")
flags.DEFINE_string("human_label_path", None, "A path to a human label file where some subset of elements in the dataset are labeled by class")
flags.DEFINE_string("experiment_name", None, "A name that will be assigned to the model checkpoint")

def main(_):
    data_path = FLAGS.data_path
    experiment_name = f"human-training-{int(time.time())}" if FLAGS.experiment_name is None else FLAGS.experiment_name
    logging.info(f"Training Network from dataset \"{data_path}\" using labels \"\" in dir ./checkpoints/{experiment_name}")

    par_dir = os.path.join("checkpoints")
    if not os.path.exists(par_dir):
        logging.info(f"Creating model checkpoint directory at {par_dir}")
        os.mkdir(par_dir)

    # Human-in-the-loop Training
    try:
        human_label_training(par_dir, experiment_name, data_path, FLAGS.cp, FLAGS.human_label_path, lr=FLAGS.lr, epochs=FLAGS.epochs, seed=FLAGS.seed)
    except Exception as e:
        logging.warning("Exception Raised in Self-Supervised Training. Training Terminated.")
        raise e

    logging.info(f"Model Checkpoint saved at ./checkpoints/{experiment_name}")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("human_label_path")
    app.run(main)
