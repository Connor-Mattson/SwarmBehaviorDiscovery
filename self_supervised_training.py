import time
import os
from absl import flags
from absl import app
from absl import logging
from src.constants import MINING_TYPES, NETWORK_TYPE
from src.training.self_training import self_supervised_training

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 500, "The number of epochs to terminate training at", lower_bound=0)
flags.DEFINE_integer("seed", None, "The initialization seed for Numpy and Pytorch")
flags.DEFINE_float("lr", 3e-2, "Learning Rate", lower_bound=0.0)
flags.DEFINE_string("data_path", None, "A path to the generated dataset")
flags.DEFINE_string("experiment_name", None, "A name that will be assigned to the model checkpoint")
flags.DEFINE_bool("save", True, "Whether to save the experiment as a checkpoint")
flags.DEFINE_bool("only_features", False, "Whether to just train the embedding layer (True) or train the entire network (False)")
flags.DEFINE_enum("mining_type", "Random", MINING_TYPES, "The type of mining method that will be used in self-supervised sampling")
flags.DEFINE_enum("network_type", "Scratch", NETWORK_TYPE, "The type of network to train from, vanilla CNN from scratch or pretrained ResNet with feature extraction")
flags.DEFINE_integer("output_size", 5, "Size of embedding vector")
flags.DEFINE_integer("batch_size", 4096, "Batch Size")


def main(_):
    data_path = FLAGS.data_path
    experiment_name = f"self-supervised-{int(time.time())}" if FLAGS.experiment_name is None else FLAGS.experiment_name
    logging.info(f"Training Network from dataset \"{data_path}\" in dir ./checkpoints/{experiment_name}")

    par_dir = os.path.join("checkpoints")
    if not os.path.exists(par_dir):
        logging.info(f"Creating model checkpoint directory at {par_dir}")
        os.mkdir(par_dir)

    # Self-Supervised Training
    try:
        self_supervised_training(par_dir, experiment_name, data_path, lr=FLAGS.lr, epochs=FLAGS.epochs,
                                 seed=FLAGS.seed, mining_method=FLAGS.mining_type, training_method=FLAGS.network_type,
                                 embedding_size=FLAGS.output_size, only_features=FLAGS.only_features, batch_size=FLAGS.batch_size)
    except Exception as e:
        logging.warning("Exception Raised in Self-Supervised Training. Training Terminated.")
        raise e

    logging.info(f"Model Checkpoint saved at {par_dir}/{data_path}")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    app.run(main)
