import time
import os
from absl import flags
from absl import app
from absl import logging
from src.constants import ROBOT_TYPES, STRATEGY_TYPES
from src.viz.visualize_behaviors import visualize


FLAGS = flags.FLAGS
flags.DEFINE_enum("strategy", "Mattson_and_Brown", STRATEGY_TYPES, "The type of evolutionary strategy to use")
flags.DEFINE_enum("type", None, ROBOT_TYPES, "The robot capability model to consider")
flags.DEFINE_string("checkpoint", None, "Latent Embedding Model to use for Evolution (Required if strategy=Mattson_and_Brown)")
flags.DEFINE_string("data_path", None, "A path to the swarm simulation dataset")
flags.DEFINE_string("label_path", None, "A path to labels for color encoding (requires clustering=False)")
flags.DEFINE_bool("heterogeneous", False, "Whether to interpret controllers in a heterogeneous form")
flags.DEFINE_bool("clustering", True, "Whether to cluster the data using k-Medoids or not")
flags.DEFINE_bool("interactive", True, "Whether to create an interactive pygame window for selecting and viewing behaviors")
flags.DEFINE_integer("k", 10, "Number of clusters to form (assumes clustering=True)")
flags.DEFINE_integer("output_size", 5, "Size of embedding vector")


def main(_):
    logging.info(f"Visualizing Behavior Space for data {FLAGS.data_path} with method {FLAGS.strategy}")
    if FLAGS.strategy == "Mattson_and_Brown":
        if FLAGS.checkpoint is None:
            raise Exception("Mattson_and_Brown visualization approach requires the parameter 'checkpoint', a path to a trained latent embedding model.")

    if FLAGS.clustering and FLAGS.label_path is not None:
        raise Exception("Visualization expected either --clustering=True OR --label_path to be defined. Not both.")

    # Human-in-the-loop Training
    try:
        # Do Here
        visualize(data_path=FLAGS.data_path, labels=FLAGS.label_path, type=FLAGS.type, strategy=FLAGS.strategy, checkpoint=FLAGS.checkpoint,
                  interactive=FLAGS.interactive, clustering=FLAGS.clustering, heterogeneous=FLAGS.heterogeneous, k=FLAGS.k, embedding_size=FLAGS.output_size)
    except Exception as e:
        logging.warning("Exception Raised in Evolution. Evolution and Clustering Terminated.")
        raise e


if __name__ == "__main__":
    flags.mark_flag_as_required("type")
    flags.mark_flag_as_required("data_path")
    app.run(main)
