import time
import os
from absl import flags
from absl import app
from absl import logging
from src.constants import ROBOT_TYPES, STRATEGY_TYPES
from src.training.hil_training import human_label_training
from src.generation.evolve_cluster import evolve_and_cluster

FLAGS = flags.FLAGS
flags.DEFINE_integer("gen", 100, "The number of generations to evolve controllers for", lower_bound=0)
flags.DEFINE_integer("pop", 100, "The number of controllers to evolve for each generation", lower_bound=0)
flags.DEFINE_integer("lifespan", 1200, "Simulation Horizon (timesteps)", lower_bound=0)
flags.DEFINE_integer("agents", 24, "Number of Agents", lower_bound=0)
flags.DEFINE_integer("seed", None, "Random Seed", lower_bound=0)
flags.DEFINE_integer("k", 12, "k value used in k-Medoids Calculation", lower_bound=0)
flags.DEFINE_bool("heuristic", True, "Whether to use the heuristic filter in evolution")
flags.DEFINE_float("cr", 0.7, "Crossover Rate", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("mr", 0.15, "Mutation Rate", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_enum("strategy", "Mattson_and_Brown", STRATEGY_TYPES, "The type of evolutionary strategy to use")
flags.DEFINE_enum("type", None, ROBOT_TYPES, "The robot capability model to consider")
flags.DEFINE_string("checkpoint", None, "Latent Embedding Model to use for Evolution")

def main(_):
    gen, pop, cr, mr = FLAGS.gen, FLAGS.pop, FLAGS.cr, FLAGS.mr
    logging.info(f"Evolving {gen} generations @ {pop} population with CR={cr}, MR={mr}")
    if FLAGS.strategy == "Mattson_and_Brown":
        if FLAGS.checkpoint is None:
            raise Exception("Mattson_and_Brown evolutionary approach requires the parameter 'checkpoint', a path to a trained latent embedding model.")

    name = f"{FLAGS.strategy}_{FLAGS.type}_{gen}_{pop}_{int(time.time())}"

    # Human-in-the-loop Training
    try:
        evolve_and_cluster(name, FLAGS.type, FLAGS.checkpoint, gen, pop, cr, mr, FLAGS.k, FLAGS.seed, FLAGS.agents, FLAGS.lifespan)
    except Exception as e:
        logging.warning("Exception Raised in Evolution. Evolution and Clustering Terminated.")
        raise e


if __name__ == "__main__":
    flags.mark_flag_as_required("type")
    app.run(main)
