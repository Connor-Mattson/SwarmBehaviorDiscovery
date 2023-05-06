import time
import os
from absl import flags
from absl import app
from absl import logging
from src.constants import ROBOT_TYPES
from src.generation.create_dataset import create_dataset

FLAGS = flags.FLAGS
flags.DEFINE_integer("size", 10000, "The number of simulated swarms in the dataset", lower_bound=0)
flags.DEFINE_integer("lifespan", 1200, "The timestep horizon of the simulation", lower_bound=0)
flags.DEFINE_integer("agents", 24, "Number of Agents", lower_bound=0, upper_bound=50)
flags.DEFINE_enum("type", None, ROBOT_TYPES, "A specified Robot Capability Model")
flags.DEFINE_bool("heuristic_filter", True, "Whether to explicitly filter out controllers that may be random or uninteresting")
flags.DEFINE_string("dataset_name", None, "A name that will be assigned to the output directory")

def main(_):
    robot = FLAGS.type
    experiment_name = f"{robot}-{int(time.time())}" if FLAGS.dataset_name is None else FLAGS.dataset_name
    logging.info(f"Generating Swarm Trajectories for Robot Type \"{robot}\" in dir ./data/{experiment_name}")

    # Generate Data
    export_to = os.path.join("data/", experiment_name)
    if not os.path.exists(export_to):
        logging.info(f"Creating output directory at {export_to}")
        os.mkdir(export_to)

    try:
        create_dataset(export_to, robot_type=robot, horizon=FLAGS.lifespan, n_agents=FLAGS.agents, filter=FLAGS.heuristic_filter, size=FLAGS.size)
    except Exception as e:
        logging.warning("Exception Raised in Dataset Creation. Dataset Creation Terminated.")
        raise e

    logging.info(f"Dataset Successfully Created at {export_to}")

if __name__ == "__main__":
    flags.mark_flag_as_required("type")
    app.run(main)
