import random
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import numpy as np

from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld
from src.ui.img_process import get_selected_representations
from src.generation.heuristic_filtering import ControllerFilter


def generate_random_controller():
    while True:
        controller = [random.uniform(-1, 1) for _ in range(4)]
        valid = ControllerFilter.homogeneous_filter(controller)
        if valid:
            return controller


def get_scaling_factor(controller):
    abs_controller = [abs(x) for x in controller]
    scaling_factor = 1 / max(abs_controller)
    return scaling_factor


def get_scaled_controller(controller):
    scaling_factor = get_scaling_factor(controller)
    scaled_controller = [scaling_factor * x for x in controller]
    return scaled_controller


def get_world(controller):
    sensors = SensorSet([BinaryLOSSensor(angle=0)])
    agent_config = DiffDriveAgentConfig(
        controller=controller,
        sensors=sensors,
        seed=0,
    )
    config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        seed=0,
        behavior=[],
        agentConfig=agent_config,
        padding=15,
        show_walls=True,
        collide_walls=True
    )

    return RectangularWorld(config)


def generate_gifs():
    """
    Randomly generates 1000 homogeneous controllers.
    Scales all of them, and stores the unscaled versions' GIFs in one directory and
    the scaled versions in another folder.
    """
    number_of_images = 1000
    unscaled_pool = mp.Pool(processes=4)
    scaled_pool = mp.Pool(processes=4)
    for i in range(number_of_images):
        unscaled_controller = generate_random_controller()
        scaled_controller = get_scaled_controller(unscaled_controller)

        unscaled_pool.apply_async(
            get_selected_representations,
            args=(
                unscaled_controller,
                ["trail", "gif"],
                [f"/home/jeremy/Desktop/unscaled_trails/unscaled-{i}-{unscaled_controller}.png",
                 f"/home/jeremy/Desktop/unscaled_gifs/unscaled-{i}-{unscaled_controller}.gif"]
            )
        )
        scaled_pool.apply_async(
            get_selected_representations,
            args=(
                scaled_controller,
                ["trail", "gif"],
                [f"/home/jeremy/Desktop/scaled_trails/scaled-{i}-{unscaled_controller}.png",
                 f"/home/jeremy/Desktop/scaled_gifs/scaled-{i}-{unscaled_controller}.gif"]
            )
        )

    scaled_pool.close()
    scaled_pool.join()
    unscaled_pool.close()
    unscaled_pool.join()


def plot_comparison():
    directory = "/home/jeremy/Desktop/classifications/"
    behavior_names = ["aggregation", "cyclic-pursuit", "dispersal", "milling", "random", "wall-following"]
    unscaled_counts = []
    scaled_counts = []

    for name in behavior_names:
        unscaled_files = 0
        scaled_files = 0
        behavior_dir = os.path.join(directory, name)

        for file_name in os.listdir(behavior_dir):
            if "un" in file_name:
                unscaled_files += 1
            else:
                scaled_files += 1

        # Update counters
        unscaled_counts.append(unscaled_files)
        scaled_counts.append(scaled_files)

    x = np.arange(len(behavior_names))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, unscaled_counts, width, color='green', label='Unscaled Controllers')
    rects2 = ax.bar(x + width / 2, scaled_counts, width, color='red', label='Scaled Controllers')

    # Add some text for labels, title, and legend
    ax.set_ylabel('Number of Instances')
    ax.set_title('Comparison of behavior distribution between 1000 controllers')
    ax.set_xticks(x)
    ax.set_xticklabels(behavior_names)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    plot_comparison()
