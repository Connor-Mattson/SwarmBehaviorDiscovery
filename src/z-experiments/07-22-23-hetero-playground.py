"""
Playground for heterogeneous worlds.
Author: Jeremy Clark
"""
import math
from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld
from src.ui.img_process import get_image_map


def truncate_controller(c):
    """
    Truncate the decimals on the given controller so that it can be displayed conveniently
    """
    return [round(n, 2) for n in c]


def generate_heterogeneous_world(control):
    """
    Helper function for _generate_world_config

    :param control: The given 9-element long controller
    :return: A world config for heterogeneous controllers
    """
    pop = 30  # Population of agents
    rat = abs(control[0])  # Ratio of agent 1 to agent 2
    # Calculate populations of the two groups
    p1 = math.floor(rat * pop)
    p2 = pop - p1
    # Controllers of the two groups are embedded in the parent controller
    c1, c2 = control[1:5], control[5:9]
    sensors = SensorSet([BinaryLOSSensor(angle=0)])
    # sensors = SensorSet([BinaryFOVSensor(degrees=True, theta=15)])
    a1_config = DiffDriveAgentConfig(controller=c1, sensors=sensors, seed=None, body_color=(200, 0, 0))
    a2_config = DiffDriveAgentConfig(controller=c2, sensors=sensors, seed=None, body_color=(0, 200, 0))
    agent_config = HeterogeneousSwarmConfig()
    agent_config.add_sub_populuation(a1_config, p1)
    agent_config.add_sub_populuation(a2_config, p2)
    config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        seed=None,
        behavior=[],
        agentConfig=agent_config,
        padding=15,
        show_walls=True,
        collide_walls=True
    )
    return RectangularWorld(config)


interesting_behaviors = [
    [0.602, 0.326, -0.579, 0.533, 0.472, 0.293, 0.424, 0.817, 0.795],  # Opposing cyclic pursuit
    [0.142, -0.151, 0.864, -0.841, -0.706, 0.649, -0.893, 0.629, 0.861],  # Flail
    [0.615, -0.461, -0.248, -0.935, 0.108, 0.624, 0.229, 0.403, 0.738],  # Guards
    [0.529, 0.322, 0.0437, -0.238, -0.732, -0.678, 0.367, 0.342, 0.793],  # Aggregation in middle, walls outside
    [0.734, 0.256, -0.735, 0.297, 0.494, 0.791, 0.596, 0.835, -0.085]  # IDK how to describe this but it's cool
]

if __name__ == '__main__':
    for trial, controller in enumerate(interesting_behaviors):
        controller[0] = 0
        truncated_controller = truncate_controller(controller)
        truncated_controller_string = f"[{', '.join([str(x) for x in truncated_controller])}]"
        print(f"Trial {trial}: {truncated_controller_string}")
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for i in range(21):
            world = generate_heterogeneous_world(controller)
            get_image_map(
                controller,
                "gif",
                frame_start=2500,
                filepath=f"/home/jeremy/Desktop/ratio-phase/trial-{trial}/{alphabet[i]}-{controller[0]}.gif",
                world=world
            )
            controller[0] += 0.05
