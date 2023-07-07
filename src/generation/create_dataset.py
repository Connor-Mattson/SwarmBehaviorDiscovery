import time
import pygame
import numpy as np
from novel_swarms.novelty.BehaviorDiscovery import BehaviorDiscovery
from novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.novelty.GeneRule import GeneRule
from novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.sensors.GenomeDependentSensor import GenomeBinarySensor
from novel_swarms.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from src.generation.halted_evolution import HaltedEvolution
from data.swarmset import SwarmDataset, DataBuilder
from novel_swarms.novelty.GeneRule import GeneBuilder, GeneRule
from src.constants import SINGLE_SENSOR_SET, TWO_SENSOR_SET, TWO_SENSOR_GENE_MODEL, SINGLE_SENSOR_GENE_MODEL, \
    SINGLE_SENSOR_HETEROGENEOUS_MODEL, HETEROGENEOUS_SUBGROUP_BEHAVIOR


def create_dataset(out_path, size=10000, robot_type="single-sensor", horizon=1200, filter=True, n_agents=24,
                   heterogeneous=False, custom_controller_set=None, custom_world_seeds=None, seed=None):
    agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

    sensors = None
    gene_specifications = None
    if heterogeneous is True and robot_type == "single-sensor":
        sensors = SINGLE_SENSOR_SET
        gene_specifications = SINGLE_SENSOR_HETEROGENEOUS_MODEL

        agent_A = DiffDriveAgentConfig(controller=None, sensors=sensors, dt=1.0,
                                       body_color=(255, 0, 0))
        agent_B = DiffDriveAgentConfig(controller=None, sensors=sensors, dt=1.0,
                                       body_color=(0, 255, 0))
        h_config = HeterogeneousSwarmConfig()
        h_config.add_sub_populuation(agent_A, n_agents // 2)
        h_config.add_sub_populuation(agent_B, n_agents // 2)
        agent_config = h_config

    elif robot_type == "single-sensor":
        agent_config.sensors = SINGLE_SENSOR_SET
        gene_specifications = SINGLE_SENSOR_GENE_MODEL
    elif robot_type == "two-sensor":
        agent_config.sensors = TWO_SENSOR_SET
        gene_specifications = TWO_SENSOR_GENE_MODEL

    gene_specifications.heuristic_validation = filter

    if heterogeneous:
        phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR + HETEROGENEOUS_SUBGROUP_BEHAVIOR
    else:
        phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=n_agents,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15,
        show_walls=False,
        collide_walls=True
    )

    novelty_config = GeneticEvolutionConfig(
        gene_builder=gene_specifications,
        phenotype_config=phenotype,
        n_generations=100,
        n_population=100,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=horizon,
        display_novelty=False,
        save_archive=False,
        show_gui=True,
        use_external_archive=False,
    )

    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")
    screen = pygame.display.set_mode((world_config.w, world_config.h))

    output_config = OutputTensorConfig(
        timeless=True,
        total_frames=80,
        steps_between_frames=2,
        screen=screen,
        colored=True
    )

    halted_evolution = HaltedEvolution(
        world=world_config,
        evolution_config=novelty_config,
        output_config=output_config,
        heterogeneous=heterogeneous
    )

    baseline_data = DataBuilder(out_path, ev=halted_evolution, resize=(50, 50), seed=seed)
    baseline_data.create(sample_size=size, custom_pool=custom_controller_set, custom_seeds=custom_world_seeds)
    baseline_data.evolution.close()


def _create_dataset_from_controllers(out_path):
    controllers = [
        [-1.0, 1.0, 0.3, 0.2, 0.3, -0.9, 0.4, 1.0, 0.4],  # Segments
        [0.7, 1.0, 0.4, 0.5, -0.9, -0.4, -0.3, 0.6, 0.7],  # Nucleus
        [0.1, 0.9, 1.0, 0.8, 0.2, 0.7, -0.5, -0.1, 0.8],  # Containment
        [-1.0, 1.0, 0.3, 0.2, -1.0, 1.0, 0.6, 0.2, 0.3],  # Spiral
        [0.0, -0.7, 0.5, 0.6, 0.6, 0.5, 0.3, 0.4, 0.2],  # Nested Cycles
        [-0.7, -1.0, 1.0, -1.0, 1.0, 0.95, 1.0, 1.0, 0.2],  # Perimeter
        [-0.7, 0.3, 1.0, 1.0, 0.1, -0.1, 0.1, -0.1, 0.7],  # Site Traversal
        [-0.6, 1.0, 1.0, 0.4, 0.7, -0.6, 0.7, 1.0, 0.1],  # Flail
        [-0.12, -0.2, 1.0, -1.0, 0.8, 0.9, 0.5, 0.6, 0.3],  # Hurricane
        [-0.7, 0.7, -0.6, -1.0, 0.8, 0.1, 0.2, 0.5, 0.2],  # Snake
        [0.6, 0.7, 0.95, 0.95, 0.1, 0.7, 0.9, 0.9, 0.01],  # Leader
        [1.0, -1.0, 0.7, 0.5, 0.9, 0.7, -1.0, -0.2, 0.4],  # Dipole
        [-0.2, -1.0, 1.0, -1.0, -0.2, -1.0, 1.0, -1.0, 0.4],  # Aggregation
        [-0.7, 0.3, 1.0, 1.0, -0.7, 0.3, 1.0, 1.0, 0.4],  # Cyclic Pursuit
        [0.2, 0.7, -0.5, -0.1, 0.2, 0.7, -0.5, -0.1, 0.4],  # Dispersal
        [0.65, 1.0, 0.35, 0.5, 0.65, 1.0, 0.35, 0.5, 0.4],  # Milling
        [1.0, 0.95, 1.0, 1.0, 1.0, 0.95, 1.0, 1.0, 0.4],  # Wall Following
        [-0.83, -0.75, 0.27, -0.57, -0.83, -0.75, 0.27, -0.57, 0.4],  # Random
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],  # Nothing (Test)
    ]

    NUM_PER = 50
    data = []
    for c in controllers:
        for n in range(NUM_PER):
            data.append(c)

    # Was 5000
    create_dataset(out_path, heterogeneous=True, custom_controller_set=data, horizon=5000, seed=1)


if __name__ == "__main__":
    _create_dataset_from_controllers("../../data/mrs-q6-samples")
