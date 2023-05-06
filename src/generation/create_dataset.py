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
from src.generation.halted_evolution import HaltedEvolution
from data.swarmset import SwarmDataset, DataBuilder
from novel_swarms.novelty.GeneRule import GeneBuilder, GeneRule
from src.constants import SINGLE_SENSOR_SET, TWO_SENSOR_SET, TWO_SENSOR_GENE_MODEL, SINGLE_SENSOR_GENE_MODEL

def create_dataset(out_path, size=10000, robot_type="single-sensor", horizon=1200, filter=True, n_agents=24):

    agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

    sensors = None
    gene_specifications = None
    if robot_type == "single-sensor":
        sensors = SINGLE_SENSOR_SET
        gene_specifications = SINGLE_SENSOR_GENE_MODEL
    elif robot_type == "two-sensor":
        sensors = TWO_SENSOR_SET
        gene_specifications = TWO_SENSOR_GENE_MODEL

    agent_config.sensors = sensors
    gene_specifications.heuristic_validation = filter

    phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR
    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=n_agents,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
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
        use_external_archive=False
    )

    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")
    screen = pygame.display.set_mode((world_config.w, world_config.h))

    output_config = OutputTensorConfig(
        timeless=True,
        total_frames=80,
        steps_between_frames=2,
        screen=screen
    )

    halted_evolution = HaltedEvolution(
        world=world_config,
        evolution_config=novelty_config,
        output_config=output_config
    )

    baseline_data = DataBuilder(out_path, ev=halted_evolution, resize=(50,50))
    baseline_data.create(sample_size=size)
    baseline_data.evolution.close()