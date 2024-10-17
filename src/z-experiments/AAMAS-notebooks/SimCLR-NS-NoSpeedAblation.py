from novel_swarms.behavior import AverageSpeedBehavior, AngularMomentumBehavior, RadialVarianceBehavior, \
    ScatterBehavior, GroupRotationBehavior
from novel_swarms.config.AgentConfig import HeroRobotConfig
from novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.novelty.GeneRule import GeneBuilder, GeneRuleContinuous
from novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.initialization.GridInit import GridInitialization
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.results.results import main as report
import os
import pygame
import time

import numpy as np
from simclr import SimCLR
from novel_swarms.novelty.OptimBehaviorDiscovery import OptimizedBehaviorDiscovery

def get_config():
    sensors = SensorSet([
        BinaryFOVSensor(theta=0.12, distance=100 * 3, show=False),
    ])

    agent_config = HeroRobotConfig(
        controller=None,
        sensors=sensors,
        trace_length=0,
        body_filled=True,
        dt=0.1,
    )

    # Create a Genotype Ruleset that matches the size and boundaries of your robot controller _max and _min represent
    # the maximum and minimum acceptable values for that index in the genome. mutation_step specifies the largest
    # possible step in any direction that the genome can experience during mutation.
    genotype = GeneBuilder([
        GeneRuleContinuous(_max=60.0, _min=-60.0, mutation_step=3, round_digits=3, exclude=[[-5, 5]]),
        GeneRuleContinuous(_max=3, _min=-3, mutation_step=0.6, round_digits=3, exclude=[[-0.1, 0.1]]),
        GeneRuleContinuous(_max=60.0, _min=-60.0, mutation_step=3, round_digits=3, exclude=[[-5, 5]]),
        GeneRuleContinuous(_max=3, _min=-3, mutation_step=0.6, round_digits=3, exclude=[[-0.1, 0.1]]),
    ])

    # Use the default Behavior Vector (from Brown et al.) to measure the collective swarm behaviors
    phenotype = [
        AverageSpeedBehavior(normalization_constant=6),
        AngularMomentumBehavior(normalization_constant=6),
        RadialVarianceBehavior(),
        ScatterBehavior(),
        GroupRotationBehavior(normalization_constant=6),
    ]

    # Define an empty Rectangular World with size (w, h) and n agents.
    num_agents = 8
    world_config = RectangularWorldConfig(
        size=(171 * 3, 142 * 3),
        n_agents=num_agents,
        seed=None,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15,
        init_type=GridInitialization(
            num_agents=num_agents,
            grid_size=(4, 3),
            bb=((25 * 3, 30 * 3), (146 * 3, 111 * 3))
        ),
        stop_at=None,
    )

    novelty_config = GeneticEvolutionConfig(
        gene_builder=genotype,
        phenotype_config=phenotype,
        n_generations=100,
        n_population=50,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=605,
        display_novelty=False,
        save_archive=True,
        show_gui=True,
        seed=3,
    )
    return novelty_config, world_config, genotype

if __name__ == "__main__":

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    save_results = True
    display_plots = True
    trial_name = f"{int(time.time())}"

    # Load the model
    model = SimCLR.load("~/Desktop/DevProjects/SwarmBehaviorDiscovery/src/z-experiments/Sum24-notebooks/checkpoints/embeddings/seed3.pt")
    model.eval()

    # Configure the Evolutionary Parameters
    config, world_config, gene_builder = get_config()
    evolution = OptimizedBehaviorDiscovery(
        generations=config.generations,
        population_size=config.population,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        world_config=config.world_config,
        lifespan=config.lifespan,
        k_neighbors=config.k,
        genome_builder=gene_builder,
        behavior_config=config.behavior_config,
        mutation_flip_chance=config.mutation_flip_chance,
        allow_external_archive=config.use_external_archive,
        genome_dependent_world=config.world_metadata,
        seed=config.seed,
        num_threads=12,
    )

    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")
    screen = pygame.display.set_mode((world_config.w, world_config.h))

    output_config = OutputTensorConfig(
        timeless=False,
        total_frames=9,
        steps_between_frames=30,
        screen=None
    )

    # Generation Loop
    archive = np.array([])
    controllers = np.array([])
    last_gen_timestamp = time.time()
    for generation in range(evolution.total_generations):
        evolution.curr_generation = generation

        # Run the entire generation all at once -- The money-maker for optimization
        o, b, c = evolution.runEntireGeneration(screen, output_config)

        # Relabel the most recent generation using the encoder model
        for i, (img, behavior, controller) in enumerate(zip(o, b, c)):
            embedding = model.encode(img[::3]).flatten().detach().cpu().numpy()

            if len(archive) == 0:
                archive = np.array([embedding])
                controllers = np.array([controller])
            else:
                archive = np.concatenate((archive, [embedding]))
                controllers = np.concatenate((controllers, [controller]))

            evolution.behavior = archive[-evolution.population_size:, :]
            evolution.population = controllers[-evolution.population_size:, :]
            evolution.archive.archive = archive
            evolution.archive.genotypes = controllers

        evolution.evaluate()
        evolution.evolve()
        current_time = time.time()
        print(f"Generation {generation} completed in {current_time - last_gen_timestamp}")
        last_gen_timestamp = current_time

        if save_results and config.save_every is not None:
            if generation % config.save_every == 0:
                evolution.archive.saveArchive(f"b_{trial_name}_{generation}")
                evolution.archive.saveGenotypes(f"g_{trial_name}_{generation}")

    if save_results:
        evolution.archive.saveArchive(f"b_{trial_name}_final")
        evolution.archive.saveGenotypes(f"g_{trial_name}_final")

    if display_plots:
        evolution.results()

    res_conf = ConfigurationDefaults.RESULTS
    res_conf.archive = evolution.archive
    res_conf.world = world_config
    report(res_conf)