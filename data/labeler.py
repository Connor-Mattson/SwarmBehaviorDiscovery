import time

import pygame
import matplotlib.image
from NovelSwarmBehavior.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from NovelSwarmBehavior.novel_swarms.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.novel_swarms.config.defaults import ConfigurationDefaults
from NovelSwarmBehavior.novel_swarms.novelty.GeneRule import GeneRule
from NovelSwarmBehavior.novel_swarms.util.datasets.GenomeDataSet import GenomeDataSet
from NovelSwarmBehavior.novel_swarms.world.WorldFactory import WorldFactory
from generation.HaltedEvolution import HaltedEvolution


def main():
    agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

    phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
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

    TRAIN_LOCATION = "./eval/eval"
    TEST_LOCATION = "./eval/eval"
    TRAIN_FOR_EVERY_TEST = 4
    TIMESTEPS = 1000

    confirm_a = input(f"CONFIRM IO DESTINATIONS: {TRAIN_LOCATION} AND {TEST_LOCATION}")

    dataset = GenomeDataSet(file="./to_label.csv")

    for i, data in enumerate(dataset.data):
        world_config.agentConfig.controller = data
        world = WorldFactory.create(world_config)
        output = world.evaluate(TIMESTEPS, output_capture=output_config)

        t = time.time()
        if i % TRAIN_FOR_EVERY_TEST == 0:
            matplotlib.image.imsave(f'{TEST_LOCATION}/{t}.png', output, cmap='gray')
        else:
            matplotlib.image.imsave(f'{TRAIN_LOCATION}/{t}.png', output, cmap='gray')

    print("Done!")


if __name__ == "__main__":
    main()
