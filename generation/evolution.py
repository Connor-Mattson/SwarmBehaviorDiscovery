from NovelSwarmBehavior.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.novel_swarms.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.novel_swarms.config.defaults import ConfigurationDefaults
from NovelSwarmBehavior.novel_swarms.novelty.GeneRule import GeneRule
from NovelSwarmBehavior.novel_swarms.config.OutputTensorConfig import OutputTensorConfig
import numpy as np
from generation.halted_evolution import HaltedEvolution
import cv2
import pygame

class ModifiedHaltingEvolution(HaltedEvolution):
    def __init__(self,
                 world: RectangularWorldConfig,
                 output_config: OutputTensorConfig,
                 evolution_config: GeneticEvolutionConfig,
                 screen=None):
        super().__init__(world, output_config, evolution_config, screen)

    def next(self):
        output = self.behavior_discovery.runSinglePopulation(
            screen=None,
            i=self.behavior_discovery.curr_genome,
            seed=self.world.seed,
            output_config=self.output_configuration
        )
        genome = self.behavior_discovery.population[self.behavior_discovery.curr_genome]
        self.behavior_discovery.curr_genome += 1
        return output, genome

    def miniNext(self):
        img, genome = self.next()
        img = self.resize(img)
        return img, genome

    def resize(self, img, size=(200, 200)):
        return cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

    def evolve(self):
        self.behavior_discovery.evaluate()
        self.behavior_discovery.evolve()

    @staticmethod
    def defaultEvolver(steps=1200, evolve_population=100, k_samples=15, n_agents=30):
        agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

        genotype = [
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
            GeneRule(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=4),
        ]

        phenotype = ConfigurationDefaults.BEHAVIOR_VECTOR

        world_config = RectangularWorldConfig(
            size=(500, 500),
            n_agents=n_agents,
            behavior=phenotype,
            agentConfig=agent_config,
            padding=15
        )

        novelty_config = GeneticEvolutionConfig(
            gene_rules=genotype,
            phenotype_config=phenotype,
            n_generations=100,
            n_population=evolve_population,
            crossover_rate=0.7,
            mutation_rate=0.15,
            world_config=world_config,
            k_nn=k_samples,
            simulation_lifespan=steps,
            display_novelty=False,
            save_archive=False,
            show_gui=True
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

        halted_evolution = ModifiedHaltingEvolution(
            world=world_config,
            evolution_config=novelty_config,
            output_config=output_config
        )

        return halted_evolution, screen
