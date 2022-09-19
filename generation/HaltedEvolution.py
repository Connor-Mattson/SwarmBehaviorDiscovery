import pygame

from NovelSwarmBehavior.novel_swarms.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from NovelSwarmBehavior.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.novel_swarms.novelty.BehaviorDiscovery import BehaviorDiscovery

"""
This file runs evolution but halts and waits until hearing back from the end of the pipeline
regarding the quality of the genome that was outputted by this code
"""


class HaltedEvolution:
    def __init__(self,
                 world: RectangularWorldConfig,
                 output_config: OutputTensorConfig,
                 evolution_config: GeneticEvolutionConfig,
                 screen=None,
                 ):
        self.world = world
        self.output_configuration = output_config
        self.evolve_config = evolution_config
        self.screen = screen
        self.behavior_discovery = BehaviorDiscovery(
            generations=evolution_config.generations,
            population_size=evolution_config.population,
            genotype_rules=evolution_config.gene_rules,
            crossover_rate=evolution_config.crossover_rate,
            mutation_rate=evolution_config.mutation_rate,
            lifespan=evolution_config.lifespan,
            world_config=world,
            behavior_config=evolution_config.behavior_config,
            k_neighbors=evolution_config.k,
        )

    def setup(self):
        print("Hello World!")

    def close(self):
        pygame.quit()

    def next(self):

        if self.behavior_discovery.curr_generation > 0 and self.behavior_discovery.curr_generation % self.evolve_config.generations == 0:
            print("Evolution Concluded")
            return None, None

        if self.behavior_discovery.curr_genome > 0 and self.behavior_discovery.curr_genome % self.evolve_config.population == 0:
            self.behavior_discovery.evolve()
            self.behavior_discovery.curr_genome = 0

        output = self.behavior_discovery.runSinglePopulation(
            screen=None,
            i=self.behavior_discovery.curr_genome,
            seed=self.world.seed,
            output_config=self.output_configuration
        )
        self.behavior_discovery.curr_genome += 1

        behavior = self.behavior_discovery.behavior[-1]

        return output, behavior
