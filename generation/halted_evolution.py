import pygame

from NovelSwarmBehavior.novel_swarms.novelty.BehaviorDiscovery import BehaviorDiscovery
from NovelSwarmBehavior.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.novel_swarms.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.novel_swarms.config.defaults import ConfigurationDefaults
from NovelSwarmBehavior.novel_swarms.novelty.GeneRule import GeneRule
from NovelSwarmBehavior.novel_swarms.config.OutputTensorConfig import OutputTensorConfig

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
            print("Evolving Genomes")
            self.behavior_discovery.evolve()
            self.behavior_discovery.curr_genome = 0

        output = self.behavior_discovery.runSinglePopulation(
            screen=None,
            i=self.behavior_discovery.curr_genome,
            seed=self.world.seed,
            output_config=self.output_configuration
        )
        behavior = self.behavior_discovery.behavior[self.behavior_discovery.curr_genome]
        genome = self.behavior_discovery.population[self.behavior_discovery.curr_genome]
        self.behavior_discovery.curr_genome += 1
        return output, behavior, genome

    def simulation(self, genome):
        output, behavior = self.behavior_discovery.runSinglePopulation(
            screen=None,
            save=False,
            genome=genome,
            seed=self.world.seed,
            output_config=self.output_configuration
        )
        return output, behavior

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

        halted_evolution = HaltedEvolution(
            world=world_config,
            evolution_config=novelty_config,
            output_config=output_config
        )

        return halted_evolution, screen
