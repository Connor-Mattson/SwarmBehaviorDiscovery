from novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.novelty.GeneRule import GeneRule
from novel_swarms.config.OutputTensorConfig import OutputTensorConfig
import numpy as np
from generation.halted_evolution import HaltedEvolution
from novel_swarms.novelty.NoveltyArchive import NoveltyArchive
import cv2
import pygame
from sklearn.neighbors import NearestNeighbors

class ModifiedNoveltyArchieve(NoveltyArchive):
    def __init__(self, max_size=None, pheno_file=None, geno_file=None):
        super().__init__(max_size, pheno_file, geno_file)
        self.randoms = np.array([])

    # def getNovelty(self, k, vec):
    #     # Compute with k = k + 1 because the value vec is in the archive space
    #     distances = self.kNearestDistances(k + 1, vec)
    #     random_novelty = sum(self.getKNearestRandoms(5, vec)) / 5
    #     novelty_score = sum(distances) / k
    #     ALPHA, BETA = 0.5, 0.5
    #     blended_novelty = (ALPHA * novelty_score) + (BETA * random_novelty)
    #     print(f"BLENDING: {blended_novelty}")
    #     return blended_novelty

    def setRandoms(self, random_archive):
        self.randoms = []
        for i in random_archive.archive:
            self.randoms.append(self.archive[int(i)])
        self.randoms = np.array(self.randoms)

    def getKNearestRandoms(self, k, vec):
        if k > len(self.randoms):
            return [0]
        query = np.array([vec])
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.randoms)
        distances, _ = nbrs.kneighbors(query)
        return distances[0]

class ModifiedHaltingEvolution(HaltedEvolution):
    def __init__(self,
                 world: RectangularWorldConfig,
                 output_config: OutputTensorConfig,
                 evolution_config: GeneticEvolutionConfig,
                 screen=None):
        super().__init__(world, output_config, evolution_config, screen)
        self.archive = ModifiedNoveltyArchieve()

    def restart_screen(self):
        pygame.init()
        pygame.display.set_caption("Evolutionary Novelty Search")
        screen = pygame.display.set_mode((self.world.w, self.world.h))

        self.output_configuration = OutputTensorConfig(
            timeless=True,
            total_frames=80,
            steps_between_frames=2,
            screen=screen
        )

    def next(self):
        output, behavior = self.behavior_discovery.runSinglePopulation(
            screen=None,
            i=self.behavior_discovery.curr_genome,
            seed=self.world.seed,
            output_config=self.output_configuration,
            save=False
        )
        genome = self.behavior_discovery.population[self.behavior_discovery.curr_genome]

        self.behavior_discovery.curr_genome += 1
        return output, genome, behavior

    def miniNext(self):
        img, genome, behavior = self.next()
        img = self.resize(img)
        return img, genome, behavior

    def getPopulation(self):
        return self.behavior_discovery.population

    def resize(self, img, size=(200, 200)):
        return cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

    def overwriteBehavior(self, behavior):
        self.behavior_discovery.behavior = behavior

    def overwriteArchive(self, archive, randoms=None):
        self.behavior_discovery.archive = archive
        if randoms is not None:
            self.behavior_discovery.archive.setRandoms(randoms)

    def evolve(self):
        self.behavior_discovery.curr_genome = 0
        self.behavior_discovery.evaluate()

        print(self.archive.__class__)
        print(self.behavior_discovery.scores)

        self.behavior_discovery.evolve()

    def saveArchive(self, name):
        self.behavior_discovery.archive.saveArchive(f"{name}_b_")
        self.behavior_discovery.archive.saveGenotypes(f"{name}_g_")

    def close(self):
        pass

    @staticmethod
    def defaultEvolver(steps=1200, evolve_population=100, k_samples=15, n_agents=30, seed=None):
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
            seed=seed,
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
