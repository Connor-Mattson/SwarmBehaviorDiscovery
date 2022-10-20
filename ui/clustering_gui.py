import numpy
import numpy as np
import pygame
import os
import matplotlib
from PIL import Image
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from NovelSwarmBehavior.novel_swarms.results.Cluster import Cluster
from NovelSwarmBehavior.novel_swarms.config.ResultsConfig import ResultsConfig
from NovelSwarmBehavior.novel_swarms.novelty.NoveltyArchive import NoveltyArchive


class ClusteringGUI(Cluster):
    def __init__(self, config: ResultsConfig, auto_quit=False, output_folder_name=None):
        super().__init__(config)
        self.running = True
        self.auto_quit = auto_quit
        self.output_folder_name = output_folder_name

    def initTSNE(self):
        pass

    def clustering(self):
        kmedoids = KMedoids(n_clusters=self.results_config.k, random_state=0).fit(self.archive.archive)
        self.cluster_indices = kmedoids.labels_
        self.medoid_indices = kmedoids.medoid_indices_
        self.high_dim_medoids = kmedoids.cluster_centers_
        self.cluster_medoids = [[] for _ in self.high_dim_medoids]
        self.medoid_genomes = [[] for _ in self.high_dim_medoids]
        for i, j in enumerate(self.medoid_indices):
            self.medoid_genomes[i] = self.archive.genotypes[j]
        self.postTSNE()

    def postTSNE(self):
        self.reduced = TSNE(
            n_components=2,
            learning_rate="auto",
            init="pca",
            perplexity=self.results_config.perplexity,
            early_exaggeration=self.results_config.early_exaggeration
        ).fit_transform(self.archive.archive)

        for i, index in enumerate(self.medoid_indices):
            self.cluster_medoids[i] = self.reduced[index]
    def quit(self):
        self.running = False
    def runDisplayLoop(self, screen):
        self.running = True
        saved = False
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    self.clickInGUI(pos)

            screen.fill((0, 0, 0))
            for cluster_point in self.point_population:
                cluster_point.draw(screen)

            for cluster_center in self.cluster_medoids:
                pygame.draw.circle(screen, (255, 255, 255), (int(cluster_center[0]), int(cluster_center[1])),
                                   self.MEDOID_RADIUS, width=0)

            pygame.display.flip()

            if self.auto_quit and not saved:
                par_dir = f"./data/clusters/{self.output_folder_name}"
                if not os.path.isdir(par_dir):
                    os.mkdir(par_dir)
                screen = pygame.display.get_surface()
                screen_capture = pygame.surfarray.array3d(screen)
                name = len(os.listdir(par_dir))
                im2 = Image.fromarray(screen_capture.astype(np.uint8))
                im2.save(f'{par_dir}/epoch_{name}.png')
                saved = True
                self.running = False

