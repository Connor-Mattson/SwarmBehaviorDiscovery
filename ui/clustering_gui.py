import numpy
import numpy as np
import pygame
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from NovelSwarmBehavior.novel_swarms.results.Cluster import Cluster
from NovelSwarmBehavior.novel_swarms.config.ResultsConfig import ResultsConfig
from NovelSwarmBehavior.novel_swarms.novelty.NoveltyArchive import NoveltyArchive


class ClusteringGUI(Cluster):
    def __init__(self, config: ResultsConfig):
        super().__init__(config)

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
