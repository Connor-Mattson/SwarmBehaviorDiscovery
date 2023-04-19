from sklearn.manifold import TSNE
import pygame
import torch
from sklearn_extra.cluster import KMedoids
import numpy as np
from generation.evolution import ModifiedNoveltyArchieve
from ui.class_similarity import SimilarityGUI
from networks.archive import DataAggregationArchive
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.novelty.NoveltyArchive import NoveltyArchive
from novel_swarms.config.ResultsConfig import ResultsConfig
from ui.clustering_gui import ClusteringGUI
import time
from PIL import Image
import os
import matplotlib
import cv2

class HIL:
    def __init__(self, name=None, synthetic=False, data_limiter=2000, clusters=8):
        if not name:
            self.name = f"{str(int(time.time()))}"
        else:
            self.name = name

        self.synthetic = synthetic
        self.synthetic_knowledge = None
        self.data_limiter = data_limiter
        self.clusters=clusters
        print("HIL Init!")

    def medoid_accuracy(self, medoid_indices):
        correct = 0
        total = 0
        for i, medoid_i in enumerate(medoid_indices):
            for j in range(i, len(medoid_indices)):
                medoid_i_class = self.synthetic_knowledge.labels_[medoid_i]
                medoid_j_class = self.synthetic_knowledge.labels_[medoid_indices[j]]
                total += 1
                if medoid_j_class != medoid_i_class:
                    correct += 1

        return correct / total

    def clustering_accuracy(self, class_labels):
        correct = 0
        total = 0
        validation_labels = self.synthetic_knowledge.labels_
        for i in range(len(class_labels)):
            for j in range(i, len(class_labels)):
                total += 1

                # If the classes do not match in the testing set, we check to see if they don't match in the validation set
                if class_labels[i] != class_labels[j] and validation_labels[i] != validation_labels[j]:
                    correct += 1

                # Pairs that are classified as the same in testing should also cluster into the same class in validation
                elif class_labels[i] == class_labels[j] and validation_labels[i] == validation_labels[j]:
                    correct += 1

        return correct / total

    def record_medoids(self, network, dataset, medoids=12, size=50):
        archive = ModifiedNoveltyArchieve()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network.eval()
        for i in range(len(dataset)):
            if i > self.data_limiter:
                break
            anchor_encoding, genome = dataset[i][0], dataset[i][1]
            anchor_encoding = self.resizeInput(anchor_encoding, size)
            anchor_encoding = torch.from_numpy(anchor_encoding).to(device).float()
            embedding = network(anchor_encoding.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            archive.addToArchive(vec=embedding, genome=genome)

        kmedoids = KMedoids(n_clusters=medoids, random_state=0).fit(archive.archive)
        medoids = kmedoids.medoid_indices_
        labels = kmedoids.labels_

        par_directory = f"./data/medoids/{self.name}"
        if not os.path.isdir(par_directory):
            os.mkdir(par_directory)
        trial_folder = f"{par_directory}/epoch_{len(os.listdir(par_directory))}"
        os.mkdir(trial_folder)
        for i, medoid_i in enumerate(medoids):
            medoid_image = dataset[medoid_i][0]
            im2 = Image.fromarray(medoid_image.astype(np.uint8))
            im2.save(f'{trial_folder}/medoid_{i}.png')

        text = ""
        for i, medoid_i in enumerate(medoids):
            genome = dataset[medoid_i][1]
            text += f"{str(genome)}\n"
        with open(f'{trial_folder}/genomes.txt', "w") as f:
            f.write(text)
            f.close()

        if self.synthetic:
            if not self.synthetic_knowledge:
                self.synthetic_knowledge = self.syntheticBehaviorSpace(dataset)

            print("Calculating Accuracy...")
            medoid_accuracy = self.medoid_accuracy(medoids)
            cluster_accuracy = self.clustering_accuracy(labels)
            print(f"Medoid Accuracy: {medoid_accuracy}, Cluster Accuracy: {cluster_accuracy}")
            return medoid_accuracy, cluster_accuracy

        return 0, 0

    def show_clusters(self, archive, auto_quit=False):
        agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT
        world_config = ConfigurationDefaults.RECTANGULAR_WORLD
        world_config.addAgentConfig(agent_config)
        config = ResultsConfig(archive=archive, k_clusters=self.clusters, world_config=world_config, tsne_perplexity=8,
                               tsne_early_exaggeration=12, skip_tsne=False)
        gui = ClusteringGUI(config, auto_quit=auto_quit, output_folder_name=self.name)
        gui.displayGUI()
        pygame.display.quit()
        pygame.quit()

    def embed_and_cluster(self, network, dataset, auto_quit=False, size=50):
        archive = ModifiedNoveltyArchieve()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network.eval()
        for i in range(len(dataset)):
            # if i > self.data_limiter:
            #     break
            anchor_encoding, genome = dataset[i][0], dataset[i][1]
            anchor_encoding = self.resizeInput(anchor_encoding, size)
            anchor_encoding = torch.from_numpy(anchor_encoding).to(device).float()
            embedding = network(anchor_encoding.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            archive.addToArchive(vec=embedding, genome=genome)
        self.show_clusters(archive, auto_quit=auto_quit)

    def syntheticBehaviorSpace(self, dataset):
        archive = ModifiedNoveltyArchieve()
        for i in range(len(dataset)):
            behavior, genome = dataset[i][2], dataset[i][1]
            archive.addToArchive(vec=behavior, genome=genome)
        kmedoids = KMedoids(n_clusters=8, random_state=0).fit(archive.archive)
        return kmedoids

    def syntheticMedoidSeparation(self, subject_behaviors, anchor_dataset):
        if not self.synthetic_knowledge:
            self.synthetic_knowledge = self.syntheticBehaviorSpace(anchor_dataset)
        output = self.synthetic_knowledge.predict(subject_behaviors)
        return output

    def syntheticBehaviorComparison(self, anchor_behavior, subject_behaviors, anchor_dataset):
        if not self.synthetic_knowledge:
            self.synthetic_knowledge = self.syntheticBehaviorSpace(anchor_dataset)

        anchor_b = self.synthetic_knowledge.predict([anchor_behavior])[0]
        subjects_b = self.synthetic_knowledge.predict(subject_behaviors)

        output = []
        for sub_class in subjects_b:
            if sub_class == anchor_b:
                output.append(0)
            else:
                output.append(1)

        return output

    def getEmbeddedArchive(self, dataset, network, concat_behavior=False, size=50):
        network.eval()
        archive = ModifiedNoveltyArchieve()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(len(dataset)):
            if i > self.data_limiter:
                break
            anchor_encoding, genome, behavior = dataset[i]
            anchor_encoding = self.resizeInput(anchor_encoding, size)
            anchor_encoding = torch.from_numpy(anchor_encoding).to(device).float()
            embedding = network(anchor_encoding.unsqueeze(0)).squeeze(0).cpu().detach().numpy()

            if concat_behavior:
                embedding = np.concatenate((embedding, behavior))

            archive.addToArchive(vec=embedding, genome=genome)

        return archive

    def humanInput(self, anchor_dataset, network, optim, loss_fn, data_archive, random_indices, stop_at=None, retrieve_last = 0):
        # Begin by clustering all known embeddings into n classes.
        # Sample from these clusters and use the medoids as the anchors to hopefully scoop up hard samples.
        SHOW_CLUSTERS = False
        RECORD_MEDOIDS = False

        CLUSTERS = self.clusters
        network.eval()
        archive = ModifiedNoveltyArchieve()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        keys = {
            pygame.K_KP_PLUS: -1,
            pygame.K_KP_PERIOD: -1,
            pygame.K_KP_0: 0,
            pygame.K_KP_1: 1,
            pygame.K_KP_2: 2,
            pygame.K_KP_3: 3,
            pygame.K_KP_4: 4,
        }

        gui_labels = {
            -1: "Random",
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
        }

        for i in range(len(anchor_dataset)):
            if i > self.data_limiter:
                break
            anchor_encoding, genome = anchor_dataset[i][0], anchor_dataset[i][1]
            anchor_encoding = torch.from_numpy(anchor_encoding).to(device).float()
            embedding = network(anchor_encoding.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            archive.addToArchive(vec=embedding, genome=genome)

        if SHOW_CLUSTERS:
            print("Showing Clusters!")
            self.show_clusters(archive, auto_quit=False)

        print("Clustering Archive Size: ", len(archive.archive))

        kmedoids = KMedoids(n_clusters=self.clusters, random_state=0).fit(archive.archive)
        labels = kmedoids.labels_
        medoids = kmedoids.medoid_indices_

        if RECORD_MEDOIDS:
            self.record_medoids(archive, medoids)

        network.train()
        user_help_cases, avg_loss, total_attempts = 0, 0, 1
        skip_clusters = []
        cluster_sampling = {}
        for cluster_class in range(CLUSTERS + 1):

            # Run Triplet Query on the Medoids themselves
            if cluster_class == 0:
                samples = [medoids[med] for med in range(0, len(medoids))]
                subject_images = [anchor_dataset[sampled_class][0] for sampled_class in samples]

                if not self.synthetic:
                    ui_input = SimilarityGUI(None, subject_images, keystrokes=keys, labels=gui_labels)
                    ui_input.run()
                    user_responses = np.array(ui_input.assignment)
                else:
                    subject_behaviors = [anchor_dataset[sampled_class][2] for sampled_class in samples]
                    user_responses = np.array(self.syntheticMedoidSeparation(subject_behaviors, anchor_dataset))

                # Assign all the skipped classes
                for i, res in enumerate(user_responses):
                    if res < 0:
                        skip_clusters.append(i)
                        random_indices.append_scalars([medoids[i]])

                # Run triplet loss on medoids.
                for i in range(8):
                    same_class = np.where(user_responses == i)[0]
                    different_class = np.where(user_responses != i)[0]
                    if len(same_class) < 2:
                        continue
                    for j in range(len(same_class)):
                        for k in range(i + 1, len(same_class)):
                            for l in range(len(different_class)):
                                anchor_image = anchor_dataset[medoids[j]][0]
                                pos_image = anchor_dataset[medoids[k]][0]
                                neg_image = anchor_dataset[medoids[l]][0]

                                optim.zero_grad()
                                anchor_out, pos_out, neg_out = network.network_with_transforms(anchor_image, pos_image, neg_image)

                                loss = loss_fn(anchor_out, pos_out, neg_out)
                                avg_loss += loss.item()
                                total_attempts += 1
                                if loss.item() > 0:
                                    loss.backward()
                                    optim.step()
                                    user_help_cases += 1

                                # Add to User Collected Archive
                                data_archive.append(medoids[j], medoids[k], medoids[l])
                                for random in skip_clusters:
                                    data_archive.append(medoids[j], medoids[k], medoids[random])

            else:
                cluster_class -= 1

                # Initialize Cluster Sampling
                cluster_sampling[cluster_class] = []

                is_random = cluster_class in skip_clusters
                if is_random:
                    continue

                examples = np.where(labels == cluster_class)[0]
                samples = np.random.choice(examples, min(10, len(examples)), False)

                anchor_image = anchor_dataset[medoids[cluster_class]][0]
                subject_images = [anchor_dataset[sampled_class][0] for sampled_class in samples]
                if len(subject_images) < 4:
                    continue

                if not self.synthetic:
                    ui_input = SimilarityGUI(anchor_image, subject_images)
                    ui_input.run()
                    user_responses = np.array(ui_input.assignment)
                else:
                    anchor_behavior = anchor_dataset[medoids[cluster_class]][2]
                    subject_behaviors = [anchor_dataset[sampled_class][2] for sampled_class in samples]
                    user_responses = np.array(self.syntheticBehaviorComparison(anchor_behavior, subject_behaviors, anchor_dataset))

                different_class = np.where(user_responses == 1)[0]
                same_class = np.where(user_responses == 0)[0]
                for i in range(len(different_class)):
                    for j in range(i, len(same_class)):
                        pos_index = same_class[j]
                        neg_index = different_class[i]
                        anchor_image = anchor_dataset[medoids[cluster_class]][0]
                        pos_image = anchor_dataset[samples[pos_index]][0]
                        neg_image = anchor_dataset[samples[neg_index]][0]

                        optim.zero_grad()
                        anchor_out, pos_out, neg_out = network.network_with_transforms(anchor_image, pos_image, neg_image)

                        loss = loss_fn(anchor_out, pos_out, neg_out)
                        avg_loss += loss.item()
                        total_attempts += 1
                        # print("User loss: ", loss)
                        if loss.item() > 0:
                            loss.backward()
                            optim.step()
                            user_help_cases += 1

                        # Add to User Collected Archive
                        data_archive.append(medoids[cluster_class], samples[pos_index], samples[neg_index])
                        cluster_sampling[cluster_class].append(
                            (medoids[cluster_class], samples[pos_index], samples[neg_index]))

                # if is_random:
                #     for index in same_class:
                #         random_indices.append_scalars([samples[index]])

        # Run Hierarchical Comparison in the cluster elements.
        for i in range(-1, 8):
            same_class = np.where(user_responses == i)[0]
            for j in range(len(same_class)):
                for k in range(i + 1, len(same_class)):
                    if j in cluster_sampling and k in cluster_sampling:
                        combined_samples = self.combine_hierarchy_samples(cluster_sampling[j], cluster_sampling[k])
                        for anc, pos, neg in combined_samples:
                            data_archive.append(anc, pos, neg)

        return user_help_cases, avg_loss / total_attempts, user_help_cases / total_attempts, archive

    def combine_hierarchy_samples(self, class_a, class_b):
        out = []
        for ancA, posA, negA in class_a:
            for ancB, posB, negB in class_b:
                if posB != posA:
                    out.append((ancA, posB, negA))
                    out.append((ancB, posA, negB))
                    out.append((posA, posB, negA))
                    out.append((posB, posA, negB))
        return out

    def resizeInput(self, X, w=200):
        frame = X.astype(np.uint8)
        resized = cv2.resize(frame, dsize=(w, w), interpolation=cv2.INTER_AREA)
        return resized
