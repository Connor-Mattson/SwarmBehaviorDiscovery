import matplotlib
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib import image
from torchvision.io import read_image
from generation.halted_evolution import HaltedEvolution
from PIL import Image
import os
import numpy as np
import time
import random

def vecToCSVLine(vector):
    line = ""
    for i, val in enumerate(vector):
        line += str(val)
        if i < len(vector) - 1:
            line += ", "
    line += "\n"
    return line

def CSVLineToVec(line):
    line_list = line.strip().replace("\n", "").split(",")
    float_list = []
    for i in line_list:
        float_list.append(float(i))
    float_list = np.array(float_list)
    return float_list

class ContinuingDataset(Dataset):
    def __init__(self, directory, create=True, folder_name=None):
        if create:
            folder_name = f"trial-{str(int(time.time()))}"
            self.dir = directory
            self.base_path = os.path.join(directory, folder_name)
            self.image_path = os.path.join(self.base_path, "images")
            self.context_path = os.path.join(self.base_path, "context")
            os.mkdir(self.base_path)
            os.mkdir(self.image_path)
            os.mkdir(self.context_path)
        else:
            self.dir = directory
            self.base_path = os.path.join(directory, folder_name)
            self.image_path = os.path.join(self.base_path, "images")
            self.context_path = os.path.join(self.base_path, "context")

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, index):
        if index >= len(self):
            print(f"Attempted to Access item at index {index}, which is out of range")
            index = len(self) - 1
        image = np.array(Image.open(os.path.join(self.image_path, f"{index}.png")).convert('L'))
        context_path = os.path.join(self.context_path, f"{index}.txt")
        genome = None
        behavior = None
        with open(context_path, "r") as f:
            try:
                genome = CSVLineToVec(f.readline())
                behavior = CSVLineToVec(f.readline())
            except Exception as e:
                print(f"Could not read genome from file at index {index}")
        return image, genome, behavior

    def new_entry(self, image, genome, behavior):
        img_name = f"{len(self)}.png"
        context_name = f"{len(self)}.txt"
        img_path = os.path.join(self.image_path, img_name)
        matplotlib.image.imsave(img_path, image, cmap='gray')
        context_path = os.path.join(self.context_path, context_name)
        with open(context_path, "x") as f:
            f.write(vecToCSVLine(genome))
            f.write(vecToCSVLine(behavior))

class SwarmDataset(Dataset):
    def __init__(self, parent_dir, rank=0):
        if not os.path.isdir(parent_dir):
            raise Exception("The provided Dataset Directory Does not exist")
        self.dir = parent_dir
        self.data_folders = os.listdir(parent_dir)
        to_remove = []
        for i, obj in enumerate(self.data_folders):
            subdir = os.path.join(parent_dir, obj)
            if not os.path.isdir(subdir):
                to_remove.append(i)
            self.data_folders[i] = subdir

        ####
        # Remove to_removes eventually
        ####

        self._rank = rank

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, index):
        folder = self.data_folders[index]
        image = np.array(Image.open(os.path.join(folder, "behavior.png")).convert('L'))
        context_path = os.path.join(folder, "context.txt")
        with open(context_path, "r") as f:
            genome = CSVLineToVec(f.readline())
            behavior = CSVLineToVec(f.readline())
            encoded = None
            decoded_img = None
            embedding = None
            classification = None
            if self._rank >= 1:
                encoded = CSVLineToVec(f.readline())
                decoded_img = np.array(Image.open(os.path.join(folder, "decoded.png")).convert('L'))
            if self._rank >= 2:
                embedding = CSVLineToVec(f.readline())
            if self._rank >= 3:
                classification = int(f.readline())

        return image, genome, behavior, encoded, decoded_img, embedding, classification

    def set_rank(self, rank):
        self._rank = rank

    def new_sample(self, image, genome, behavior=None):
        name = f"{len(self.data_folders)}"
        path = os.path.join(self.dir, name)
        os.mkdir(path)
        matplotlib.image.imsave(f'{path}/behavior.png', image, cmap='gray')
        with open(os.path.join(path, "context.txt"), "x") as f:
            f.write(vecToCSVLine(genome))
            if behavior is not None:
                f.write(vecToCSVLine(behavior))
        self.data_folders.append(path)

    def add_rank(self, index, item, is_array=False):
        with open(os.path.join(self.data_folders[index], "context.txt"), "a") as f:
            if is_array:
                f.write(vecToCSVLine(item))
            else:
                f.write(item)

    def add_image(self, index, image):
        if self._rank == 0:
            path = self.data_folders[index]
            matplotlib.image.imsave(os.path.join(path, "decoded.png"), image, cmap='gray')
        else:
            raise Exception("You cannot call add image when a decoded image has already been called!")



class DataBuilder:
    def __init__(self, data_dir, is_anti=False, is_similar=False, fixed_a_b=False, steps=3000, agents=30, ev=None, screen=None):
        self.dataset = SwarmDataset(data_dir)
        self.is_anti = is_anti
        self.is_similar = is_similar
        self.fixed_a_b = fixed_a_b
        if len(self.dataset) > 0:
            raise Exception("Requested to build new dataset in folder that contains items")

        if not ev:
            self.evolution, self.screen = HaltedEvolution.defaultEvolver(steps=steps, n_agents=agents)
        else:
            self.evolution, self.screen = ev, screen

    def create(self):
        TRIALS = 1
        pool = self.build_genome_pool()
        for trial in range(TRIALS):
            for i, genome in enumerate(pool):
                output, behavior = self.evolution.simulation(genome)
                self.dataset.new_sample(output, genome, behavior)
                print(f"{(i*100) / len(pool)}% Complete")
        return self.dataset

    def build_genome_pool(self):
        gene_pool = []
        if self.fixed_a_b:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if self.is_anti:
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for i in range(50, 100, 1):
                a = (i * 0.01) - (0.11 if self.is_similar else 0)
                b = (a - 0.2)
                for off_d in directions:
                    for on_d in directions:
                        # Skip the boring stationary behaviors
                        if off_d == (1, -1) or off_d == (-1, 1):
                            if on_d == (1, -1) or on_d == (-1, 1):
                                continue
                        genome = [a * off_d[0], b * off_d[1], a * on_d[0], b * on_d[1]]
                        gene_pool.append(genome)

        # Meshgrid size 4
        # else:
        #     DENSITY = 100
        #     SAMPLE_SIZE = 5000
        #     r_0_s = np.linspace(-1.0, 1.0, num=DENSITY)
        #     r_1_s = np.linspace(-1.0, 1.0, num=DENSITY)
        #     l_0_s = np.linspace(-1.0, 1.0, num=DENSITY)
        #     l_1_s = np.linspace(-1.0, 1.0, num=DENSITY)
        #
        #     mesh = np.meshgrid(r_0_s, l_0_s, r_1_s, l_1_s)
        #     genomes = np.array(mesh).T.reshape(-1, 4)
        #     print(f"Size of Genome scope: {len(genomes)}")
        #
        #     index_sample = np.random.randint(len(genomes), size=SAMPLE_SIZE)
        #     gene_pool = [genomes[i] for i in index_sample]


        # Meshgrid size 10
        else:
            SAMPLE_SIZE = 10000
            for _ in range(SAMPLE_SIZE):
                gene_pool.append(
                    [
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2) - 1,
                        (random.random() * 2 * np.pi) - np.pi,
                        (random.random() * 2 * np.pi) - np.pi,
                    ]
                )

        return gene_pool
