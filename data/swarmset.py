import matplotlib
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib import image
from torchvision.io import read_image
from generation.halted_evolution import HaltedEvolution
from NovelSwarmBehavior.novel_swarms.novelty.GeneRule import GeneBuilder
from PIL import Image
import os
import numpy as np
import time
import random
import cv2

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
    def __init__(self, parent_dir, rank=0, resize=None):
        if not os.path.isdir(parent_dir):
            raise Exception("The provided Dataset Directory Does not exist")
        self.dir = parent_dir
        self.resize = resize
        # self.data_folders = os.listdir(parent_dir)
        # to_remove = []
        # for i, obj in enumerate(self.data_folders):
        #     subdir = os.path.join(parent_dir, obj)
        #     if not os.path.isdir(subdir):
        #         to_remove.append(i)
        #     self.data_folders[i] = subdir

        ####
        # Remove to_removes eventually
        ####

        self._rank = rank

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        folder = os.path.join(self.dir, str(index))
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
        name = f"{len(self)}"
        path = os.path.join(self.dir, name)
        os.mkdir(path)
        print(self.resize)
        save_image = image
        if self.resize:
            frame = image.astype(np.uint8)
            save_image = cv2.resize(frame, dsize=self.resize, interpolation=cv2.INTER_AREA)
        matplotlib.image.imsave(f'{path}/behavior.png', save_image, cmap='gray')
        with open(os.path.join(path, "context.txt"), "x") as f:
            f.write(vecToCSVLine(genome))
            if behavior is not None:
                f.write(vecToCSVLine(behavior))
        # self.data_folders.append(path)

    def add_rank(self, index, item, is_array=False):
        with open(os.path.join(self.data_folders[index], "context.txt"), "a") as f:
            if is_array:
                f.write(vecToCSVLine(item))
            else:
                f.write(item)

    def add_image(self, index, image):
        save_image = image
        print(self.resize)
        if self.resize:
            frame = image.astype(np.uint8)
            save_image = cv2.resize(frame, dsize=self.resize, interpolation=cv2.INTER_AREA)
        if self._rank == 0:
            path = self.data_folders[index]
            matplotlib.image.imsave(os.path.join(path, "decoded.png"), save_image, cmap='gray')
        else:
            raise Exception("You cannot call add image when a decoded image has already been called!")



class DataBuilder:
    def __init__(self, data_dir, gene_builder=None, steps=3000, agents=30, ev=None, screen=None, resize=None):
        self.dataset = SwarmDataset(data_dir, resize=resize)

        if len(self.dataset) > 0:
            raise Exception("Requested to build a new dataset in folder that contains items")

        if not ev and (gene_builder is None or not isinstance(gene_builder, GeneBuilder)):
            raise Exception("DataBuilder must be supplied with a GeneBuilder Configuration. See novel_swarms/novelty/GeneRule")


        if not ev:
            self.evolution, self.screen = HaltedEvolution.defaultEvolver(steps=steps, n_agents=agents, gene_builder=gene_builder)
        else:
            self.evolution, self.screen = ev, screen
        self.gene_builder = self.evolution.behavior_discovery.gene_builder

    def create(self, sample_size=1000):
        TRIALS = 1
        pool = self.build_genome_pool(sample_size)
        for trial in range(TRIALS):
            for i, genome in enumerate(pool):
                output, behavior = self.evolution.simulation(genome)
                self.dataset.new_sample(output, genome, behavior)
                print(f"{(i*100) / len(pool)}% Complete")
        return self.dataset

    def build_genome_pool(self, sample_size=1000):
        gene_pool = []
        for sample in range(sample_size):
            gene_pool.append(self.gene_builder.fetch_random_genome())
        return gene_pool
