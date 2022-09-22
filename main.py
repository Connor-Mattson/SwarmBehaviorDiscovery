import numpy as np
import pygame
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plot

from NovelSwarmBehavior.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.novel_swarms.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.novel_swarms.config.defaults import ConfigurationDefaults
from NovelSwarmBehavior.novel_swarms.novelty.GeneRule import GeneRule
from NovelSwarmBehavior.novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from generation.HaltedEvolution import HaltedEvolution

# This script uses tensorboard, run
#    tensorboard --logdir=runs
# to launch tensorboard

class BehaviorIdentificationModel(torch.nn.Module):
    def __init__(self, n_classes=2):
        super(BehaviorIdentificationModel, self,).__init__()

        self.n_classes = n_classes

        self.conv1 = torch.nn.Conv2d(1, 1, 5, stride=2, padding=2)
        self.activation1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.activation2 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=2)
        self.activation3 = torch.nn.LeakyReLU()

        self.pooling = torch.nn.MaxPool2d(2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(3844, 400)
        self.activation5 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(400, 100)
        self.activation6 = torch.nn.LeakyReLU()
        self.classification_layer = torch.nn.Linear(100, self.n_classes)

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        # print(x.size())
        # x = self.conv3(x)
        # x = self.activation3(x)

        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation5(x)
        x = self.linear2(x)
        x = self.activation6(x)
        x = self.classification_layer(x)
        return x

    def increaseClassCount(self):
        self.n_classes += 1
        self.classification_layer = torch.nn.Linear(100, self.n_classes)

def initializeHaltedEvolution():
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
        n_agents=30,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
    )

    novelty_config = GeneticEvolutionConfig(
        gene_rules=genotype,
        phenotype_config=phenotype,
        n_generations=100,
        n_population=100,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=600,
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

    return halted_evolution


if __name__ == '__main__':

    data_transform_training = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomApply(
            transforms=[
                transforms.RandomAffine(
                    degrees=(0, 180),
                    translate=(0.0, 0.3),
                    scale=(0.5, 1.0)
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.6,
                    p=0.3
                )
            ],
            p=1.0
        ),
        transforms.ToTensor(),
    ])

    data_transform_testing = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_folder = torchvision.datasets.ImageFolder(root="./data/train", transform=data_transform_training)
    test_folder = torchvision.datasets.ImageFolder(root="./data/test", transform=data_transform_testing)

    train_folder = torch.utils.data.ConcatDataset([train_folder] * 5)

    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=4,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_folder,
        batch_size=4,
        shuffle=True
    )

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    epochs = 100
    model = BehaviorIdentificationModel(n_classes=2)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-6)

    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        # Training
        size = len(train_loader.dataset)
        loss_sum = 0
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        epoch_loss = loss_sum / len(train_loader)
        writer.add_scalar('Loss/Train', epoch_loss, global_step=epoch)
        print(f"Training loss: {epoch_loss:>7f}")

        # Testing (Only test every few epochs)
        if epoch % 1 == 0:
            size = len(test_loader.dataset)
            num_batches = len(test_loader)
            test_loss, accuracy = 0, 0

            with torch.no_grad():
                for X, y in test_loader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    soft_func = torch.nn.Softmax(dim=1)
                    softy = soft_func(pred)

                    for i, row in enumerate(softy):
                        if row.argmax() == y[i]:
                            accuracy += 1

                # print(pred, softy)

            test_loss /= num_batches
            accuracy /= size
            print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            writer.add_scalar('Loss/Test', test_loss, global_step=epoch)
            writer.add_scalar('Accuracy/Test', accuracy, epoch)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
