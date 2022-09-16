import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms



from NovelSwarmBehavior.src.config.EvolutionaryConfig import GeneticEvolutionConfig
from NovelSwarmBehavior.src.config.WorldConfig import RectangularWorldConfig
from NovelSwarmBehavior.src.config.defaults import ConfigurationDefaults
from NovelSwarmBehavior.src.novelty.GeneRule import GeneRule
from NovelSwarmBehavior.src.config.OutputTensorConfig import OutputTensorConfig
from generation.HaltedEvolution import HaltedEvolution

# This script uses tensorboard, run
#    tensorboard --logdir=runs
# to launch tensorboard

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

    output_config = OutputTensorConfig(
        timeless=True,
        total_frames=80,
        steps_between_frames=1,
    )

    evolution = HaltedEvolution(
        world=world_config,
        evolution_config=novelty_config,
        output_config=output_config
    )

    return evolution


if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    evolution = initializeHaltedEvolution()
    evolution.setup()

    # writer.add_image('images', grid, 0)
    writer.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
