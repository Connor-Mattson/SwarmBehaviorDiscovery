import torch
import numpy as np
from novel_swarms.results.results import main as base_results
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.novelty.NoveltyArchive import NoveltyArchive
from data.swarmset import SwarmDataset
from ..networks.network_wrapper import NetworkWrapper
from ..constants import SINGLE_SENSOR_WORLD_CONFIG, TWO_SENSOR_WORLD_CONFIG


def brown_archive(dataset):
    arch = NoveltyArchive()
    for i in range(len(dataset)):
        controller, behavior = dataset[i][1], dataset[i][2]
        arch.addToArchive(behavior, controller)
    return arch

def mattson_archive(dataset, checkpoint=None, embedding_size=5):
    arch = NoveltyArchive()
    network = NetworkWrapper(output_size=embedding_size, margin=1.0, new_model=True)
    network.load_from_path(checkpoint)
    network.eval_mode()

    for i in range(len(dataset)):
        image, controller = dataset[i][0], dataset[i][1]
        image = np.expand_dims(image, axis=0)
        embed = network.batch_out(image)
        embed = embed.detach().cpu().squeeze(dim=0).numpy()
        arch.addToArchive(embed, controller)

    return arch

def visualize(data_path=None, labels=None, type=None, strategy=None, checkpoint=None, interactive=True, clustering=True, heterogeneous=False, k=10, embedding_size=5):
    sampled_dataset = SwarmDataset(data_path, rank=0)
    archive = None
    if strategy == "Mattson_and_Brown" and checkpoint is not None:
        archive = mattson_archive(sampled_dataset, checkpoint, embedding_size)
    else:
        archive = brown_archive(sampled_dataset)

    if interactive:
        results_config = ConfigurationDefaults.RESULTS
        results_config.archive = archive
        results_config.k = k
        results_config.world = SINGLE_SENSOR_WORLD_CONFIG if type == "single-sensor" else TWO_SENSOR_WORLD_CONFIG
        print(heterogeneous)
        base_results(results_config, heterogeneous=heterogeneous)

    print(archive.archive, archive.genotypes)

