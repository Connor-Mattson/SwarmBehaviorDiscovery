from src.generation.evolution import ModifiedNoveltyArchieve
from PIL import Image
from sklearn_extra.cluster import KMedoids
import cv2
import os
import numpy as np
import torch
from data.swarmset import ContinuingDataset, SwarmDataset
from src.networks.embedding import NoveltyEmbedding
from src.generation.evolution import ModifiedHaltingEvolution
from src.networks.archive import DataAggregationArchive
from src.hil.HIL import HIL
import time
from src.networks.ensemble import Ensemble
from src.networks.network_wrapper import NetworkWrapper, ResNetWrapper
from src.constants import DEFAULT_OUTPUT_CONFIG, TWO_SENSOR_GENE_MODEL, SINGLE_SENSOR_GENE_MODEL, SINGLE_SENSOR_WORLD_CONFIG, TWO_SENSOR_WORLD_CONFIG, \
    SINGLE_SENSOR_HETEROGENEOUS_MODEL, SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AGNOSTIC, SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE, HETEROGENEOUS_SUBGROUP_BEHAVIOR
from novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from novel_swarms.config.defaults import ConfigurationDefaults


def resizeInput(X, w=200):
    frame = X.astype(np.uint8)
    resized = cv2.resize(frame, dsize=(w, w), interpolation=cv2.INTER_AREA)
    return resized


def embed(image, behavior, network, concat_behavior=False, size=50, strategy="Mattson_and_Brown", species_aware=False):
    image = resizeInput(image, size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = None
    if network is not None:
        if strategy == "Mattson_and_Brown":
            network.network.eval()
            if species_aware:
                embedding = network.colored_input(image)
            else:
                image = torch.from_numpy(image).to(device).float()
                embedding = network.network(image.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        if strategy == "ResNet":
            if species_aware:
                embedding = network.forward(image).detach().cpu().squeeze(dim=0).numpy()
            else:
                embedding = network.forward(image, greyscale=True).detach().cpu().squeeze(dim=0).numpy()
    else:
        embedding = behavior

    if concat_behavior:
        embedding = np.concatenate((embedding, behavior))

    return embedding

def getEmbeddedArchive(dataset, network, concat_behavior=False, size=50, strategy="Mattson_and_Brown", species_aware=False):
    archive = ModifiedNoveltyArchieve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(len(dataset)):
        image, genome, behavior = dataset[i]
        embedding = embed(image, behavior, network, concat_behavior, size, strategy, species_aware)
        archive.addToArchive(vec=embedding, genome=genome)
    return archive


def record_medoids(network, dataset, medoids=12, size=50, name="Null"):
    archive = ModifiedNoveltyArchieve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(len(dataset)):
        anchor_encoding, genome, behavior = dataset[i][0], dataset[i][1], dataset[i][2]
        if network is not None:
            anchor_encoding = resizeInput(anchor_encoding, size)
            anchor_encoding = torch.from_numpy(anchor_encoding).to(device).float()
            network.eval()
            embedding = network(anchor_encoding.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            archive.addToArchive(vec=embedding, genome=genome)
        else:
            archive.addToArchive(vec=behavior, genome=genome)

    kmedoids = KMedoids(n_clusters=medoids, random_state=0).fit(archive.archive)
    medoids = kmedoids.medoid_indices_
    labels = kmedoids.labels_

    par_directory = name
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

    return 0, 0


def execute_from_model(MODEL):
    out_name = MODEL['out_name']
    print(f"Beginning Evolution with configuration = {MODEL}")
    if MODEL["checkpoint"]:
        network = NetworkWrapper(output_size=5, lr=0.0, margin=0.0, weight_decay=1e-6, new_model=True,
                                 manual_schedulers=True, dynamic_lr=True, warmup=5)
        network.load_from_path(MODEL["checkpoint"])
        network.eval_mode()
        print(f"Network Loaded ==> {network.__class__}")
    elif MODEL["strategy"] == "ResNet":
        network = ResNetWrapper()
        network.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        network = None

    out_config = None
    if MODEL["requires_image_out"]:
        out_config = DEFAULT_OUTPUT_CONFIG
        if MODEL["evaluate_sub_groups"]:
            out_config.colored = True

    p_dir = f"./evolution/{out_name}"
    os.makedirs(p_dir)
    dataset = ContinuingDataset(p_dir)
    config = MODEL["config"]
    evolution = ModifiedHaltingEvolution(
        evolution_config=MODEL["g_e"],
        world=MODEL["g_e"].world_config,
        output_config=out_config,
        heterogeneous=MODEL["heterogeneous"],
    )
    evolution.restart_screen()

    for gen in range(config["generations"]):
        print(f"Starting Generation: {gen}")
        for i in range(len(evolution.getPopulation())):
            print(f"Population Member: {i}")
            # The collection of the original behavior vector below is only used to collect data to compare with the baseline
            visual_behavior, genome, baseline_behavior = evolution.next()

            # dataset.new_entry(visual_behavior, genome, baseline_behavior)
            if MODEL["strategy"] == "Brown_et_al":
                embedding = baseline_behavior
            else:
                embedding = embed(visual_behavior, baseline_behavior, network,
                                  concat_behavior=True if MODEL["checkpoint"] and MODEL["concat"] else False,
                                  strategy=MODEL["strategy"], species_aware=MODEL["evaluate_sub_groups"])

            evolution.behavior_discovery.archive.addToArchive(embedding, genome)


        if MODEL["export_medoids"]:
            _, _ = record_medoids(network, dataset, medoids=MODEL["config"]["k"], name=out_name)

        # RECALCULATE_ARCHIVE_AT_EACH_GEN = False
        # if MODEL["strategy"] in ["Mattson_and_Brown", "ResNet"] and RECALCULATE_ARCHIVE_AT_EACH_GEN:
        #     embedded_archive = getEmbeddedArchive(dataset, network, concat_behavior=True if MODEL["checkpoint"] and MODEL["concat"] else False, strategy=MODEL["strategy"], species_aware=MODEL["evaluate_sub_groups"])
        #     evolution.overwriteArchive(embedded_archive)
        #     embedded_behavior = embedded_archive.archive[-evolution.evolve_config.population:]
        #     evolution.overwriteBehavior(embedded_behavior)
        # else:

        # Important: Overwrite the behavior_discovery's population behavior attr so that novelty scores can be calculated!
        evolution.overwriteBehavior(evolution.behavior_discovery.archive.archive[-evolution.evolve_config.population:])

        evolution.evolve()
        evolution.restart_screen()
        print(f"Evolution complete for gen{gen}")

    evolution.saveArchive(out_name)
    print("Completed Model.")


def evolve_and_cluster(name, _type, checkpoint=None, gen=100, pop=100, cr=0.7, mr=0.15, k=12, seed=None, agents=24,
                       lifespan=1200, heterogeneous=False, concat=False, export_medoids=False, evaluate_sub_groups=False,
                       strategy="Mattson_and_Brown"):

    gene_builder = None
    if heterogeneous:
        gene_builder = SINGLE_SENSOR_HETEROGENEOUS_MODEL
        world = SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE if evaluate_sub_groups else SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AGNOSTIC
    else:
        gene_builder = TWO_SENSOR_GENE_MODEL if _type == "two-sensor" else SINGLE_SENSOR_GENE_MODEL
        world = TWO_SENSOR_WORLD_CONFIG if _type == "two-sensor" else SINGLE_SENSOR_WORLD_CONFIG

    world.population_size = agents
    world.behavior = ConfigurationDefaults.BEHAVIOR_VECTOR if not evaluate_sub_groups else HETEROGENEOUS_SUBGROUP_BEHAVIOR
    model = {
        "out_name": name,
        "checkpoint": checkpoint,
        "heterogeneous": heterogeneous,
        "strategy": strategy,
        "requires_image_out": True if strategy == "Mattson_and_Brown" or strategy == "ResNet" else False,
        "config": {
            "generations": gen,
            "population": pop,
            "lifespan": lifespan,
            "agents": agents,
            "seed": seed,
            "k": k,
            "crossover_rate": cr,
            "mutation_rate": mr,
        },
        "concat": concat,
        "export_medoids": export_medoids,
        "evaluate_sub_groups": evaluate_sub_groups,
        "g_e": GeneticEvolutionConfig(
            gene_builder=gene_builder,
            # Potentially Remove this
            phenotype_config=ConfigurationDefaults.BEHAVIOR_VECTOR if not concat else HETEROGENEOUS_SUBGROUP_BEHAVIOR,
            n_generations=gen,
            n_population=pop,
            crossover_rate=cr,
            mutation_rate=mr,
            world_config=world,
            k_nn=k,
            simulation_lifespan=lifespan,
            display_novelty=False,
            save_archive=False,
            show_gui=True,
            use_external_archive=False,
            seed=seed,
        )
    }

    execute_from_model(model)
