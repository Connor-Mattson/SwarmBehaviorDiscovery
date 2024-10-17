from novel_swarms.novelty.NoveltyArchive import NoveltyArchive
from novel_swarms.world.simulate import main as sim
from src.constants import SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE

if __name__ == "__main__":
    archive = NoveltyArchive(
        pheno_file="[Novelty Search Phenotype File Here]",
        geno_file="[Novelty Search Genotype File Here]",
        # EXAMPLE
        # pheno_file="NS5_s1_t1686745687_b__1686786653.csv",
        # geno_file="NS5_s1_t1686745687_g__1686786653.csv",
        absolute=True
    )
    proxy_archive = NoveltyArchive()

    world_config = SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE
    world_config.population_size = 24
    world_config.seed = 1

    GENERATIONS = 100
    START_AT_GEN = 0
    for i in range(len(archive.archive)):
        behavior, genome = archive.archive[i], archive.genotypes[i]
        proxy_archive.addToArchive(behavior, genome)
        if i > 0 and i % GENERATIONS == 0:
            print(f"GENERATION: {i // GENERATIONS}")
            novelty_scores = []
            for j in range(GENERATIONS):
                novelty_scores.append((proxy_archive.getNovelty(15, proxy_archive.archive[-j]), len(proxy_archive.archive) - j - 1))
            novelty_scores.sort()
            print("Most Novel Scores Calculated!")
            print(novelty_scores[-3:])
            print("Ask the human about the following genomes...")

            queries = [archive.genotypes[score[1]] for score in novelty_scores[-3:]]
            if i // GENERATIONS < START_AT_GEN:
                continue
            for query in queries:
                world_config.population_size = 24
                world_config.agentConfig.from_n_species_controller(query)
                world_config.agentConfig.attach_world_config(world_config)
                sim(SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE, step_size=8)