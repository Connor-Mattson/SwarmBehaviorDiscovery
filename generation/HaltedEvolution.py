

# from NovelSwarmBehavior.src.config.WorldConfig import RectangularWorldConfig
# from NovelSwarmBehavior.src.config.OutputTensorConfig import OutputTensorConfig
# from NovelSwarmBehavior.src.config.EvolutionaryConfig import GeneticEvolutionConfig
# from NovelSwarmBehavior.src.novelty.BehaviorDiscovery import BehaviorDiscovery

"""
This file runs evolution but halts and waits until hearing back from the end of the pipeline
regarding the quality of the genome that was outputted by this code
"""


class HaltedEvolution:
    def __init__(self,
                 world: RectangularWorldConfig,
                 output_config: OutputTensorConfig,
                 evolution_config: GeneticEvolutionConfig
                 ):
        self.world = world,
        self.output_configuration = output_config
        self.evolve_config = evolution_config

    def setup(self):
        print("Hello World!")
