"""
Constant Declarations Across Robot Models
"""
from novel_swarms.novelty.GeneRule import GeneBuilder, GeneRule
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.GenomeDependentSensor import GenomeBinarySensor
from novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.config.defaults import ConfigurationDefaults
from novel_swarms.behavior import *
import numpy as np

ROBOT_TYPES = frozenset([
    "single-sensor",
    "two-sensor"
])

STRATEGY_TYPES = frozenset([
    "Mattson_and_Brown",
    "Brown_et_al",
    "ResNet"
])

MINING_TYPES = frozenset([
    "Random",
    "Semi-Hard"
])

NETWORK_TYPE = frozenset([
    "Scratch",
    "ResNet"
])

# Note: Not Immutable
SINGLE_SENSOR_SET = SensorSet([
    BinaryLOSSensor(angle=0)
])

# Note: Not Immutable
TWO_SENSOR_SET = SensorSet([
    BinaryLOSSensor(angle=0),
    GenomeBinarySensor(genome_id=8)
])

# Note: Not Immutable
SINGLE_SENSOR_GENE_MODEL = GeneBuilder(
    round_to_digits=1,
    rules=[
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
    ]
)

# Note: Not Immutable
TWO_SENSOR_GENE_MODEL = GeneBuilder(
    round_to_digits=1,
    rules=[
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[(i / 8) * np.pi for i in range(-7, 8, 1)], step_size=2, allow_mutation=True),
    ]
)

# Note: Not Immutable
SINGLE_SENSOR_HETEROGENEOUS_MODEL = GeneBuilder(
    round_to_digits=None,
    rules=[
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[i / 10 for i in range(-10, 11, 1)], step_size=4, allow_mutation=True),
        GeneRule(discrete_domain=[(12 / 24), (8 / 24), (6 / 24), (3 / 24), (1 / 24)], step_size=2, allow_mutation=True)
    ]
)

DEFAULT_OUTPUT_CONFIG = OutputTensorConfig(
    timeless=True,
    total_frames=80,
    steps_between_frames=2,
    screen=None
)

TWO_SENSOR_AGENT_CONFIG = DiffDriveAgentConfig(
    sensors=SensorSet([
        BinaryLOSSensor(angle=0),
        GenomeBinarySensor(genome_id=8)
    ]),
)

SINGLE_SENSOR_AGENT_CONFIG = ConfigurationDefaults.DIFF_DRIVE_AGENT

TWO_SENSOR_WORLD_CONFIG = RectangularWorldConfig(
    size=(500, 500),
    behavior=ConfigurationDefaults.BEHAVIOR_VECTOR,
    agentConfig=TWO_SENSOR_AGENT_CONFIG,
    padding=15,

)

SINGLE_SENSOR_WORLD_CONFIG = RectangularWorldConfig(
    size=(500, 500),
    behavior=ConfigurationDefaults.BEHAVIOR_VECTOR,
    agentConfig=SINGLE_SENSOR_AGENT_CONFIG,
    padding=15
)

HETEROGENEOUS_SUBGROUP_BEHAVIOR = [
    SubGroupBehavior(AverageSpeedBehavior(), subgroup=0),
    SubGroupBehavior(AngularMomentumBehavior(), subgroup=0),
    SubGroupBehavior(RadialVarianceBehavior(), subgroup=0),
    SubGroupBehavior(ScatterBehavior(), subgroup=0),
    SubGroupBehavior(GroupRotationBehavior(), subgroup=0),
    SubGroupBehavior(AverageSpeedBehavior(), subgroup=1),
    SubGroupBehavior(AngularMomentumBehavior(), subgroup=1),
    SubGroupBehavior(RadialVarianceBehavior(), subgroup=1),
    SubGroupBehavior(ScatterBehavior(), subgroup=1),
    SubGroupBehavior(GroupRotationBehavior(), subgroup=1),
    AverageSpeedBehavior(),
    AngularMomentumBehavior(),
    RadialVarianceBehavior(),
    ScatterBehavior(),
    GroupRotationBehavior(),
]


def SINGLE_SENSOR_HETERO_AGENT_CONFIG(colored=False):
    from novel_swarms.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
    agent_A = DiffDriveAgentConfig(controller=None, sensors=SINGLE_SENSOR_SET, dt=1.0,
                                   body_color=(255, 0, 0) if colored else (255, 255, 255))
    agent_B = DiffDriveAgentConfig(controller=None, sensors=SINGLE_SENSOR_SET, dt=1.0,
                                   body_color=(0, 255, 0) if colored else (255, 255, 255))
    h_config = HeterogeneousSwarmConfig()
    h_config.add_sub_populuation(agent_A, 12)  # 12 will be intentionally overwritten by controller
    h_config.add_sub_populuation(agent_B, 12)  # 12 will be intentionally overwritten by controller
    return h_config


SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AGNOSTIC = RectangularWorldConfig(
    size=(500, 500),
    behavior=ConfigurationDefaults.BEHAVIOR_VECTOR,
    agentConfig=SINGLE_SENSOR_HETERO_AGENT_CONFIG(colored=False),
    padding=15
)

SINGLE_SENSOR_HETEROGENEOUS_WORLD_CONFIG_AWARE = RectangularWorldConfig(
    size=(500, 500),
    behavior=ConfigurationDefaults.BEHAVIOR_VECTOR,
    agentConfig=SINGLE_SENSOR_HETERO_AGENT_CONFIG(colored=True),
    padding=15
)
