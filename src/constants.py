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
import numpy as np

ROBOT_TYPES = frozenset([
    "single_sensor",
    "two_sensor"
])

STRATEGY_TYPES = frozenset([
    "Mattson_and_Brown",
    "Brown_et_al"
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
