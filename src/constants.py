"""
Constant Declarations Across Robot Models
"""
from novel_swarms.novelty.GeneRule import GeneBuilder, GeneRule
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.GenomeDependentSensor import GenomeBinarySensor
import numpy as np

ROBOT_TYPES = frozenset([
    "single-sensor",
    "two-sensor"
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


