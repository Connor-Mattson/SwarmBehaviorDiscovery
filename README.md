# Exploring Behavior Discovery Methods for Heterogeneous Swarms of Limited-Capability Robots
Contributors: Connor Mattson, Jeremy C. Clark, Daniel S. Brown

[//]: # (# Leveraging Human Feedback to Evolve and Discover Novel Emergent Behaviors in Robot Swarms)

[//]: # (Contributors: Connor Mattson, Daniel S. Brown)

## Required Software
- Python & Pip
- External Python Packages as defined in [requirements.txt](requirements.txt) 

## Setup
Install Python Packages
    
    pip install -r requirements.txt

[//]: # (# Robot Swarm Simulation)

[//]: # ()
[//]: # (We use a custom swarm simulator to parameterize agents and environments. See this repository for details on how to create your own agent types and simulate swarm behaviors: https://github.com/Connor-Mattson/RobotSwarmSimulator)

[//]: # ()
[//]: # (## Test Simulation)

[//]: # ()
[//]: # (To test the simulation package, 'novel_swarms', run the following command where agents perform a cyclic pursuit collective behavior)

[//]: # (`python demo.py`)

[//]: # ()
[//]: # (# Novel Behavior Discovery)

[//]: # (## Generate Dataset)

[//]: # (Requires directory 'data' at project root.)

[//]: # ()
[//]: # (`python generate_dataset.py --help`)

[//]: # ()
[//]: # (OR see [latest releases]&#40;https://github.com/Connor-Mattson/SwarmBehaviorDiscovery/releases/tag/v0.1.0-alpha&#41; for an example dataset to examine: "gecco-filtered-two-sensor.zip", this file should be unzipped in the top-level "data" folder.)

[//]: # ()
[//]: # (## Self-Supervised Learning &#40;Contrastive Learning&#41;)

[//]: # (Requires a Contrastive Learning Dataset, generated from the previous file)

[//]: # ()
[//]: # (`python self_supervised_training.py --help`)

[//]: # ()
[//]: # (OR see [latest releases]&#40;https://github.com/Connor-Mattson/SwarmBehaviorDiscovery/releases/tag/v0.1.0-alpha&#41; for model parameters of self-supervised training alone: "Two-Sensor-Self-Supervised-Models.zip")

[//]: # ()
[//]: # (## Human-in-the-loop &#40;HIL&#41; Learning)

[//]: # (Should be performed after pretraining and requires a set of human labeled behaviors)

[//]: # ()
[//]: # (`python human_query_training.py --help`)

[//]: # ()
[//]: # (OR see [latest releases]&#40;https://github.com/Connor-Mattson/SwarmBehaviorDiscovery/releases/tag/v0.1.0-alpha&#41; for model parameters of HIL training on top of self-supervised learning: "Two-Sensor-HIL-Models.zip")

## Novelty Search and Evolution
Requires a saved PyTorch Checkpoint containing training weights (See Releases Folder)

`python evolve.py --help`


### Hand-Crafted Metrics (Type-Agnostic)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy Brown_et_al --seed 0`

### Hand-Crafted Metrics (Type-Aware)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy Brown_et_al --species_aware --seed 0`

### Latent Learning (Type-Agnostic)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy Mattson_and_Brown --checkpoint [MODEL_CHECKPOINT_HERE] --seed 0`

### Latent Learning (Type-Aware)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy Mattson_and_Brown  --checkpoint [MODEL_CHECKPOINT_HERE] --species_aware --seed 0`

### Resnet18 (Type-Aware)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy ResNet --seed 0`

### Resnet18 (Type-Agnostic)
`python evolve.py --gen 50 --pop 100 --heterogeneous --type single-sensor --strategy ResNet --species_aware --seed 0`





[//]: # (## Visualize Embedding Spaces)

[//]: # (Requires a saved PyTorch Checkpoint containing training weights)

[//]: # (see `visualize-embeddings.ipynb`)

[//]: # ()
[//]: # (## Obtain L2 Accuracy)

[//]: # (Requires a saved PyTorch Checkpoint containing training weights)

[//]: # (see `accuracy-results.ipynb`)


[//]: # (## Augmentation)

[//]: # (We have explored the idea of augmenting this framework further to allow more complex world, sensor, controller, and actuator spaces. )

[//]: # (Much of the backbone to support these augmentations is present in this codebase, but lacks testing and robustness.)

[//]: # ()
[//]: # (We invite you to augment cautiously and carefully test output validity.)
