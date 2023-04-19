# Leveraging Human Feedback to Evolve and Discover Novel Emergent Behaviors in Robot Swarms
Contributors: Connor Mattson, Daniel S. Brown

## Required Software
- Python & Pip
- External Python Packages as defined in [requirements.txt](requirements.txt) 

## Setup
Install Python Packages
    
    pip install -r requirements.txt

Test Simulation

    python demo.py

# Robot Swarm Simulation

We use a custom swarm simulator to parameterize agents and environments. See this repository for details on how to create your own agent types and simulate swarm behaviors: https://github.com/Connor-Mattson/RobotSwarmSimulator

# Novel Behavior Discovery 

## Generate Dataset
Requires directory 'data' at project root.
See `generate_trajectories.ipynb` for cells that allow you to define a robot model, genomes, and output directory to output data.

OR see latest releases for an example dataset to examine: "gecco-filtered-two-sensor.zip", this file should be unzipped in the top-level "data" folder.

## Self-Supervised Learning (Contrastive Learning)
Requires a Contrastive Learning Dataset, generated from the previous file
see `pretraining.ipynb`

OR see latest releases for model parameters of self-supervised training alone: "Two-Sensor-Self-Supervised-Models.zip"

## Human-in-the-loop (HIL) Learning
Should be performed after Pretraining to replicate our results
see `hil-training.ipynb`

OR see latest releases for model parameters of HIL training on top of self-supervised learning: "Two-Sensor-HIL-Models.zip"

## Novelty Search and Evolution
Requires a saved PyTorch Checkpoint containing training weights
see `novelty_evolution.ipynb`

## Visualize Embedding Spaces
Requires a saved PyTorch Checkpoint containing training weights
see `visualize-embeddings.ipynb`

## Obtain L2 Accuracy
Requires a saved PyTorch Checkpoint containing training weights
see `accuracy-results.ipynb`


[//]: # (## Augmentation)

[//]: # (We have explored the idea of augmenting this framework further to allow more complex world, sensor, controller, and actuator spaces. )

[//]: # (Much of the backbone to support these augmentations is present in this codebase, but lacks testing and robustness.)

[//]: # ()
[//]: # (We invite you to augment cautiously and carefully test output validity.)
