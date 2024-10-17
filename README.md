# Discovery and Deployment of Emergent Robot Swarm Behaviors via Representation Learning and Real2Sim2Real Transfer
Contributors: Anonymous

## Required Software
- Python & Pip
- External Python Packages as defined in [requirements.txt](requirements.txt) 

## Setup
Install Python Packages
    
    pip install -r requirements.txt
    pip install -i https://test.pypi.org/simple/ novel-swarms --upgrade

## Train Self-Supervised Repr. Learning
1. open "simclr.py", modify PARAMS with path to training data
2. run `python simclr.py`.

## Real2Sim2Real Novelty Search with SSL (RSRS)
1. open "SimCLR-NS.py", modify "SIMCLR_CHECKPOINT" to point to the saved model from the previous step
2. ensure that "WITH_RSRS" is set to True.
3. run `python SimCLR-NS.py`

## Novelty Search (No RSRS enhancement)
1. open "SimCLR-NS.py", modify "SIMCLR_CHECKPOINT" to point to the saved model from the previous step
2. ensure that "WITH_RSRS" is set to True.
3. run `python SimCLR-NS.py`

## Results Plots and Analysis
see ipython notebooks "compute_class_confusion.ipynb" and "results_plots.py"