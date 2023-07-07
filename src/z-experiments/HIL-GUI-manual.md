# Manual for HIL-Assisted Robotic Swarm Evolution
**To skip these instructions and go to the main GUI, quit this window.**
## Purpose Statement
Welcome to the GUI for Human-in-the-Loop assisted swarm evolution. Prior research on swarm behavior has defined an embedding-based metric for swarm novelty as the fitness function for evolutionary search. The goal of this GUI is to use a human as the fitness function instead. This methodology was inspired by **[Swarm Chemistry, Hiroki Sayama](https://direct.mit.edu/artl/article-abstract/15/1/105/2623/Swarm-Chemistry?redirectedFrom=fulltext)**.

## Instructions
Upon quitting this window, you will be taken to a configuration screen where you can customize some of the parameters that dictate the behavior of the swarm. These parameters include:

* **Heterogeneity vs. Homogeneity** &ndash; Whether you want to evolve homogenous or heterogeneous swarms.
* **Capability Model** &ndash; The type of agents in the swarm.
* **Sensor Type** &ndash; The type of sensor that the agents use.
* **Number of Agents** &ndash; How many agents in each simulation.

After configuring these parameters, you will be taken to a simulation screen. This screen consists of 8 simulation tiles and a sidebar. At the top of each simulation tile, you'll see a list of floats labeled "Params". This list represents the controller of the swarm being simulated in that tile. Note that each tile has a unique controller. On the first generation of evolution, these controllers will be randomly generated.

In the bottom-right corner of each simulation tile, you'll see two options.
* **Save** &ndash; If you select the Save option, the controller corresponding to that simulation tile will be printed to the console once the generation is terminated. You should select the Save option if you find the controller interesting and/or unique. If you change your mind and decide that you don't want to print the controller to the console, unselect the Save option.
* **Evolve** &ndash; Out of the 8 Evolve options on the GUI, you may only select 1 or 2 at a time. If you try to select more than 2 at a time, all 8 of the Evolve options will be unselected. The use for the Evolve option will be explained later in this manual.

The sidebar displays the current generation (starting from 0) of swarms and number of steps that the simulation tiles have taken since the beginning of the generation. It also
* **Advance** &ndash; Press this button once you have selected the Evolve options on 1 or 2 of the simulation tiles. If you don't have Evolve selected on any of the tiles, then this button will do nothing. If you have Evolve selected on one tile, a new generation of controllers is obtained by randomly mutating the controller corresponding to that tile. If you have Evolve selected on 2 tiles, a new generation of controllers is obtained by randomly mutating and combining the 2 controllers corresponding to those 2 tiles.
* **Skip** &ndash; Press this button if you don't see any controllers that you wish to see evolved. A new generation of randomly generated controllers will be obtained and simulated.
* **Back** &ndash; Press this button if you have reached a dead end in evolution and wish to return to the previous generation of controllers.

**To skip these instructions and go to the main GUI, quit this window.**

Good luck and happy evolving!