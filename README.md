



# Project Setup Guide

## 1. Setup Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
python -m venv venv
venv\Scripts\activate
```

Install seac_requirements.txt to run experiments using A* and Dijkstra algorithm.

```bash
pip install -r seac_requirements.txt
```

Install ad_requirements.txt to run experiments using A* and Dijkstra algorithm.

```bash
pip install -r ad_requirements.txt
```

Please make sure there is 2 separated virtual environment for (A* and Dijkstra) and SEAC as they require different dependencies.


## 2. Run experiments 
If you want to try to recreate the experiment in our report : 

### A* and Dijkstra

```bash
cd astar_dijkstra
python auto_experiment.py
```

Training result will be saved as CSV file in astar_dijkstra\result and .mp4 video in astar_dijkstra\video


### SEAC 

```bash
cd seac
python evalute.py
```
Training result will be saved as CSV file in seac\result\exp_result_{ENV}.csv

## 3. Run Preset Simulations

In addition to the predefined environments mentioned earlier, we also provide several preset simulation settings that can be executed with different configurations.

The original configurable parameters are:

- **Agent Count**: Number of autonomous agents in the environment.
- **Warehouse Layout**: Variation in the structural design and arrangement of shelves and pathways.

New parameters introduced by us : 
- **Obstacle Location**: Fixed obstacles placed within the warehouse grid.
- **Human Count**: Number of dynamic human entities present.


### 3.1 A* and Dijkstra

### 3.2 SEAC


