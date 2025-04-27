import argparse
import time
import os
import yaml
import threading
import pandas as pd
from astar_dijkstra import run_warehouse_with

# Load configuration (hardcoded values for this preset)
def load_config():
    config = {
        "TICKS": 500,  # Hardcoded ticks
        "INTERVAL": 0.05,  # Hardcoded interval
        "ALGORITHMS": ["A*", "Dijkstra"],  # Hardcoded algorithms list
        "ENVIRONMENTS": {
                "shelf_column": 5,
                "column_height": 3,
                "shelf_rows": 5,
                "agent_count": 2,
                "human_count": 5,
                "obstacles_loc": [(5,16),(10,16),(7,17),(8,18)],

        }
    }
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiment with A* or Dijkstra algorithm.")
    parser.add_argument(
        '--algorithm',
        choices=['A*', 'Dijkstra'],
        required=True,
        help="Choose the algorithm to run: A* or Dijkstra"
    )
    args = parser.parse_args()

    # Load config (hardcoded for this preset)
    config = load_config()

    ALGORITHM = args.algorithm  # Selected algorithm from CLI
    TICKS = config["TICKS"]
    INTERVAL = config["INTERVAL"]
    ENVIRONMENTS = config["ENVIRONMENTS"]

    info = []

    # Ensure selected algorithm is valid
    if ALGORITHM not in config["ALGORITHMS"]:
        print(f"Invalid algorithm: {ALGORITHM}")
        return

    

    print(f"\n--- Testing {ALGORITHM} in Environment: Preset3 ---")

    start_time = time.time()

    # Run the experiment
    reward, rpt, collision, cpt = run_warehouse_with(
            ALGORITHM,
            TICKS,
            INTERVAL,

            shelf_column = ENVIRONMENTS["shelf_column"],
            column_height = ENVIRONMENTS["column_height"],
            shelf_rows = ENVIRONMENTS["shelf_rows"],
            agent_count = ENVIRONMENTS["agent_count"],
            human_count = ENVIRONMENTS["human_count"],
            obstacles_loc = ENVIRONMENTS["obstacles_loc"],

    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution Time for {ALGORITHM} in Preset3 env: {execution_time:.4f} seconds")
    print(f"Reward: {reward}")
    print(f"Reward Per Tick: {rpt}")

    env_results = {
            "algorithm": ALGORITHM,
            "env_name": "Preset2-2a5h4o_big", 
            "execution_time": execution_time,
            "reward": reward,
            "reward_per_tick": rpt,
            "collisions": collision,
            "collisions_per_tick": cpt,
            "ticks": TICKS
    }

    info.append(env_results)

    # Save all results
    if not os.path.exists("result"):
        os.makedirs("result")

    print("Results:", info)
    

if __name__ == "__main__":
    main()
