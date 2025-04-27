import time
import pandas as pd
import threading
import yaml
import os

from astar_dijkstra import run_warehouse_with
from utils import record_window_to_video

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def record_experiment(window_title, video_name, duration, frame_rate=30):
    print("Recording the experiment window...")
    ins = record_window_to_video(window_title=window_title, video_name=video_name, duration=duration, frame_rate=frame_rate)
    return ins

def main():
    config = load_config()

    TICKS = config["TICKS"]
    INTERVAL = config["INTERVAL"]
    ALGORITHMS = config["ALGORITHMS"]
    ENVIRONMENTS = config["ENVIRONMENTS"]

    info = []

    for algo in ALGORITHMS:
        for env_name, env_params in ENVIRONMENTS.items():

            video_name = f"warehouse_{algo}_{env_name}.mp4" if algo != "A*" else f"warehouse_Astar_{env_name}.mp4"

            print(f"\n--- Testing {algo} in Environment: {env_name} ---")

            start_time = time.time()

            # Start recording
            recording_thread = threading.Thread(
                target=record_experiment, 
                args=("auto_experiment.py", video_name, TICKS * INTERVAL + 30)
            )
            recording_thread.start()

            # Run the experiment
            reward, rpt, collision, cpt = run_warehouse_with(
                algo,
                TICKS,
                INTERVAL,
                **env_params
            )

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Execution Time for {algo} in {env_name} env: {execution_time:.4f} seconds")
            print(f"Reward: {reward}")
            print(f"Reward Per Tick: {rpt}")

            env_results = {
                "algorithm": algo,
                "env_name": env_name,
                "execution_time": execution_time,
                "reward": reward,
                "reward_per_tick": rpt,
                "collisions": collision,
                "collisions_per_tick": cpt,
                "ticks": TICKS
            }

            info.append(env_results)

            # Wait for recording to finish
            recording_thread.join()
            print("Recording finished.")

    # Save all results
    if not os.path.exists("result"):
        os.makedirs("result")

    print("Results:", info)
    
    df = pd.DataFrame(info)
    name  = "result/exp_results_"+ config["ALGORITHMS"][0] + ".csv"
    df.to_csv(name, index=False)
    print(f"All results saved to {name}")

if __name__ == "__main__":
    main()
