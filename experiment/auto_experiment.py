import time
from experiment.run_aStar import run_warehouse_with
import pandas as pd
from utils import record_window_to_video
import threading

# Define the recording function to be run in a separate thread
def record_experiment(window_title, video_name, duration, frame_rate=30):
    print("Recording the experiment window...")
    ins = record_window_to_video(window_title=window_title, video_name=video_name, duration=duration, frame_rate=frame_rate)
    return ins


# evaluate 🔛 
# env_name = "rware:rware-small-4ag-v2" for 3 env ,
# diff is for 2 : with obstacle, 3: with human 4 : both

# Define the three environments settings
ENV = {
    # "Obstacle & Human": [3,8,2,2,1,[(4,10),(5,10),(4,12),(5,12)]],
    # "Normal": [3,8,2,2],
    "Obstacle": [3,8,2,2,[(4,10),(5,10),(4,12),(5,12)]],
    # "Human": [3,8,2,2,1],
}

ALGORITHMS = ["A*", "Dijkstra"]
TICKS = 300
INTERVAL = 0.05

info = []

# Loop through each environment and compare A* and Dijkstra
for algo in ALGORITHMS:
    for env_name in ENV.keys():
        

            
        video_name = f"{env_name}_{algo}.mp4"
        
        print(f"\n--- Testing {algo} in Environment: {env_name} ---")
        
        start_time = time.time()
        
        # You can start recording once the environment has started and the window is visible
        if algo == "A*":
            video_name = f"warehouse_Astar_{env_name}.mp4"
        else:
            video_name = f"warehouse_{algo}_{env_name}.mp4"
        recorded = False
        
        # Start recording in a separate thread only if not already started
        if not recorded:
            recording_thread = threading.Thread(target=record_experiment, args=("auto_experiment.py", video_name, TICKS*INTERVAL+30))
            recording_thread.start()
            recorded = True  # Set flag so recording only starts once
        
        # Run the environment and experiment based on env_name
        if env_name == "Obstacle & Human":
            shelf_column, column_height, shelf_rows, agent_count, human_count, obs_loc = ENV["Obstacle & Human"]
            reward, rpt = run_warehouse_with(
                algo, 
                TICKS,
                INTERVAL,
                shelf_column=shelf_column,
                column_height=column_height,
                shelf_rows=shelf_rows,
                agent_count=agent_count,
                human_count=human_count,
                obstacles_loc=obs_loc
            )
        elif env_name == "Normal":
            shelf_column, column_height, shelf_rows, agent_count = ENV["Normal"]
            reward, rpt = run_warehouse_with(
                algo, 
                TICKS,
                INTERVAL,
                shelf_column=shelf_column,
                column_height=column_height,
                shelf_rows=shelf_rows,
                agent_count=agent_count
            )
        elif env_name == "Obstacle": 
            shelf_column, column_height, shelf_rows, agent_count, obs_loc = ENV["Obstacle"]
            reward, rpt = run_warehouse_with(
                algo, 
                TICKS,
                INTERVAL,
                shelf_column=shelf_column,
                column_height=column_height,
                shelf_rows=shelf_rows,
                agent_count=agent_count,
                obstacles_loc=obs_loc
            )
        elif env_name == "Human":
            shelf_column, column_height, shelf_rows, agent_count, human_count = ENV["Human"]
            reward, rpt = run_warehouse_with(
                algo, 
                TICKS,
                INTERVAL,
                shelf_column=shelf_column,
                column_height=column_height,
                shelf_rows=shelf_rows,
                agent_count=agent_count,
                human_count=human_count
            )
            
        else:
            print(f"Unknown environment: {env_name}")
            continue
        
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
            "ticks": TICKS
        }
        
        info.append(env_results)
        
        # Wait for the recording thread to finish before moving to the next loop
        recording_thread.join()
        print("Recording finished.")

# Save the results to a file
env_name = "Obstacle"

df = pd.DataFrame(info)
df.to_csv(f"result/exp_results_{env_name}.csv", index=False)
