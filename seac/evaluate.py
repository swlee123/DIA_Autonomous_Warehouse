import torch
import numpy as np
import rware
# import lbforaging
import gymnasium as gym 
import pandas as pd
from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit
import os
import time 

path = "D:\\robotic-warehouse\\seac\pretrained\\rware-small-4ag"
env_name = "rware:rware-small-2ag-v2"
time_limit = 500 # 25 for LBF
EPISODES = 1
TICKS = 300
INTERVAL = 0.05

# define all shit at here
env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    print("agent id",agent.agent_id)
    agent.restore(path + f"/agent{agent.agent_id}")

for ep in range(EPISODES):
    env = gym.make(env_name)
    # env = Monitor(env, f"seac_rware-small-4ag_eval/video_ep{ep+1}")
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False for _ in range(len(agents))]

    ticks = TICKS
    while ticks > 0 :
        
        # print("while not done")

        print("whole obs",obs)
        # convert obs tuple to numpy array 
        for i, o in enumerate(obs):
            print(f"obs[{i}] shape: {len(o)}")
            print("obs:",o)
        
        if len(obs[0]) == 4 or len(obs[0]) == 2:
            obs = [torch.from_numpy(o) for o in obs[0]]
        
        else:
            obs = [torch.from_numpy(o) for o in obs]
    
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        env.render()
        obs, _, done, info = env.step(actions)
        time.sleep(INTERVAL)
        ticks -= 1
        
        
    obs = env.reset()
    print("--- Episode Finished ---")
    # print(f"Episode rewards: {sum(info['episode_reward'])}")
    print(info)
    
reward = sum(info["episode_reward"])
reward_per_ticks = reward / TICKS

# dump into csv file 
result = {
    "algorithm": "SEAC",
    "env_name": "Normal",
    "num_agents": 2,
    "reward": reward,
    "reward_per_ticks": reward_per_ticks,
    "ticks": TICKS,
}

print("Result: ", result)
df = pd.DataFrame([result])

# check if the directory exists, if not create it
if not os.path.exists("result"):
    os.makedirs("result")
    
file_name = "result/exp_results_Normal.csv"
df.to_csv(file_name, index=False)

print(f"Results saved to CSV file {file_name}")
print(" --- ")
