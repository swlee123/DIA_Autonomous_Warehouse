import torch
import numpy as np
import rware
import gymnasium as gym
import pandas as pd
from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit
import os
import time
import argparse

def run_experiment(agent_n=2):
    # Update env_name based on agent_n
    env_name = f"rware:rware-small-{agent_n}ag-v2"
    # Update path based on agent_n
    path = f"D:\\robotic-warehouse\\seac\\pretrained\\rware-small-4ag"
    
    time_limit = 500  # 25 for LBF
    EPISODES = 1
    TICKS = 300
    INTERVAL = 0.05

    # Create environment with the specified number of agents
    env = gym.make(env_name)
    agents = [
        A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]
    
    for agent in agents:
        print("agent id", agent.agent_id)
        agent.restore(path + f"/agent{agent.agent_id}")

    total_reward = 0
    
    for ep in range(EPISODES):
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=time_limit)
        env = RecordEpisodeStatistics(env)
        
        obs = env.reset()
        done = [False for _ in range(len(agents))]
        
        ticks = TICKS
        while ticks > 0:
            print("whole obs", obs)
            
            # convert obs tuple to numpy array
            for i, o in enumerate(obs):
                print(f"obs[{i}] shape: {len(o)}")
                print("obs:", o)
                
            if len(obs[0]) == 4 or len(obs[0]) == 2:
                obs = [torch.from_numpy(o) for o in obs[0]]
            else:
                obs = [torch.from_numpy(o) for o in obs]
                
            _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
            actions = [a.item() for a in actions]
            env.render()
            obs, _, done, info = env.step(actions)
            time.sleep(INTERVAL)
            ticks -= 1
                
        obs = env.reset()
        print("--- Episode Finished ---")
        print(info)
        
        reward = sum(info["episode_reward"])
        total_reward += reward
    
    reward_per_ticks = total_reward / TICKS

    # dump into csv file
    result = {
        "algorithm": "SEAC",
        "env_name": "Normal",
        "num_agents": agent_n,
        "reward": total_reward,
        "reward_per_ticks": reward_per_ticks,
        "ticks": TICKS,
    }

    print("Result: ", result)


    print(" --- ")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run warehouse experiment with specified number of agents')
    parser.add_argument('--agent_n', type=int, default=2, help='Number of agents (default: 2)')
    args = parser.parse_args()
    
    run_experiment(agent_n=args.agent_n)