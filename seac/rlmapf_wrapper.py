import sys

sys.path.append("/home/kami/Documents/RLMAPF")

from rlmapf import RLMAPF
import gymnasium as gym
import numpy as np

class GymRLMAPF(gym.Env):
    def __init__(self, env_config):
        self.env = RLMAPF(env_config)
        self.n_agents = len(self.env.get_agent_ids())
        self.action_space = gym.spaces.Tuple([self.env.action_space for _ in range(self.n_agents)])
        self.observation_space = self.get_observation_space()
        self._seed = self.env.get_seed()
        
    def get_observation_space(self):
        obs_space = self.env.observation_space
        flattened_obs_space = gym.spaces.Box(-1, 1, (4 + obs_space['obstacles'].shape[0],), dtype=np.float64)
        return gym.spaces.Tuple(tuple([flattened_obs_space] * self.n_agents))
        # return flattened_obs_space

    def seed(self, seed=None):
        self.env.set_seed(seed)
        return self._seed
    
    def flatten_observation(self, obs):
        new_obs = []
        for agent_obs in obs.values():
            agent_obs['position'] = np.array(agent_obs['position']) / self.env.grid_size
            agent_obs['goal_position'] = np.array(agent_obs['goal_position']) / self.env.grid_size
            agent_obs = np.concatenate([np.array(agent_obs['position']), np.array(agent_obs['goal_position']), np.array(agent_obs['obstacles'])])
            new_obs.append(agent_obs)
        
        # Ensure all observations have the same shape
        max_length = max(len(o) for o in new_obs)
        new_obs = [np.pad(o, (0, max_length - len(o)), mode='constant') for o in new_obs]
        print(new_obs)
        return tuple(new_obs)

    # Example usage in the reset method
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        obs, info = self.env.reset()
        # flatten the observation space dictionary
        obs = self.flatten_observation(obs)
        return obs
    
    def step(self, action):
        action = {agent: action[i] for i, agent in enumerate(self.env.get_agent_ids())}
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.flatten_observation(obs)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.env.render()

    def get_agent_ids(self):
        return self.env.get_agent_ids()

# Register the environment in gymnasium
gym.register(
    id="RLMAPF",
    entry_point="rlmapf_wrapper:GymRLMAPF",
    kwargs={"env_config":{
        "agents_num": 2,
        "render_mode": "human",
        "render_delay": 1,
        "max_steps": 1000,
        "observation_type": "position",
        "map_path": "/home/kami/Documents/RLMAPF/maps/",
        "maps_names_with_variants": {
            "empty_1-4a-5x4": None,
        }
    }},
)

if __name__ == "__main__":
    env_config = {
        "agents_num": 2,
        "render_mode": "human",
        "render_delay": 1,
        "max_steps": 1000,
        "observation_type": "position",
        "map_path": "/home/kami/Documents/RLMAPF/maps/",
        "maps_names_with_variants": {
            "empty_1-4a-5x4": None,
        }
    }

    env = GymRLMAPF(env_config=env_config)

    print(env.observation_space)
    print(env.action_space)

    obs, info = env.reset()

    print(obs)

    # Check if obs is within the observation space
    print("Observation Space Contains Obs:")
    print(env.observation_space.contains(obs))

    for ep in range(5):
        _ = env.reset()
        actions = {agent: env.action_space.sample() for agent in env.get_agent_ids()}

        _ = env.step(actions)
        env.render()
        print("--- Episode Finished ---")