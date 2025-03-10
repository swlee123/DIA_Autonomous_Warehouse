import gymnasium as gym
import time  # For adding delay between renders
from typing import Set, Tuple


def visualize_obstacles(env, obstacles: Set[Tuple[int, int]]):
    """
    Create a visual representation of the warehouse with obstacles
    """
    
    env = env.unwrapped
    # Get environment dimensions
    height, width = env.grid_size
    
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Mark obstacles
    for x, y in obstacles:
        grid[y][x] = 'X'  # X represents a shelf/obstacle
    
    # Mark goals
    for goal_x, goal_y in env.goals:
        grid[goal_y][goal_x] = 'G'  # G represents a goal
    
    # Mark agents
    for agent in env.agents:
        grid[agent.y][agent.x] = 'A'  # A represents an agent
    
    # Print the grid
    print("\nWarehouse Layout:")
    print("X: Shelf/Obstacle")
    print("G: Goal")
    print("A: Agent")
    print("-" * (width * 2 + 1))
    
    for row in grid:
        print("|" + "|".join(row) + "|")
    print("-" * (width * 2 + 1))

def run_warehouse_demo():
    # Create the environment
    # Available variants:
    # "rware-tiny-2ag-v2"    - 2 agents, small warehouse
    # "rware-small-4ag-v2"   - 4 agents, medium warehouse
    # "rware-medium-6ag-v2"  - 6 agents, large warehouse
    # render_mode : "human"(display a window) or "rgb_array"(return a numpy array) 
    env = gym.make("rware:rware-tiny-2ag-v2", render_mode="human")
    

    
    # Reset environment and get initial observation and info
    obs, info = env.reset()
    # obs contains the initial state observation for each agent
    # For 2 agents, obs will be a tuple of 2 observation arrays
    # info contains additional information about the environment state
    
    # Visualize obstacles
    # obstacles = get_obstacles(env)     
    # print("\nObstacle Positions:")
    # for x, y in sorted(obstacles):
    #     print(f"Shelf at position: ({x}, {y})")
    
    # visualize_obstacles(env, obstacles)
    # input("\nPress Enter to continue...")
    
    # Run for 100 steps
    for step in range(100):
        # Sample random actions for each agent
        actions = env.action_space.sample()
        
        # Action space : 
        # NOOP = 0
        # FORWARD = 1
        # LEFT = 2
        # RIGHT = 3
        # TOGGLE_LOAD = 4
        # .sample() will return a random action from the action space

        # Execute actions and get results
        obs, rewards, done, truncated, info = env.step(actions)
        
        # Render the current state
        env.render()
        
        # Add a small delay to make visualization viewable
        time.sleep(0.1)
        
        # Print step information
        print(f"Step {step + 1}")
        print(f"Actions taken: {actions}")
        print(f"Rewards: {rewards}")
        print("-" * 50)
        
        # If episode is done, reset the environment
        if done:
            print("Episode finished!")
            obs, info = env.reset()
    
    # Close the environment
    env.close()


def get_obstacles(env) -> Set[Tuple[int, int]]:
    """
    Get positions of shelves and other obstacles
    """
    
    env = env.unwrapped
    
    obstacles = set()
    
    # Get shelf positions from the environment grid
    
    shelf_array = env.shelfs

    for shelf in shelf_array:
        obstacles.add((shelf.x, shelf.y))
    
    # print("Obstacles:",obstacles)
    # input("Press Enter to continue...")
    return obstacles

if __name__ == "__main__":
    try:
        run_warehouse_demo()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")