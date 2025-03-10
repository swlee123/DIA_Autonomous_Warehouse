import gymnasium as gym
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import time
from rware.warehouse import Warehouse, RewardType, Action, Direction


# now the path found by the aStar function is correct ,
# but the action sequence is not correct, need to fix , get action from position 
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent=None):
        self.position = position
        self.g_cost = g_cost  # Cost from start to current node
        self.h_cost = h_cost  # Estimated cost from current node to goal
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

# checked 
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_neighbors(pos: Tuple[int, int], grid_size: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    x, y = pos
    neighbors = []
    
    # Check all four directions
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        
        # Check if within grid bounds
        if 0 <= new_x < grid_size[1] and 0 <= new_y < grid_size[0]:
            # Check if not an obstacle
            if (new_x, new_y) not in obstacles:
                neighbors.append((new_x, new_y))
    
    return neighbors

def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid_size: Tuple[int, int], 
           obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Implement A* pathfinding algorithm
    """
    start_node = Node(start, 0, manhattan_distance(start, goal))
    open_list = [start_node]
    closed_set = set()
    nodes = {start: start_node}

    while open_list:
        current = heappop(open_list)
        
        if current.position == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        closed_set.add(current.position)

        for neighbor_pos in get_neighbors(current.position, grid_size, obstacles):
            if neighbor_pos in closed_set:
                continue

            g_cost = current.g_cost + 1
            h_cost = manhattan_distance(neighbor_pos, goal)

            if neighbor_pos not in nodes:
                neighbor = Node(neighbor_pos, g_cost, h_cost, current)
                nodes[neighbor_pos] = neighbor
                heappush(open_list, neighbor)
            else:
                neighbor = nodes[neighbor_pos]
                if g_cost < neighbor.g_cost:
                    neighbor.g_cost = g_cost
                    neighbor.f_cost = g_cost + h_cost
                    neighbor.parent = current

    return []  # No path found

def get_obstacles(env) -> Set[Tuple[int, int]]:
    """
    Get positions of shelves and other obstacles
    """
    obstacles = set()
    
    # Get shelf positions from the environment grid
    
    shelf_array = env.shelfs

    
    for shelf in shelf_array:
        obstacles.add((shelf.x, shelf.y))
    
    # print("Obstacles:",obstacles)
    # input("Press Enter to continue...")
    return obstacles

def get_action_from_calculated_path(current_pos, next_pos,agent):
    """
    Convert position change to warehouse action
    
    Action(Enum): 
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    
    Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


    """
    current_direction = agent.dir.value
    # debug ! 
    # Calculate desired direction based on position difference
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    if dx == 1:
        desired_direction = Direction.RIGHT.value
    elif dx == -1:
        desired_direction = Direction.LEFT.value 
    elif dy == 1:
        desired_direction = Direction.DOWN.value
    else: # dy == -1
        desired_direction = Direction.UP.value
    
    # If already facing the desired direction, move forward
    if current_direction == desired_direction:
        return Action.FORWARD.value
    
    # Need to turn - figure out which way
    if current_direction == Direction.UP.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            return [Action.RIGHT.value,Action.FORWARD.value]
        elif desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            return [Action.LEFT.value,Action.FORWARD.value]
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            return [Action.RIGHT.value,Action.RIGHT.value]
    
    elif current_direction == Direction.DOWN.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            return [Action.LEFT.value,Action.FORWARD.value]
        elif desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            return [Action.RIGHT.value,Action.FORWARD.value]
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            return [Action.RIGHT.value,Action.RIGHT.value]
    
    elif current_direction == Direction.LEFT.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            return [Action.RIGHT.value,Action.RIGHT.value]
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            return [Action.LEFT.value,Action.FORWARD.value]
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            return [Action.RIGHT.value,Action.FORWARD.value]
    
    elif current_direction == Direction.RIGHT.value:
        if desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            return [Action.RIGHT.value,Action.RIGHT.value]
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            return [Action.RIGHT.value,Action.FORWARD.value]
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            return[Action.LEFT.value,Action.FORWARD.value]
        
        
def visualize_paths(paths, current_pos, goals, obstacles, grid_size):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
        
    # Plot obstacles
    obstacle_x = [obs[0] for obs in obstacles]
    obstacle_y = [obs[1] for obs in obstacles]
    ax.scatter(obstacle_x, obstacle_y, c='red', marker='s', s=500, label='Obstacles')
        
        # Plot all valid paths
    for i, path in enumerate(paths):
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        ax.plot(path_x, path_y, '--', linewidth=2, label=f'Path {i+1}')
        
    # Plot start and goals
    ax.scatter(current_pos[0], current_pos[1], c='green', s=200, label='Start')
    goal_x = [g[0] for g in goals]
    goal_y = [g[1] for g in goals]
    ax.scatter(goal_x, goal_y, c='blue', s=200, label='Goals')
        
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-1, grid_size[1])
    ax.set_ylim(-1, grid_size[0])
    ax.set_title('A* Pathfinding Visualization')
    plt.show(block=False)        
        
def run_warehouse_with_astar():
    # Create environment
    env = Warehouse(1, 4, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    obs, info = env.reset()

    # Get obstacles
    obstacles = get_obstacles(env)
    
    # Get goal positions from environment
    goals = env.goals 

    # Get initial agent position and calculate path
    agent = env.agents[0]  # Only using first agent
    current_pos = (agent.x, agent.y)
    
    print("Starting Position:",current_pos)
    current_direction = agent.dir.value
    
    # save the current direction, as it will be changed in the get_action_from_calculated_path function
    print("Starting Direction:",current_direction)
    
    # Find path to nearest goal
    # Find path to nearest goal using Manhattan distance
    nearest_goal = min(goals, key=lambda g: manhattan_distance(current_pos, g))
    path = a_star(current_pos, nearest_goal, env.grid_size, obstacles)
    valid_paths = [path] if path else []  # Keep same format for visualization

    if not valid_paths:
        print("No valid path found!")
        return
        
    # Get the shortest valid path
    path = min(valid_paths, key=len)
    print(f"Found path of length {len(path)}: {path}")
        # Call visualization function
    # visualize_paths(valid_paths, current_pos, goals, obstacles, env.grid_size)
    
    # Convert path to series of actions
    action_sequence = []
    
    for i in range(len(path)-1):
        pos = path[i]
        next_pos = path[i+1]
        action = get_action_from_calculated_path(pos, next_pos,agent)
        if type(action) == list:
            for a in action:
                action_sequence.append(a)
        else:
            action_sequence.append(action)

   
    # Action Enum 
    # NOOP = 0
    # FORWARD = 1
    # LEFT = 2
    # RIGHT = 3
    
    # for toggle load , we ignore it first, now goal is to find a path from initialised position to goal 
    # TOGGLE_LOAD = 4

    
    # Translate numerical actions to readable format
    action_names = {
        0: 'NOOP',
        1: 'FORWARD', 
        2: 'LEFT',
        3: 'RIGHT',
        4: 'TOGGLE_LOAD'
    }
    
    direction_names = {
        0: 'UP',
        1: 'DOWN',
        2: 'LEFT',
        3: 'RIGHT'
    }
    
    readable_actions = [action_names[a] for a in action_sequence]
    print("Action Sequence (numeric):", action_sequence)
    print("Action Sequence (readable):", readable_actions)
    
    
    # Set agent direction to the initial direction
    agent.dir = Direction(current_direction)
    print("Agent Starting Direction Reset:",agent.dir.value)
    
    # Execute action sequence
    for action in action_sequence:
        actions = [Action.NOOP.value] * env.n_agents
        actions[0] = action  # Set action for first agent
        print("Action:",action_names[action])
       
        obs, rewards, done, truncated, info = env.step(actions)
        print("Current Direction:",direction_names[agent.dir.value])
        env.render()
        time.sleep(1)
        
        # Check if agent reached goal
        agent_pos = (env.agents[0].x, env.agents[0].y)
        if agent_pos in goals:
            print("Agent successfully reached goal!")
            
            input("Press Enter to continue...")
            
        if done:
            obs, info = env.reset()
            obstacles = get_obstacles(env)
            break

if __name__ == "__main__":
    try:
        run_warehouse_with_astar()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")