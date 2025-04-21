import gymnasium as gym
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import time
from rware.warehouse import Warehouse, RewardType, Action, Direction
import importlib 


import matplotlib.pyplot as plt

# basically the implementation is based on the A* algorithm 
# but the heuristic function is adjusted to Dijkstra algorithm


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

def get_neighbors(pos, grid_size, obstacles):
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

def dijkstra(start, goal, grid_size, obstacles):
    """
    Implement Dijkstra shortest path finding algorithm
    Parameters:
    - start: Starting position (x, y)
    - goal: Goal position (x, y)
    - grid_size: Size of the grid (height, width)
    - obstacles: Set of obstacle positions (x, y)
    - carrying_shelf: True/False
    
    Returns:
    - List of positions [(x1, y1), (x2, y2), ...] representing the path from start to goal
    """
    
    # the manhattan distance is not used here in Dijkstra 
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
            
            # the Node implementation is based on A* but we dont need h_cost for Dijkstra, so we set it to 0
            h_cost = 0 
            
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

def get_obstacles(env):
    """
    Get positions of shelves and other obstacles
    """
    obstacles = set()
    # assume dealing with only one agent for now
    agent = env.agents[0]
    
    # if agent is not carrying a shelf, there are no obstacles
    if agent.carrying_shelf is None:
        return set()
    
    
    # Get shelf positions from the environment grid
    
    shelf_array = env.shelfs

    # also add obstacles(obj name), check if obs exist cuz sometimes it is not 
    if env.obstacles_loc is not None:
        for loc in env.obstacles_loc:
            obstacles.add((loc[0], loc[1]))
            
    for shelf in shelf_array:
        obstacles.add((shelf.x, shelf.y))
        
    print("Obstacles:", obstacles)
    
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
    
    "Full action transition , from pos to next_pos"
    
    """
    1. Turn the agent to the desired direction
    2. Move forward
    
    """
    
    
    actions = []
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
        
        actions.append(Action.FORWARD.value)
        return actions


    # Need to turn - figure out which way
    if current_direction == Direction.UP.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            actions.append(Action.RIGHT.value)
            
        elif desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            actions.append(Action.LEFT.value)
            
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            actions.append(Action.RIGHT.value)
            actions.append(Action.RIGHT.value)
            
                
    elif current_direction == Direction.DOWN.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            actions.append(Action.LEFT.value)
            
        elif desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            actions.append(Action.RIGHT.value)
            
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            actions.append(Action.RIGHT.value)
            actions.append(Action.RIGHT.value)
    
    elif current_direction == Direction.LEFT.value:
        if desired_direction == Direction.RIGHT.value:
            agent.dir = Direction.RIGHT
            actions.append(Action.LEFT.value)
            actions.append(Action.LEFT.value)
            
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            actions.append(Action.LEFT.value)
            
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            actions.append(Action.RIGHT.value)    
            
    elif current_direction == Direction.RIGHT.value:
        if desired_direction == Direction.LEFT.value:
            agent.dir = Direction.LEFT
            actions.append(Action.LEFT.value)
            actions.append(Action.LEFT.value)
            
            
        elif desired_direction == Direction.DOWN.value:
            agent.dir = Direction.DOWN
            actions.append(Action.RIGHT.value)
            
        elif desired_direction == Direction.UP.value:
            agent.dir = Direction.UP
            actions.append(Action.LEFT.value)
    
    actions.append(Action.FORWARD.value)
    
    return actions
    
    
        
        
def visualize_paths(paths, current_pos, goals, obstacles, grid_size):

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
        
def find_nearest_shelf_with_object(env, current_pos):
    """
    Find the nearest shelf that has an object (is in request queue)
    """
    shelves = env.shelfs
    request_queue = env.request_queue
    
    # Filter shelves that are in the request queue (target shelves to bring to goal point)
    requested_shelves = [shelf for shelf in shelves if shelf in request_queue]
    
    if not requested_shelves:
        return None
        
    # Find the nearest requested shelf using Manhattan distance
    nearest_shelf = min(requested_shelves, key=lambda s: manhattan_distance(current_pos, (s.x, s.y)))
    return nearest_shelf

def run_warehouse_with_dijkstra():
    

    agent_state = 1 
    # Create environment
    # using params to do so 

    obstacles_loc = [(1, 15)]
    
    env = Warehouse(1, 4, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL,obstacles_loc=obstacles_loc)
    obs, info = env.reset()
    
    # Get goal positions from environment
    goals = env.goals 
    agent = env.agents[0]  # Only using first agent
    current_pos = (agent.x, agent.y)
    current_direction = agent.dir.value
    
    moving_to_goal = False
        
    # saved point for shelf 
    saved_shelf_pos = None
    
    # Action Enum 
    action_names = {0: 'NOOP',1: 'FORWARD', 2: 'LEFT',3: 'RIGHT',4: 'TOGGLE_LOAD'}
        
    direction_names = {  0: 'UP',1: 'DOWN',2: 'LEFT',3: 'RIGHT'}    
    
    def execute_actions(action_sequence):
        for action in action_sequence:
            actions = [Action.NOOP.value] * env.n_agents
            actions[0] = action
            print("Action:", action_names[action])
            obs, rewards, done, truncated, info = env.step(actions)
            print("Current Direction:", direction_names[agent.dir.value])
            env.render()
            time.sleep(1)
        return obs, info
    
    # 4 state for agent . entry point is 1 , and 4 -> 1 
    
    # 1 :
    # Not carrying anyshelf , moving to nearest shelf with load 
    # 2 :
    # Carrying shelf with load , moving to goal
    # 3 :
    # Reach goal with load 
    # 4 :
    # Carrying shelf with without load , moving to shelf location to put back the shelf
    
    
    while True:
        print("--------------------------------")
        print("current_pos:",current_pos)
        print("Current Direction:", current_direction)
        print("agent dir:",agent.dir.value)
        obstacles = get_obstacles(env)
        # If agent is not carrying a shelf, find nearest shelf with object
        
        target_pos = None
    
        
        if agent_state == 1:
            # Not carrying a shelf, moving to nearest shelf with object
            
            # Find nearest shelf with object
            target_shelf = find_nearest_shelf_with_object(env, current_pos)
            if target_shelf:
                target_pos = (target_shelf.x, target_shelf.y)
                saved_shelf_pos = target_pos
                print(f"[STATE 1] Moving to shelf at {target_pos}")
            else:
                print("No shelves with objects found!")
                break
        elif agent_state == 2:
            # Carrying a loaded shelf, moving to goal

            target_pos = min(goals, key=lambda g: manhattan_distance(current_pos, (g[0], g[1])))
            print(f"[STATE 2] Moving to goal at {target_pos}")

        elif agent_state == 3:
            # At goal with loaded shelf, need upload
            
            # handle unloading 
            print(f"[STATE 3] At goal. Unloading shelf...")
            actions = [Action.NOOP.value] * env.n_agents
            actions[0] = Action.TOGGLE_LOAD.value
            env.step(actions)
            env.render()
            time.sleep(1)
            agent_state = 4
            continue
            
        elif agent_state == 4:
            # Carrying a shelf without load, moving to shelf location to put back the shelf
            target_pos = (saved_shelf_pos[0], saved_shelf_pos[1])
            print(f"[STATE 4] Returning shelf to original location at {target_pos}")

        # Find path to target pos 
        path = dijkstra(current_pos, target_pos, env.grid_size, obstacles)
        
        if not path:
            print("No valid path found!")
            break
            
        print(f"Found path of length {len(path)}: {path}")
        
        # Convert path to series of actions
        action_sequence = []
        
        for i in range(len(path)-1):
            pos = path[i]
            next_pos = path[i+1]
            action = get_action_from_calculated_path(pos, next_pos, agent)
            if type(action) == list:
                for a in action:
                    action_sequence.append(a)
            else:
                action_sequence.append(action)
        
        
        readable_actions = [action_names[a] for a in action_sequence]
        print("Action Sequence (readable):", readable_actions)
        # input("Press Enter to continue...")
        
        # Set agent direction to the initial direction
   
        agent.dir = Direction(current_direction)
        obs, info = execute_actions(action_sequence)

        current_pos = (agent.x, agent.y)
        current_direction = agent.dir.value

        if agent_state == 1 and current_pos == target_pos and agent.carrying_shelf is None:
            print("Reached shelf. Loading...")
            actions = [Action.NOOP.value] * env.n_agents
            actions[0] = Action.TOGGLE_LOAD.value
            env.step(actions)
            env.render()
            time.sleep(1)
            agent_state = 2

        elif agent_state == 2 and current_pos in goals and agent.carrying_shelf:
            print("Reached goal with shelf. Unloading...")
            agent_state = 3

        elif agent_state == 4 and current_pos == saved_shelf_pos and agent.carrying_shelf:
            print("Returned shelf. Unloading...")
            actions = [Action.NOOP.value] * env.n_agents
            actions[0] = Action.TOGGLE_LOAD.value
            env.step(actions)
            env.render()
            time.sleep(1)
            agent_state = 1

        # if done :
        #     print("Environment done. Resetting...")
        #     obs, info = env.reset()
        #     current_pos = (agent.x, agent.y)
        #     current_direction = agent.dir.value
        #     agent_state = 1
        #     saved_shelf_pos = None
        #     break

        


if __name__ == "__main__":
    try:
        run_warehouse_with_dijkstra()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")