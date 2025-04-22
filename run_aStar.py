import gymnasium as gym
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import time
from rware.warehouse import Warehouse, RewardType, Action, Direction
import matplotlib.pyplot as plt
import math


# handle collision for mutiple agent
# problem :  
#  1.multiple agent will deadlock each other and wait for other to move 
#  2. sometimes agent will move to shelf location that is already taken by other agent , and then stuck at there


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

def a_star(start, goal, grid_size, obstacles):
    """
    Implement A* pathfinding algorithm
    Parameters:
    - start: Starting position (x, y)
    - goal: Goal position (x, y)
    - grid_size: Size of the grid (height, width)
    - obstacles: Set of obstacle positions (x, y)
    - carrying_shelf: True/False
    
    Returns:
    - List of positions [(x1, y1), (x2, y2), ...] representing the path from start to goal
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

def get_obstacles(env,i):
    """
    Get positions of shelves and other obstacles
    """
    obstacles = set()
    # assume dealing with only one agent for now
    agent = env.agents[i]
    
    # if agent is not carrying a shelf, there are no obstacles
    if agent.carrying_shelf is None:
        return set()
    
    
    # Get shelf positions from the environment grid
    
    shelf_array = env.shelfs

    
    for shelf in shelf_array:
        obstacles.add((shelf.x, shelf.y))
    
    return obstacles

def get_next_action(current_pos, next_pos, agent):
    """
    Returns the immediate next action (turn or forward) to move from current_pos to next_pos.
    Only one action per call.
    """

    current_direction = agent.dir.value

    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]



    print(f"dx: {dx}, dy: {dy}")
    if dx == 1:
        desired_direction = Direction.RIGHT.value
    elif dx == -1:
        desired_direction = Direction.LEFT.value
    elif dy == 1:
        desired_direction = Direction.DOWN.value
    elif dy == -1:
        desired_direction = Direction.UP.value
    else:
        # this is where we detect moves that is not valid / collision occured before 
        print("Invalid move: next_pos must differ from current_pos by 1 unit.")        
        
        return Action.NOOP.value
    
    if desired_direction == current_direction:
        return Action.FORWARD.value
    
    elif current_direction == Direction.UP.value :
        if desired_direction == Direction.RIGHT.value:
            return Action.RIGHT.value
        elif desired_direction == Direction.LEFT.value:
            return Action.LEFT.value
        elif desired_direction == Direction.DOWN.value:
            return Action.LEFT.value
        
    elif current_direction == Direction.DOWN.value:
        if desired_direction == Direction.RIGHT.value:
            return Action.LEFT.value
        elif desired_direction == Direction.LEFT.value:
            return Action.RIGHT.value
        elif desired_direction == Direction.UP.value:
            return Action.RIGHT.value
    elif current_direction == Direction.LEFT.value:
        if desired_direction == Direction.UP.value:
            return Action.RIGHT.value
        elif desired_direction == Direction.DOWN.value:
            return Action.LEFT.value
        elif desired_direction == Direction.RIGHT.value:
            return Action.LEFT.value
    elif current_direction == Direction.RIGHT.value:
        if desired_direction == Direction.UP.value:
            return Action.LEFT.value
        elif desired_direction == Direction.DOWN.value:
            return Action.RIGHT.value
        elif desired_direction == Direction.LEFT.value:
            return Action.RIGHT.value
    
        
        
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
    nearest_shelf = None 
  

    available_shelf = [shelf for shelf in requested_shelves if shelf.taken == False]
    
    if available_shelf:    
        nearest_shelf = min(available_shelf, key=lambda s: manhattan_distance(current_pos, (s.x, s.y)))
    

    if nearest_shelf:
        nearest_shelf.taken = True
    
    
    return nearest_shelf

def run_warehouse_with_astar(agent_count=1):
    

    # Create environment
    # shelf_columns,column_height,shelf_rows,n_agents,msg_bits,sensor_range,request_queue_size
    env = Warehouse(1, 2, 3, agent_count, 0, 1, 5, None, None, RewardType.GLOBAL)
    obs, info = env.reset()
    
    # Get goal positions from environment
    goals = env.goals 
    
    agent_paths = [[] for _ in range(env.n_agents)]  # Initialize paths for each agent
    agent_state = [1 for _ in range(env.n_agents)]  # Initialize all agents to state 1
    # saved point for shelf 
    saved_shelf_pos = [None for _ in range(env.n_agents)]  # Initialize saved shelf positions for each agent
    target_pos = [None for _ in range(env.n_agents)]  # Initialize target positions for each agent
    target_shelf = [None for _ in range(env.n_agents)]  # Initialize target shelves for each agent
    
    
    # Action Enum 
    action_names = {0: 'NOOP',1: 'FORWARD', 2: 'LEFT',3: 'RIGHT',4: 'TOGGLE_LOAD'}
        
    direction_names = {  0: 'UP',1: 'DOWN',2: 'LEFT',3: 'RIGHT'}    
    
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
        
        # global var to store occupied position by other agent ,
        # to avoid collision
        occupied_positions = set()
        
        actions = [Action.NOOP.value] * env.n_agents
        
        for i,agent in enumerate(env.agents):

            current_pos = (agent.x, agent.y)



            print(f"Agent {i} Current Position: {current_pos}, State: {agent_state[i]}")
            obstacles = get_obstacles(env,i)
            # If agent is not carrying a shelf, find nearest shelf with object
            
            
            # check if there is a path to follow 
            if len(agent_paths[i]) > 0:
                
                # If there is a path, get the next position from the path, and proceed until all path is finished
                target_pos[i] = agent_paths[i][0]
                # single (x,y)
                
                
                # Get action from current position to next position
                actions[i] = get_next_action(current_pos, target_pos[i], agent)
                
                # collision detection and handling when agent move to empty position
                if actions[i] is Action.NOOP.value or (agent_state[i] == 1  and current_pos in occupied_positions):
                    
                    agent_paths[i] = []
                    
                    print("Collision detected! Press Enter to continue...")
                    continue                    
                    
                    
                    
                    
                print(f"Current Pos : {current_pos} Next pos: {target_pos[i]}, Action: {action_names[actions[i]]}")
                
                
                if actions[i] is Action.FORWARD.value:
                    
                    # collision handling and reroute
                    if target_pos[i] in occupied_positions:
                        
                        print("Collision detected! Rerouting...")
                        
                        temp_obstacles = obstacles.copy()
                        
                        for a in occupied_positions:
                            temp_obstacles.add(a)
                        
                        # If the next position is occupied, find a new path to the target position
                        new_path = a_star(current_pos, target_pos[i], env.grid_size, temp_obstacles)[1:]
                            
                        if not new_path:
                            print("No valid new path found!")
                            break
                            
                            
                        input(f"Rerouted path of length {len(new_path)}: {new_path}")
                        
                        agent_paths[i] = new_path
                    else:
                        # If the next position is not occupied, add it to occupied positions
                        occupied_positions.add(target_pos[i])
                        occupied_positions.add(current_pos)
                        
                        print("Moving forward...")
                        # If the agent is moving forward and reaches the next position, pop it from the path
                        agent_paths[i].pop(0)
                

            # If there is no path, either it is at goal or at shelf location  
            else:
                
    
                
                # agent reached the shelf location and not carrying shelf, pick up shelf 
                if agent_state[i] == 1 and current_pos == target_pos[i] and agent.carrying_shelf is None:
                    print("Reached shelf. Loading...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    agent_state[i] = 2
                
        
                    
                    
                # agent reached goal with shelf , unload 
                elif agent_state[i] == 2 and current_pos in goals and agent.carrying_shelf:
                
                    # handle unloading 
                    print("Unloading shelf at goal...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    agent_state[i] = 4

                # agent returned to shelf location and carrying shelf , unload
                elif agent_state[i] == 4 and current_pos == saved_shelf_pos[i] and agent.carrying_shelf:
                    print("Returned shelf. Unloading shelf at original place...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    
                    
                    target_shelf[i].taken = False
                    
                    target_pos[i] = None
                    saved_shelf_pos[i] = None
                    # reset target pos and saved shelf pos
                    agent_state[i] = 1
                
                # if none of the condition is met
                else : 

                    if agent_state[i] == 1:
                        # Not carrying a shelf, need to move to nearest shelf with object
                        
                        # Find nearest shelf with object
                        target_shelf[i] = find_nearest_shelf_with_object(env, current_pos)
                        
                        # check whether other agent is 
                        if target_shelf[i]:
                            target_pos[i] = (target_shelf[i].x, target_shelf[i].y)
                            saved_shelf_pos[i] = target_pos[i]
                            print(f"[STATE 1] Moving to shelf at {target_pos[i]}")
                            
                    
                        else:
                            print("No shelves with objects found!")
                            continue
                    elif agent_state[i] == 2:
                        # Carrying a loaded shelf, moving to goal

                        target_pos[i] = min(goals, key=lambda g: manhattan_distance(current_pos, (g[0], g[1])))
                        print(f"[STATE 2] Moving to goal at {target_pos[i]}")

                    elif agent_state[i] == 3:
                        # At goal with loaded shelf, need unload
                        
                        # handle unloading 
                        print(f"[STATE 3] At goal. Unloading shelf...")
                        agent_state[i] = 4
                        continue
                        
                    elif agent_state[i] == 4:
                        # Carrying a shelf without load, need to move to shelf location to put back the shelf
                        target_pos[i] = (saved_shelf_pos[i][0], saved_shelf_pos[i][1])
                        print(f"[STATE 4] Returning shelf to original location at {target_pos[i]}")

                    # Find path to target pos 
                    path = a_star(current_pos, target_pos[i], env.grid_size, obstacles)[1:]
                    
                    if not path:
                        print("No valid path found!")
                        actions[i] = Action.NOOP.value
                        continue
                        
                    print(f"Found path of length {len(path)}: {path}")
                    
                    agent_paths[i] = path
                    
                    # Set agent direction to the initial direction
    
            
        # 
        env.step(actions)
        env.render()
        print("Current Direction:", direction_names[agent.dir.value])
        time.sleep(0.2)
        
        
        



if __name__ == "__main__":
    try:
        run_warehouse_with_astar(2)
        
        
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")