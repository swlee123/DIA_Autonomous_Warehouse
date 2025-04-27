import gymnasium as gym
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import time
from rware.warehouse import Warehouse, RewardType, Action, Direction
import matplotlib.pyplot as plt
import math
from utils import record_window_to_video


# handle collision for mutiple agent
# problem :  
# implement a mechanism that check if a agent stay at same place for mutiple tick (5), and was not holding anything, and at goal position 
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

    # print(f"Start: {start}, Goal: {goal}, Obstacles: {obstacles}")
    while open_list:
        current = heappop(open_list)
        
        if current.position == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            
            # print(f"Path found: {path}")
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

def get_obstacles(env,i):
    """
    Get positions of shelves and other obstacles
    """
    obstacles = set()
    # assume dealing with only one agent for now
    agent = env.agents[i]
    

    # print(f"in obstacle function Agent {i} , carrying shelf: {agent.carrying_shelf}")
    
    
    # Get shelf positions from the environment grid
    
    shelf_array = env.shelfs

    
    # also add obstacles(obj name), check if obs exist cuz sometimes it is not 
    if env.obstacles_loc is not None:
        for loc in env.obstacles_loc:
            obstacles.add((loc[0], loc[1]))
            
    if env.humans is not None:
        for human in env.humans:
            obstacles.add((human.x, human.y))
    
    if agent.carrying_shelf is not None:      
    
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



    # print(f"dx: {dx}, dy: {dy}")
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
        # print("Invalid move: next_pos must differ from current_pos by 1 unit.")        

        return Action.RECAL.value
    
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
    requested_shelves = [shelf for shelf in shelves if shelf in request_queue and shelf.taken == False]
    
    # for s in request_queue:
    #     print(f"Request Queue: ({s.x}, {s.y}), Shelf.taken {s.taken}")
    
    # for shelf in requested_shelves:
    #     print(f"Shelf Position: ({shelf.x}, {shelf.y}), Shelf.taken {shelf.taken} In Request Queue: {shelf in request_queue}")
   

    if not requested_shelves:
        
        return None
        
    # Find the nearest requested shelf using Manhattan distance
    nearest_shelf = None 
  

    available_shelf = [shelf for shelf in requested_shelves]
    
    if available_shelf:    
        nearest_shelf = min(available_shelf, key=lambda s: manhattan_distance(current_pos, (s.x, s.y)))
    
    nearest_shelf.taken = True
    
    # print(f"Nearest Shelf Position: ({nearest_shelf.x}, {nearest_shelf.y})")
    return nearest_shelf

def random_wait_after_collision():
    
    wait_count = np.random.randint(1, 3)
    
    return wait_count

def random_action():
    
    # introduce some randomness in the action taken by the agent when no path is found
    
    # print("Random action called !")
    return np.random.choice([Action.NOOP.value, Action.LEFT.value, Action.RIGHT.value])


def run_warehouse_with(algorithm,TICKS,interval,agent_count=1,shelf_column=3,column_height=3,shelf_rows=2,human_count=0,obstacles_loc=None):
    

    # Create environment
    # shelf_columns,column_height,shelf_rows,n_agents,msg_bits,sensor_range,request_queue_size
    env_name = None
    if obstacles_loc is None and human_count == 0:
        env = Warehouse(shelf_column,column_height,shelf_rows, agent_count, 0, 1, 5, None, None, RewardType.GLOBAL)
        env_name = "Normal"
    elif obstacles_loc is not None and human_count > 0:
        env =   Warehouse(shelf_column,column_height,shelf_rows, agent_count, 0, 1, 5, None, None, RewardType.GLOBAL, obstacles_loc=obstacles_loc , human_count=human_count)
        env_name = "Obstacle & Human"
    elif obstacles_loc is not None :
        env = Warehouse(shelf_column,column_height,shelf_rows, agent_count, 0, 1, 5, None, None, RewardType.GLOBAL, obstacles_loc=obstacles_loc)
        env_name = "Obstacle"
    elif human_count > 0:
        env = Warehouse(shelf_column,column_height,shelf_rows, agent_count, 0, 1, 5, None, None, RewardType.GLOBAL, human_count=human_count)
        env_name = "Human"
        
        
    obs, info = env.reset()
    
    algo_function = None
    
    if algorithm == 'A*':
        algo_function = a_star
    elif algorithm == 'Dijkstra':
        algo_function = dijkstra
    else:
        raise ValueError("Invalid algorithm. Choose 'A*' or 'Dijkstra'.")
    
    # Get goal positions from environment
    goals = env.goals 
    
    agent_paths = [[] for _ in range(env.n_agents)]  # Initialize paths for each agent
    agent_state = [1 for _ in range(env.n_agents)]  # Initialize all agents to state 1

    saved_shelf_pos = [None for _ in range(env.n_agents)]  # Initialize saved shelf positions for each agent
    target_pos = [None for _ in range(env.n_agents)]  # Initialize target positions for each agent
    target_shelf = [None for _ in range(env.n_agents)]  # Initialize target shelves for each agent
    
    
    
    final_reward = [0.0 for _ in range(env.n_agents)]  # Initialize final rewards for each agent
    
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
    
    Ticks = TICKS
    
    collision = 0
    
    
    while Ticks>0:
        
        # if not recorded: 
        #     # Start recording in a separate thread
        #     recording_thread = threading.Thread(target=record_experiment, args=("auto_experiment.py", video_name, TICKS * interval, 30))
        #     recording_thread.start()
            
        #     # Set recorded flag to True after starting the thread
        #     recorded = True

                
        # global var to store occupied position by other agent ,
        # to avoid collision
        occupied_positions = set()
        
        actions = [Action.NOOP.value] * env.n_agents
        
        for i,agent in enumerate(env.agents):
            
            current_pos = (agent.x, agent.y)



            # print(f"Agent {i} Current Position: {current_pos}, State: {agent_state[i]}")
            obstacles = get_obstacles(env,i)
            # If agent is not carrying a shelf, find nearest shelf with object
            
            if agent.random_wait > 0:
                actions[i] = Action.NOOP.value
                agent.random_wait -= 1
                # print(f"Agent {i} is waiting for {agent.random_wait} ticks.")
                
            # check if there is a path to follow 
            elif agent_paths[i] != []:
                
                # If there is a path, get the next position from the path, and proceed until all path is finished
                target_pos[i] = agent_paths[i][0]
                # single (x,y)
                
                
                # Get action from current position to next position
                actions[i] = get_next_action(current_pos, target_pos[i], agent)
                
                # collision detection and handling when agent move to empty position
                if actions[i] is Action.RECAL.value or (agent_state[i] == 1  and current_pos in occupied_positions):
 
                    
                    path = a_star(current_pos, target_pos[i], env.grid_size, obstacles)[1:]
                    
                    agent_paths[i] = path
                    
                    # let agent wait random time before moving again to prevent deadlock
                    agent.random_wait = random_wait_after_collision()
                    
                    # input("Collision detected! Recalculating route...")
                    continue                    
                    
                    
                    
                    
                # print(f"Current Pos : {current_pos} Next pos: {target_pos[i]}, Action: {action_names[actions[i]]}")
                
                
                if actions[i] is Action.FORWARD.value:
                    
                    # collision handling and reroute
                    if target_pos[i] in occupied_positions:
                        
                        # print("Collision detected! Rerouting...")
                        
                        collision += 1
                        
                        temp_obstacles = obstacles.copy()
                        
                        for a in occupied_positions:
                            temp_obstacles.add(a)
                        
                        # If the next position is occupied, find a new path to the target position
                        new_path = algo_function(current_pos, target_pos[i], env.grid_size, temp_obstacles)[1:]
                            
                        if not new_path:
                            # print("No valid new path found!")
                            actions[i] = random_action()
                            continue
                            
                            
                        # input(f"Rerouted path of length {len(new_path)}: {new_path}")
                        
                        agent_paths[i] = new_path
                    else:
                        # If the next position is not occupied, add it to occupied positions
                        occupied_positions.add(target_pos[i])
                        occupied_positions.add(current_pos)
                        
                        # print("Moving forward...")
                        # If the agent is moving forward and reaches the next position, pop it from the path
                        agent_paths[i].pop(0)
                

            # If there is no path, either it is at goal or at shelf location  
            else:
                
            
                
                # agent reached the shelf location and not carrying shelf, pick up shelf 
                if agent_state[i] == 1 and current_pos == target_pos[i] and agent.carrying_shelf is None:
                    # print("Reached shelf. Loading...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    agent_state[i] = 2
                
        
                    
                    
                # agent reached goal with shelf , unload 
             
                elif agent_state[i] == 2 and current_pos in goals:
                    # handle unloading 
                    # print("Unloading shelf at goal...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    agent_state[i] = 4

                # agent returned to shelf location and carrying shelf , unload
                elif agent_state[i] == 4 and current_pos == saved_shelf_pos[i] and agent.carrying_shelf:
                    # print("Returned shelf. Unloading shelf at original place...")
                    actions[i] = Action.TOGGLE_LOAD.value
                    
                    # set the shelf.taken in request queue to False
                    for shelf in env.request_queue:
                        if shelf.x == target_shelf[i].x and shelf.y == target_shelf[i].y:
                            shelf.taken = False
                            
                    # env.agents[i].carrying_shelf = None
                    target_shelf[i] = None
            
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
                            # print(f"[STATE 1] Moving to shelf at {target_pos[i]}")
                            
                    
                        else:
                            # input("No shelves with objects found!")
                            actions[i] = Action.NOOP.value
                            continue
                            
                    elif agent_state[i] == 2:
                        # Carrying a loaded shelf, moving to goal

                        target_pos[i] = min(goals, key=lambda g: manhattan_distance(current_pos, (g[0], g[1])))
                        # print(f"[STATE 2] Moving to goal at {target_pos[i]}")

                    elif agent_state[i] == 3:
                        # At goal with loaded shelf, need unload
                        
                        # handle unloading 
                        # print(f"[STATE 3] At goal. Unloading shelf...")
                        # input(agent_paths[i])
                        agent_state[i] = 4
                        continue
                        
                    elif agent_state[i] == 4:
                        # Carrying a shelf without load, need to move to shelf location to put back the shelf
                        target_pos[i] = (saved_shelf_pos[i][0], saved_shelf_pos[i][1])
                        # print(f"[STATE 4] Returning shelf to original location at {target_pos[i]}")
        
                    # Find path to target pos 
                    path = a_star(current_pos, target_pos[i], env.grid_size, obstacles)[1:]
                    
                    if not path:
                        # input("No valid path found!")
                        actions[i] = random_action()
                        continue
                        
                    # print(f"Found path of length {len(path)}: {path}")
                    
                    agent_paths[i] = path
                    
                    # Set agent direction to the initial direction
    
            
        # 
        obs, reward,  done, truncated, info = env.step(actions)

        final_reward+= reward
        
        # print("REWARD:", reward)
        env.render()
        # print("Current Direction:", direction_names[agent.dir.value])
        time.sleep(interval)
        
        Ticks-=1
    
    env.reset()
    env.close()
    
    print("Simulation finished.")
    
    
    r = 0.0 
    for fr in final_reward:
        r+=fr
        
        
    reward_per_tick = r / TICKS
    collisions_per_tick = collision / TICKS
    
    return r, reward_per_tick , collision, collisions_per_tick
        
        
        



# if __name__ == "__main__":
#     try:
        
#         # Run the warehouse simulation with A* algorithm and 3 agents
#         run_warehouse_with("A*",3)
        
        
        
#     except KeyboardInterrupt:
#         print("\nSimulation stopped by user")