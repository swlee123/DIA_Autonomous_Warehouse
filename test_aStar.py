import gymnasium as gym
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import time
from rware.warehouse import Warehouse, RewardType
from run_aStar import Node, manhattan_distance, get_neighbors

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
    
    shelf_array = env.unwrapped.shelfs
    print("Shelf Array:",shelf_array)
    
    
    for shelf in shelf_array:
        obstacles.add((shelf.x, shelf.y))
    
    # print("Obstacles:",obstacles)
    # input("Press Enter to continue...")
    return obstacles


def visualize_path(start: Tuple[int, int], goal: Tuple[int, int], 
                  path: List[Tuple[int, int]], grid_size: Tuple[int, int], 
                  obstacles: Set[Tuple[int, int]]):
    """
    Visualize the path with obstacles
    """
    # Create grid
    grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    
    # Mark obstacles
    for x, y in obstacles:
        grid[y][x] = '#'
    
    # Mark path
    for x, y in path:
        grid[y][x] = '*'
    
    # Mark start and goal
    grid[start[1]][start[0]] = 'S'
    grid[goal[1]][goal[0]] = 'G'
    
    # Print grid
    print("\nPath Visualization:")
    print("S: Start")
    print("G: Goal")
    print("#: Obstacle")
    print("*: Path")
    print("-" * (grid_size[1] * 2 + 1))
    for row in grid:
        print("|" + "|".join(row) + "|")
    print("-" * (grid_size[1] * 2 + 1))

# Example usage:
def test_pathfinding():
    env = gym.make("rware:rware-tiny-2ag-v2", render_mode="human")
    obs, info = env.reset()
    
    # Get obstacles
    obstacles = get_obstacles(env)
    
    # Get start (agent position) and goal
    agent = env.unwrapped.agents[0]
    start = (agent.x, agent.y)
    goal = env.unwrapped.goals[0]  # First goal position
    
    # Find path
    path = a_star(start, goal, env.unwrapped.grid_size, obstacles)
    
    # Visualize
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles: {obstacles}")
    visualize_path(start, goal, path, env.unwrapped.grid_size, obstacles)
    
    return path

path = test_pathfinding()