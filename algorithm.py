import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import time

# ---------------- Helper Functions ----------------
moves = [(0,1),(0,-1),(1,0),(-1,0)]

def bfs_with_steps(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        (x, y), path = queue.popleft()
        if (x,y) == goal:
            yield visited.copy(), path
            return
        if (x,y) in visited:
            continue
        visited.add((x,y))
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and maze[nx][ny]==0 and (nx,ny) not in visited:
                queue.append(((nx,ny), path+[ (nx,ny) ]))
        yield visited.copy(), path
    yield visited, None

def dfs_with_steps(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    stack = [(start, [start])]
    while stack:
        (x, y), path = stack.pop()
        if (x,y) == goal:
            yield visited.copy(), path
            return
        if (x,y) in visited:
            continue
        visited.add((x,y))
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and maze[nx][ny]==0 and (nx,ny) not in visited:
                stack.append(((nx,ny), path+[ (nx,ny) ]))
        yield visited.copy(), path
    yield visited, None

def ucs_with_steps(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    pq = [(0, start, [start])]
    while pq:
        cost, (x,y), path = heapq.heappop(pq)
        if (x,y) == goal:
            yield visited.copy(), path
            return
        if (x,y) in visited:
            continue
        visited.add((x,y))
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and maze[nx][ny]==0 and (nx,ny) not in visited:
                heapq.heappush(pq, (cost+1, (nx,ny), path+[ (nx,ny) ]))
        yield visited.copy(), path
    yield visited, None

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_with_steps(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    pq = [(heuristic(start, goal), 0, start, [start])]
    while pq:
        f, g, (x,y), path = heapq.heappop(pq)
        if (x,y) == goal:
            yield visited.copy(), path
            return
        if (x,y) in visited:
            continue
        visited.add((x,y))
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and maze[nx][ny]==0 and (nx,ny) not in visited:
                new_g = g+1
                new_f = new_g + heuristic((nx,ny), goal)
                heapq.heappush(pq, (new_f, new_g, (nx,ny), path+[ (nx,ny) ]))
        yield visited.copy(), path
    yield visited, None

# ---------------- Streamlit App ----------------
st.title("🌐 Dynamic Search Algorithm Visualizer")

# ---------------- Sidebar ----------------
st.sidebar.header("Maze Settings")

# Maze input option
maze_option = st.sidebar.radio("Maze Input Method", ["Upload CSV", "Random Maze"])

maze = None  # initialize

if maze_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Maze CSV (0=free,1=wall)")
    if uploaded_file is not None:
        maze = np.loadtxt(uploaded_file, delimiter=",", dtype=int)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()  # stop the app until file is uploaded
else:
    rows = st.sidebar.slider("Rows", 5, 50, 10)
    cols = st.sidebar.slider("Cols", 5, 50, 10)
    density = st.sidebar.slider("Obstacle Density %", 0, 70, 30)
    np.random.seed(42)
    maze = np.random.choice([0,1], size=(rows, cols), p=[1-density/100, density/100])

# Now maze is guaranteed to exist
maze_rows, maze_cols = maze.shape
st.write(f"Maze Size: {maze_rows} x {maze_cols}")


# Select start and goal dynamically
start_row = st.number_input("Start Row", 0, maze_rows-1, 0)
start_col = st.number_input("Start Col", 0, maze_cols-1, 0)
goal_row = st.number_input("Goal Row", 0, maze_rows-1, maze_rows-1)
goal_col = st.number_input("Goal Col", 0, maze_cols-1, maze_cols-1)

start, goal = (start_row, start_col), (goal_row, goal_col)
maze[start] = 0
maze[goal] = 0

# Algorithm selection
algo = st.sidebar.selectbox("Choose Algorithm", ["BFS", "DFS", "UCS", "A*"])
speed = st.sidebar.slider("Animation Speed (sec)", 0.01, 1.0, 0.1)

# Run search animation
if st.button("Run Search"):
    if algo=="BFS":
        search = bfs_with_steps(maze, start, goal)
    elif algo=="DFS":
        search = dfs_with_steps(maze, start, goal)
    elif algo=="UCS":
        search = ucs_with_steps(maze, start, goal)
    else:
        search = astar_with_steps(maze, start, goal)

    placeholder = st.empty()
    final_path = None

    for step, (visited, path) in enumerate(search):
        if step % max(1, maze_rows*maze_cols//500) == 0:  # speed optimization for large maze
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(maze, cmap="gray_r")

            # Draw visited nodes
            for (x,y) in visited:
                ax.add_patch(plt.Rectangle((y-0.5,x-0.5),1,1,color="cyan",alpha=0.3))
            # Draw path
            if path:
                for (x,y) in path:
                    ax.add_patch(plt.Rectangle((y-0.5,x-0.5),1,1,color="yellow",alpha=0.6))
            # Start & Goal
            ax.scatter(start[1], start[0], c="green", s=200, marker="*")
            ax.scatter(goal[1], goal[0], c="red", s=200, marker="*")

            placeholder.pyplot(fig)
            time.sleep(speed)

        if path and path[-1]==goal:
            final_path = path
            break

    if final_path:
        st.success(f"✅ Path found! Length = {len(final_path)-1} steps")
    else:
        st.error("❌ No path found")
