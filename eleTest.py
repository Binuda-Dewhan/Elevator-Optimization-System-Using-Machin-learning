import multiprocessing
import time
import os

# ✅ Import your YOLO and Elevator Simulation functions
from YOLO import run_yolo  # YOLO function for extracting passenger data
# from ENV import ElevatorEnv  # Elevator Environment class
from ENVsql import ElevatorEnv  # Elevator Environment class

def run_simulation():
    """Function to run the Elevator Simulation"""
    env = ElevatorEnv()  # Initialize the environment
    
    while True:
        obs, _, _, _ = env.step()  # Run the simulation step
        env.render_2d()  # Update visualization
        time.sleep(1)  # Delay to match real-time updates

if __name__ == "__main__":
    # ✅ Step 1: Ensure MySQL Server is Running Before Starting
    print("🚀 Starting YOLO & Elevator Simulation...")

    # ✅ Step 2: Create multiprocessing for YOLO and Simulation
    yolo_process = multiprocessing.Process(target=run_yolo)  # YOLO in a separate process
    sim_process = multiprocessing.Process(target=run_simulation)  # Simulation in a separate process

    # ✅ Step 3: Start both processes
    yolo_process.start()
    sim_process.start()

    # ✅ Step 4: Keep the processes running
    yolo_process.join()
    sim_process.join()
