import gym
import numpy as np
import pygame
import pandas as pd
from gym import spaces
import time
import pandas as pd
import os
from datetime import datetime, timedelta

import mysql.connector

# ✅ Database Connection Setup
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",   # Change this if MySQL is hosted elsewhere
        user="root",        # Your MySQL username (default is 'root')
        password="root",  # Change this to your MySQL password
        database="elevator_system"  # Database name
    )



class ElevatorEnv(gym.Env):
    def __init__(self, num_floors=6, num_elevators=3):
        super(ElevatorEnv, self).__init__()
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.time_per_floor = 5  # 5 seconds per floor movement
        self.time_per_step = 5  # 5 seconds per step in simulation
        self.max_capacity = 10  # Maximum passengers per elevator

        # ✅ Initialize Metrics
        self.service_times = []  # Stores time taken to serve passengers
        self.passenger_board_times = {}  # Tracks when each passenger enters the elevator
        self.passenger_wait_times = {}  # Tracks when passengers request an elevator
        self.wait_times = []  # Stores wait times for tracking

        # Store CSV file path but DO NOT load it fully at start
        # self.csv_file = csv_file
        self.current_index = 0  # Start processing passengers from the first row
        

        # State: Elevator positions, waiting passengers, and elevator loads
        self.state = {
            'elevator_positions': np.ones(num_elevators, dtype=int),  # Elevators start at floor 1
            'passengers_waiting': {floor: {'up': [], 'down': []} for floor in range(1, num_floors+1)},
            'elevator_passengers': [[] for _ in range(num_elevators)],
            'elevator_actions': [1] * num_elevators,  # 1 = idle initially
            'elevator_load': [0] * num_elevators  # Track number of passengers inside each elevator
        }
        
        # Actions: Move up (+1), Move down (-1), Stay (0) for each elevator
        self.action_space = spaces.MultiDiscrete([3] * num_elevators)
        
        # Observation space: Elevator positions
        self.observation_space = spaces.Dict({
            'elevator_positions': spaces.MultiDiscrete([num_floors] * num_elevators),
        })
        
        # ✅ Set initial simulation time (earliest entry in CSV)
        # ✅ Simulation Time Setup
        self.current_time = pd.to_datetime("08:00:00 AM", format="%I:%M:%S %p")  # Always start at 08:00 AM

        self.energy_usage = []
        self.service_efficiency = []
        
        # Pygame Setup
        pygame.init()
        self.screen_width = 800
        self.screen_height = max(600, num_floors * 100)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Elevator Simulation")
        self.clock = pygame.time.Clock()
    

    def step(self, action=None):
        """Run the simulation continuously, adding new passengers dynamically."""
        
        energy_consumed = 0  # Track energy usage per step

        # ✅ Keep Pygame running (Do NOT restrict by time)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
        # ✅ Move Simulation Time Forward by `time_per_step`
        self.current_time += timedelta(seconds=self.time_per_step)
        
        
        # # ✅ Check for new passengers from YOLO-generated CSV
        # self.update_passengers_from_csv()
        
        # ✅ Fetch new passenger data from MySQL
        self.update_passengers_from_db()


        # ✅ Detect Rush or Non-Rush Time
        is_rush_time = self.detect_rush_time()

        for i in range(self.num_elevators):
            if self.state['elevator_load'][i] > 0:
                move = self.move_to_passenger_destination(i)  # ✅ Handle passengers inside
            else:
                move = self.nearest_car_scan(i) if is_rush_time else self.energy_efficient_routing(i)  # ✅ Follow rush/non-rush algorithm

            # ✅ Update Elevator Position
            old_position = self.state['elevator_positions'][i]  
            new_position = old_position + move  

            if 1 <= new_position <= self.num_floors:
                self.state['elevator_positions'][i] = new_position
                energy_consumed += abs(new_position - old_position)  # ✅ Track energy usage

            # ✅ Handle Passenger Pickup & Dropoff
            self.handle_passenger_movement(i, new_position)

        # ✅ Get the new state of the environment
        obs = self._get_observation()
        
        info = {
            'energy': energy_consumed,
            'total_wait_time': np.sum(self.wait_times) if self.wait_times else 0,
            'total_service_time': np.sum(self.service_times) if self.service_times else 0
        }

        self.energy_usage.append(energy_consumed)

        # ✅ Update Pygame display
        self.render_2d()

        time.sleep(1)  # ✅ Slow down simulation for smoother visuals

        return obs, 0, False, info

    
    def move_to_passenger_destination(self, elevator_index):
        """Move towards the destination floor of the nearest passenger inside the elevator."""
        
        # ✅ Get all passenger destinations inside the elevator
        if not self.state['elevator_passengers'][elevator_index]:
            return 0  # Stay idle if no passengers

        current_floor = self.state['elevator_positions'][elevator_index]
        destinations = [dest for _, dest in self.state['elevator_passengers'][elevator_index]]

        # ✅ Find the closest destination floor
        nearest_destination = min(destinations, key=lambda x: abs(x - current_floor))

        return np.sign(nearest_destination - current_floor)  # Move towards the nearest destination

    
    def detect_rush_time(self):
        """Determine if it's rush time based on multiple real-time conditions."""

        # ✅ Thresholds for Rush Time Detection
        rush_passenger_threshold = 5   # More than 5 passengers waiting → Rush
        rush_request_threshold = 10    # More than 10 elevator requests per minute → Rush
        rush_occupancy_threshold = 70  # Elevators more than 70% full → Rush
        rush_peak_hours = [(8, 10), (17, 19)]  # Typical peak hours (Morning & Evening)
        rush_stop_threshold = 15  # More than 15 stops per minute → Rush

        # ✅ Calculate Current Conditions
        total_waiting = sum(len(v['up']) + len(v['down']) for v in self.state['passengers_waiting'].values())
        total_requests = len(self.passenger_wait_times)  # Requests made so far
        average_occupancy = (sum(self.state['elevator_load']) / (self.num_elevators * self.max_capacity)) * 100
        elevator_stops = sum(1 for e in self.state['elevator_positions'] if e in self.state['passengers_waiting'])

        # ✅ Check if Current Time Falls in Peak Hours
        current_hour = self.current_time.hour
        is_peak_hour = any(start <= current_hour <= end for start, end in rush_peak_hours)

        # ✅ Determine Rush Time (If Any 3 Conditions are Met, Set as Rush)
        rush_conditions = [
            total_waiting > rush_passenger_threshold,
            total_requests > rush_request_threshold,
            average_occupancy > rush_occupancy_threshold,
            is_peak_hour,
            elevator_stops > rush_stop_threshold
        ]
        
        return sum(rush_conditions) >= 3  # If 3 or more conditions are met → Rush Time

    def handle_passenger_movement(self, elevator_index, new_position):
        """Handles passengers getting in and out of the elevator."""

        # ✅ Drop off passengers at their destination
        self.state['elevator_passengers'][elevator_index] = [
            (p_id, dest) for p_id, dest in self.state['elevator_passengers'][elevator_index] if dest != new_position
        ]

        self.state['elevator_load'][elevator_index] = len(self.state['elevator_passengers'][elevator_index])

        # ✅ Pick up new passengers
        for direction in ['up', 'down']:
            while self.state['passengers_waiting'][new_position][direction] and self.state['elevator_load'][elevator_index] < self.max_capacity:
                passenger_id, destination = self.state['passengers_waiting'][new_position][direction].pop(0)

                self.state['elevator_passengers'][elevator_index].append((passenger_id, destination))
                self.state['elevator_load'][elevator_index] += 1

                print(f"🚪 Passenger {passenger_id} ENTERED Elevator {elevator_index+1} at Floor {new_position} → Destination {destination}")




    def nearest_car_scan(self, elevator_index):
        """Implements Nearest Car Assignment + SCAN Algorithm."""
        current_floor = self.state['elevator_positions'][elevator_index]

        # Find nearest passenger request
        nearest_request = None
        min_distance = float('inf')

        for floor in range(1, self.num_floors + 1):
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']:
                distance = abs(floor - current_floor)
                if distance < min_distance:
                    min_distance = distance
                    nearest_request = floor

        if nearest_request:
            return np.sign(nearest_request - current_floor)  # Move towards request

        return 0  # Stay idle if no requests
    
    def energy_efficient_routing(self, elevator_index):
        """Implements Smart Routing for Energy Efficiency."""
        current_floor = self.state['elevator_positions'][elevator_index]

        # Find nearby requests within the assigned zone
        assigned_zone = (elevator_index % 3) + 1  # Assign zones dynamically
        zone_start = (assigned_zone - 1) * (self.num_floors // self.num_elevators) + 1
        zone_end = assigned_zone * (self.num_floors // self.num_elevators)

        for floor in range(zone_start, zone_end + 1):
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']:
                return np.sign(floor - current_floor)  # Move towards request

        return 0  # Stay idle if no requests

    
    # def update_passengers_from_csv(self):
    #     """Read CSV & Add new passengers who haven't been processed yet."""
    #     try:
    #         df = pd.read_csv(self.csv_file)

    #         # ✅ Convert 'Time' column to datetime format
    #         df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce')
    #         df = df.dropna(subset=['Time'])  # Remove invalid timestamps

    #         # ✅ Filter for passengers whose request time is <= current simulation time
    #         new_passengers = df[(df['Time'] <= self.current_time) & (~df['Passenger ID'].isin(self.processed_passengers))]

    #         for _, row in new_passengers.iterrows():
    #             passenger_id = row['Passenger ID']
    #             floor = int(row['Floor'])
    #             direction = 'up' if row['Direction (Up/Down)'].strip().lower() == 'up' else 'down'
    #             destination = int(row['Destination Floor'])

    #             # ✅ Only add passenger if they haven't been added yet
    #             if passenger_id not in self.processed_passengers:
    #                 self.state['passengers_waiting'][floor][direction].append((passenger_id, destination))
    #                 self.processed_passengers.add(passenger_id)  # ✅ Mark as processed

    #                 print(f"🟢 New Passenger {passenger_id} added at {self.current_time.strftime('%I:%M:%S %p')} | Floor {floor} → {destination}")

    #     except Exception as e:
    #         print(f"❌ Error Reading CSV: {e}")
    
    def update_passengers_from_db(self):
        """Fetch passenger data from MySQL and update the environment."""
        try:
            conn = connect_to_db()
            cursor = conn.cursor(dictionary=True)

            # ✅ Query: Fetch passengers whose request time is less than or equal to the current simulation time
            cursor.execute("SELECT * FROM passengers WHERE STR_TO_DATE(time, '%I:%i:%S %p') <= %s", 
                        (self.current_time.strftime('%I:%M:%S %p'),))
            new_passengers = cursor.fetchall()

            conn.close()  # ✅ Close the connection

            for row in new_passengers:
                passenger_id = row['passenger_id']
                floor = int(row['floor'])
                direction = 'up' if row['direction'].strip().lower() == 'up' else 'down'
                destination = int(row['destination_floor'])

                if (passenger_id, destination) not in self.state['passengers_waiting'][floor][direction]:
                    self.state['passengers_waiting'][floor][direction].append((passenger_id, destination))
                    print(f"🟢 Passenger {passenger_id} requested elevator at {self.current_time}")

        except Exception as e:
            print(f"❌ Error Fetching Data from MySQL: {e}")


    
    
    
            
    def _get_observation(self):
        """Return current state as observation."""
        return {
            'elevator_positions': self.state['elevator_positions'].copy(),
            'elevator_actions': self.state['elevator_actions'].copy(),
            'elevator_load': self.state['elevator_load'].copy(),
            'average_wait_time': sum(self.wait_times) if self.wait_times else 0,
            'average_service_time': np.mean(self.service_times) if self.service_times else 0
        }


    def reset(self):
        """Reset environment to initial state."""
        self.state['elevator_positions'] = np.ones(self.num_elevators, dtype=int)
        self.current_time = self.passenger_data['Time'].min()
        try:
            df = pd.read_csv(self.csv_file)
            df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')
            self.current_time = df['Time'].min()  # ✅ Get first timestamp
        except Exception:
            self.current_time = pd.to_datetime("08:00:00 AM", format="%I:%M:%S %p")  # ✅ Default to 08:00 AM

        self.energy_usage = []
        self.service_times = []
        self.wait_times = []
        self.passenger_board_times = {}
        self.passenger_wait_times = {}
        return self._get_observation()
    
            
    def track_wait_time(self, passenger_id, boarding_time):
        """Calculate and store wait time when a passenger enters the elevator."""
        if passenger_id in self.passenger_wait_times:
            request_time = self.passenger_wait_times.pop(passenger_id)  # Retrieve request time
            wait_time = (boarding_time - request_time).total_seconds()
            self.wait_times.append(wait_time)  # ✅ Store wait time

            # 🔍 Debugging Output
            print(f"🟡 Passenger {passenger_id} entered elevator at {boarding_time} (Requested at {request_time})")
            print(f"⏳ Wait Time Recorded: {wait_time} seconds")


    def track_service_time(self, passenger_id, exit_time):
        """Calculate and store service time when a passenger reaches their destination."""
        if passenger_id in self.passenger_board_times:
            boarding_time = self.passenger_board_times.pop(passenger_id)
            service_time = (exit_time - boarding_time).total_seconds()
            self.service_times.append(service_time)  # ✅ Store service time

            # 🔍 Debugging Output
            print(f"🔴 Passenger {passenger_id} exited elevator at {exit_time} (Boarded at {boarding_time})")
            print(f"⌛ Service Time Recorded: {service_time} seconds")

            
    # def render_2d(self):
    #     """Render the environment using Pygame."""
    #     self.screen.fill((255, 255, 255))  
    #     floor_height = self.screen_height // self.num_floors

    #     # ✅ Display simulation time
    #     font = pygame.font.Font(None, 30)
    #     time_text = font.render(f"Time: {self.current_time.strftime('%I:%M:%S %p')}", True, (0, 0, 0))
    #     self.screen.blit(time_text, (self.screen_width // 2 - 50, 10))

    #     # ✅ Determine Rush or Non-Rush Mode
    #     is_rush_time = self.detect_rush_time()
    #     rush_mode_text = "RUSH TIME" if is_rush_time else "NON-RUSH TIME"
    #     rush_mode_color = (255, 0, 0) if is_rush_time else (0, 150, 0)  # Red for Rush, Green for Non-Rush

    #     rush_text = font.render(f"MODE: {rush_mode_text}", True, rush_mode_color)
    #     self.screen.blit(rush_text, (self.screen_width // 2 - 50, 40))

    #     # ✅ Display Total Wait Time, Service Time, and Energy Consumption
    #     total_wait = sum(self.wait_times) if self.wait_times else 0
    #     total_service = sum(self.service_times) if self.service_times else 0
    #     total_energy = sum(self.energy_usage) if self.energy_usage else 0

    #     stats_text = font.render(
    #         f"🚶 Wait Time: {total_wait:.2f}s  |  ⏳ Service Time: {total_service:.2f}s  |  ⚡ Energy: {total_energy} units",
    #         True, (0, 0, 0)
    #     )
    #     self.screen.blit(stats_text, (self.screen_width // 2 - 250, 70))

    #     # ✅ Draw floors and waiting passengers
    #     for floor in range(1, self.num_floors+1):
    #         y = self.screen_height - floor * floor_height
            
    #         # Draw floor line
    #         pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.screen_width, y), 2)
            
    #         # Display floor number
    #         font = pygame.font.Font(None, 24)
    #         text = font.render(f"Floor {floor}", True, (0, 0, 0))
    #         self.screen.blit(text, (10, y + 5))

    #         # Display waiting passengers for up and down directions
    #         up_count = len(self.state['passengers_waiting'][floor]['up'])
    #         down_count = len(self.state['passengers_waiting'][floor]['down'])
            
    #         up_text = font.render(f"⬆ {up_count}", True, (255, 0, 0))  # Red for up
    #         down_text = font.render(f"⬇ {down_count}", True, (0, 0, 255))  # Blue for down
    #         self.screen.blit(up_text, (self.screen_width - 80, y + 5))
    #         self.screen.blit(down_text, (self.screen_width - 40, y + 5))

    #     # ✅ Draw elevators with load count
    #     elevator_width = 40
    #     elevator_spacing = self.screen_width // (self.num_elevators + 1)
    #     for i, pos in enumerate(self.state['elevator_positions']):
    #         x = (i + 1) * elevator_spacing - (elevator_width // 2)
    #         y = self.screen_height - pos * floor_height + 5  
    #         pygame.draw.rect(self.screen, (0, 0, 255), (x, y, elevator_width, floor_height - 10))

    #         # ✅ Display passengers inside elevator
    #         font = pygame.font.Font(None, 24)
    #         text = font.render(f"{self.state['elevator_load'][i]}/10", True, (255, 255, 255))
    #         self.screen.blit(text, (x + 10, y + 20))

    #     pygame.display.flip()
    #     self.clock.tick(10)
    
    import pygame

    def render_2d(self):
        """Render the Elevator Environment in Pygame (Fixed & Optimized)."""

        # ✅ Handle Pygame Events to Prevent Freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # ✅ Clear the Screen
        self.screen.fill((0, 0, 0))  # Black Background

        # ✅ Calculate Floor Heights
        floor_height = self.screen_height // self.num_floors
        font = pygame.font.Font(None, 28)

        # ✅ Display Simulation Time
        # ✅ Display Simulation Time (Updating in Real-Time)
        time_text = font.render(f"Time: {self.current_time.strftime('%I:%M:%S %p')}", True, (255, 255, 255))
        self.screen.blit(time_text, (self.screen_width // 2 - 50, 10))
        
        
        # ✅ Check Rush or Non-Rush Mode
        is_rush_time = self.detect_rush_time()
        rush_mode_text = "RUSH TIME" if is_rush_time else "NON-RUSH TIME"
        rush_mode_color = (255, 0, 0) if is_rush_time else (0, 255, 0)  # Red for Rush, Green for Non-Rush

        rush_text = font.render(f"MODE: {rush_mode_text}", True, rush_mode_color)
        self.screen.blit(rush_text, (self.screen_width // 2 - 50, 40))

        # ✅ Draw Floors and Elevators
        for floor in range(1, self.num_floors + 1):
            y = self.screen_height - floor * floor_height

            # Draw Floor Lines
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_width, y), 2)

            # Floor Number
            floor_label = font.render(f"Floor {floor}", True, (255, 255, 255))
            self.screen.blit(floor_label, (10, y + 5))

            # Draw Waiting Passengers at Each Floor
            up_count = len(self.state['passengers_waiting'][floor]['up'])
            down_count = len(self.state['passengers_waiting'][floor]['down'])

            up_text = font.render(f"⬆ {up_count}", True, (255, 255, 0))  # Yellow for Up
            down_text = font.render(f"⬇ {down_count}", True, (0, 255, 255))  # Cyan for Down
            self.screen.blit(up_text, (self.screen_width - 100, y + 5))
            self.screen.blit(down_text, (self.screen_width - 50, y + 5))

        # ✅ Draw Elevators
        elevator_width = 50
        elevator_spacing = self.screen_width // (self.num_elevators + 1)

        for i, pos in enumerate(self.state['elevator_positions']):
            x = (i + 1) * elevator_spacing - (elevator_width // 2)
            y = self.screen_height - pos * floor_height + 5  
            pygame.draw.rect(self.screen, (0, 0, 255), (x, y, elevator_width, floor_height - 10))

            # ✅ Display Passengers Inside Elevator
            elevator_label = font.render(f"{self.state['elevator_load'][i]}/10", True, (255, 255, 255))
            self.screen.blit(elevator_label, (x + 10, y + 20))

        # ✅ Update Display and Prevent Freezing
        pygame.display.update()
        self.clock.tick(30)  # Limit FPS to 30

    def close(self):
        pygame.quit()

