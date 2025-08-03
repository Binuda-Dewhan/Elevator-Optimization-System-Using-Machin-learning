import gym
import numpy as np
import pygame
import pandas as pd
from gym import spaces
import firebase_admin
from firebase_admin import credentials, db, firestore
from datetime import datetime, timedelta
# Camera transformation using spherical coordinates
from math import sin, cos, radians
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from OpenGL.GLUT import glutInit

from OpenGL.GL import *
from OpenGL.GLUT import *

import mysql.connector

# ✅ Database Connection Setup
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",   # Change this if MySQL is hosted elsewhere
        user="root",        # Your MySQL username (default is 'root')
        password="root",  # Change this to your MySQL password
        database="elevator_system"  # Database name
    )


glutInit()

# Only initialize once
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\Binuda Dewhan\Desktop\V2\serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://elevator-personalization-default-rtdb.firebaseio.com"
    })
# Initialize Firestore for Component 2 only if not already initialized
try:
    app2 = firebase_admin.get_app("traffic-firestore-app")
except ValueError:
    cred2 = credentials.Certificate(r"C:\Users\Binuda Dewhan\Desktop\V2\serviceAccountKey_component2.json")
    app2 = firebase_admin.initialize_app(cred2, name="traffic-firestore-app")

# Get Firestore DB client for Component 2
db2 = firestore.client(app=app2)

# ✅ Safely initialize a named app
try:
    maintenance_app = firebase_admin.get_app("maintenance-firestore-app")
except ValueError:
    cred = credentials.Certificate(r"C:\Users\Binuda Dewhan\Desktop\V2\serviceAccountKey_maintenance.json")
    maintenance_app = firebase_admin.initialize_app(cred, name="maintenance-firestore-app")

# ✅ Get the Firestore client from the named app
db_maintenance = firestore.client(app=maintenance_app)


MODE_COLOR_MAP = {
    "VIP":         (1.0, 0.4, 0.7),
    "PRESCHEDULE": (0.4, 0.6, 1.0),
    "RUSH":        (1.0, 0.0, 0.0),
    "MAINTENANCE": (0.5, 0.5, 0.5),
    "NORMAL":      (0.0, 1.0, 0.0),
    "ENERGY-SAVING": (0.0, 1.0, 0.5),
    "DYNAMIC-ASSIGN": (1.0, 1.0, 0.0),
}



class ElevatorEnv(gym.Env):
    def __init__(self, num_floors=6, num_elevators=3):
        super(ElevatorEnv, self).__init__()
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.time_per_floor = 5  # 5 seconds per floor movement
        self.time_per_step = 5   # 5 seconds per simulation step
        self.max_capacity = 10   # Elevator capacity

        # 🧠 VIP + Prioritization Logic
        self.vip_targets = []
        self.vip_elevator_id = None
        self.active_reservation_window = None
        self.handled_vips = set()

        # 🔮 Pre-scheduling Logic from Firestore
        self.pre_schedule = self.fetch_peak_demand_data_from_firestore()
        self.elevator_targets = {}
        self.preschedule_active = False
        self.preschedule_event = None
        self.elevators_arrived = set()

        # 🛠️ Predictive Maintenance Logic
        self.maintenance_schedule = self.fetch_maintenance_schedule()
        self.maintenance_active = False
        self.maintenance_elevator_id = None
        self.maintenance_window = None

        print("🛠️ Maintenance schedule loaded:", self.maintenance_schedule)

        # 📊 Metrics
        self.service_times = []
        self.passenger_board_times = {}
        self.passenger_wait_times = {}
        self.wait_times = []
        self.processed_passengers = set()  # ✅ Track who is already added


        # ✅ Start simulation from 8 AM (or you can make it dynamic)
        self.current_time = pd.to_datetime("08:00:00 AM", format="%I:%M:%S %p")

        # ⚡ Energy usage tracking
        self.energy_usage = []
        self.service_efficiency = []

        # 📦 Elevator State
        self.state = {
            'elevator_positions': np.ones(num_elevators, dtype=int),
            'passengers_waiting': {floor: {'up': [], 'down': []} for floor in range(1, num_floors + 1)},
            'elevator_passengers': [[] for _ in range(num_elevators)],
            'elevator_actions': [1] * num_elevators,
            'elevator_load': [0] * num_elevators
        }

        # 🕹️ Action/Observation Space
        self.action_space = spaces.MultiDiscrete([3] * num_elevators)
        self.observation_space = spaces.Dict({
            'elevator_positions': spaces.MultiDiscrete([num_floors] * num_elevators),
        })

        # 🎥 Camera Setup (for 3D)
        self.camera_distance = 40
        self.camera_angle_y = 0
        self.camera_angle_x = 20

        # 🎮 Pygame GUI Setup
        pygame.init()
        self.screen_width = 800
        self.screen_height = max(600, num_floors * 100)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Elevator Simulation")
        self.clock = pygame.time.Clock()

        
    def fetch_reservations(self):
        ref = db.reference("reservations")
        raw_data = ref.get()
        reservations = {}

        if raw_data:
            for user_uid, res_entry in raw_data.items():
                for reservation_id, details in res_entry.items():
                    details['reservation_id'] = reservation_id
                    details['user_uid'] = user_uid
                    reservations[user_uid] = details
                    break  # Only take latest reservation per user
        return reservations
    
    
    def fetch_recognized_users(self):
        ref = db.reference("recognized_users_log")
        logs = ref.get()
        
        if not logs:
            return []

        recognized_users = []
        
        for user_id, date_entries in logs.items():
            for timestamp, info in date_entries.items():
                if isinstance(info, dict) and "firebaseUID" in info:
                    info['userID'] = user_id
                    info['recognized_time'] = timestamp
                    recognized_users.append(info)
                    
        return recognized_users
    
    # def fetch_recognized_users(self):
    #     ref = db.reference("recognized_users")
    #     return ref.get() or {}
    
    def fetch_peak_demand_data_from_firestore(self):
        predictions_ref = db2.collection("unique_prediction")
        docs = predictions_ref.stream()

        schedule = {}

        for doc in docs:
            entry = doc.to_dict()
            try:
                floor = int(entry.get("floor", 0))
                num_elevators = int(entry.get("num_elevators", 0))

                # 🕒 Extract and round timestamp1 to the nearest lower 5-second interval
                raw_timestamp = entry.get("timestamp1", "")
                if raw_timestamp:
                    dt_obj = datetime.fromisoformat(raw_timestamp)

                    # Round down to nearest 5 seconds
                    seconds = dt_obj.second
                    rounded_seconds = seconds - (seconds % 5)
                    rounded_dt = dt_obj.replace(second=rounded_seconds, microsecond=0)

                    # Convert to AM/PM format as "12:56:00 AM"
                    time_key = rounded_dt.strftime("%I:%M:%S %p")  # 👈 Here’s your change

                    schedule[time_key] = {
                        "floor": floor,
                        "num_elevators": num_elevators,
                        "timestamp_str": raw_timestamp,
                        "time_only": time_key,
                        "passengers_up": entry.get("passengers_up", 0),
                        "passengers_down": entry.get("passengers_down", 0),
                        "is_peak_demand_time": entry.get("is_peak_demand_time", 0),
                        "is_peak_floor": entry.get("is_peak_floor", 0)
                    }

            except Exception as e:
                print(f"⚠️ Error reading prediction entry: {e}")

        return schedule
    
    def fetch_maintenance_schedule(self):
        schedule_ref = db_maintenance.collection("sensor-data-prediction")
        docs = schedule_ref.stream()

        schedule = {}

        for doc in docs:
            data = doc.to_dict()
            if data.get("maintenance_required", False):
                # Handle possible time formatting issues
                raw_time = data.get("time") or data.get("time ") or ""
                raw_date = data.get("date", "")

                try:
                    # Parse raw time to datetime object
                    raw_time_clean = raw_time.strip()
                    dt_obj = pd.to_datetime(raw_time_clean, format="%I:%M %p")

                    # Round down to nearest 5 seconds
                    seconds = dt_obj.second
                    rounded_seconds = seconds - (seconds % 5)
                    rounded_dt = dt_obj.replace(second=rounded_seconds, microsecond=0)

                    # Format into AM/PM style string (same as preschedule)
                    time_key = rounded_dt.strftime("%I:%M:%S %p")

                    schedule[time_key] = {
                        "maintenance_required": True,
                        "elevator_id": 0,  # default elevator ID
                        "active": True,
                        "raw_time": raw_time_clean,
                        "date": raw_date
                    }

                except Exception as e:
                    print(f"⚠️ Error parsing maintenance time: {raw_time} → {e}")

        print("🛠️ Maintenance schedule loaded:", schedule)
        return schedule




    def track_reservations(self):
        reservations = self.fetch_reservations()

        for uid, res in reservations.items():
            if uid in self.handled_vips:
                continue  # ✅ Skip already handled VIP

            try:
                res_time = pd.to_datetime(res.get('time'), format='%I:%M %p')
            except Exception as e:
                print(f"❌ Error parsing reservation time: {e}")
                continue

            if 0 <= (self.current_time - res_time).total_seconds() <= 60:
                if not self.active_reservation_window:
                    self.active_reservation_window = {
                        'firebaseUID': res['firebaseUID'],
                        'reservation': res,
                        'start_time': self.current_time
                    }
                    self.handled_vips.add(uid)  # ✅ Mark as handled
                    print(f"🔔 VIP reservation triggered for UID: {res['firebaseUID']} at {self.current_time}")
                    return

    # def check_vip_recognition(self):
    #     """Detect if the VIP has been recognized via Firebase."""
    #     if not self.active_reservation_window:
    #         return None

    #     vip_uid = self.active_reservation_window['firebaseUID']
    #     recognized_users = self.fetch_recognized_users()

    #     for _, user in recognized_users.items():
    #         if user.get('firebaseUID') == vip_uid:
    #             print(f"👑 VIP recognized: {vip_uid}")
    #             return user  # Match found
    #     return None
    
    def check_vip_recognition(self):
        """Detect if the VIP has been recognized via Firebase."""
        if not self.active_reservation_window:
            return None

        vip_uid = self.active_reservation_window['firebaseUID']
        recognized_users = self.fetch_recognized_users()

        for user_data in recognized_users:
            if user_data.get('firebaseUID') == vip_uid:
                print(f"👑 VIP recognized: {vip_uid}")
                user_data["reservation"] = self.active_reservation_window["reservation"]
                user_data["firebaseUID"] = vip_uid
                return user_data

        return None
    
    def generate_vip_passengers(self, vip_user):
        
        if self.vip_targets:
            return  # ✅ Already generated
        
        """Spawn VIP group at entry floor for pickup."""
        reservation = vip_user['reservation']
        entry = int(reservation['entryFloor'])
        dest = int(reservation['destinationFloor'])
        count = int(reservation['numberOfPeople'])

        # Track VIP group
        self.vip_targets = [{
            'entry_floor': entry,
            'destination_floor': dest,
            'firebaseUID': vip_user['firebaseUID'],
            'group_size': count,
            'picked_up': False,
            'wait_start_time': self.current_time
        }]

        for i in range(count):
            pid = f"VIP_{vip_user['firebaseUID']}_{i}"
            self.state['passengers_waiting'][entry]['up'].append((pid, dest))
            self.passenger_wait_times[entry] = self.current_time

        print(f"🎯 VIP group of {count} generated at Floor {entry} → going to {dest}")



    def step(self, action=None):
        """Update the environment state based on elevator mode."""
        energy_consumed = 0  # Track energy usage per step

        # 🔁 Phase 1: Monitor and handle reservations
        self.track_reservations()

        # 🔁 Check if we are in VIP waiting window
        vip_user = None
        if self.active_reservation_window:
            vip_user = self.check_vip_recognition()
            
            if not vip_user and not self.vip_targets:
                reservation = self.active_reservation_window['reservation']
                entry = int(reservation['entryFloor'])
                dest = int(reservation['destinationFloor'])
                count = int(reservation['numberOfPeople'])
                uid = reservation['firebaseUID']

                self.vip_targets = [{
                    'entry_floor': entry,
                    'destination_floor': dest,
                    'firebaseUID': uid,
                    'group_size': count,
                    'picked_up': False,
                    'wait_start_time': self.current_time
                }]

            if vip_user:
                self.generate_vip_passengers(vip_user)  # ✅ Recognized within 1 min
                self.active_reservation_window = None
            else:
                # ✅ Still within the 1-minute wait window?
                start_time = self.active_reservation_window['start_time']
                if (self.current_time - start_time).total_seconds() > 60:
                    print(f"❌ VIP No-Show: Releasing reservation for {self.active_reservation_window['firebaseUID']}")
                    self.active_reservation_window = None

        # ✅ Detect current elevator mode (VIP, VIP_PENDING, RUSH, etc.)
        mode = self.detect_elevator_mode()
        
        # ✅ Generate time key as "HH:MM:SS" string
        time_key = self.current_time.strftime("%I:%M:%S %p")

        if mode == "PRESCHEDULE":
            if not self.preschedule_active and time_key in self.pre_schedule:
                self.preschedule_active = True
                self.preschedule_event = self.pre_schedule[time_key]
                self.elevator_targets = {}
                self.elevators_arrived = set()
                print(f"🚨 PRESCHEDULE MODE: Assigning {self.preschedule_event['num_elevators']} elevators to Floor {self.preschedule_event['floor']}")

            # Assign elevators dynamically based on availability
            floor = self.preschedule_event["floor"]
            required = self.preschedule_event["num_elevators"]

            assigned_count = len(self.elevator_targets)
            if assigned_count < required:
                for eid in range(self.num_elevators):
                    if eid in self.elevators_arrived:
                        continue
                    if self.state["elevator_load"][eid] > 0:
                        continue
                    if eid not in self.elevator_targets:
                        self.elevator_targets[eid] = floor
                        assigned_count += 1
                        print(f"🚀 Assigned Elevator {eid} to Pre-Schedule Floor {floor}")
                        if assigned_count >= required:
                            break

        for i in range(self.num_elevators):
            
            if mode == "MAINTENANCE":
                if i == self.maintenance_elevator_id:
                    move = 0  # 🚫 Elevator is offline
                elif self.state['elevator_load'][i] > 0:
                    move = self.move_to_passenger_destination(i)  # 🔄 Drop off passengers
                else:
                    move = self.nearest_car_scan(i)  # 🆕 Pick up waiting passengers
                            
            elif mode in ["VIP", "VIP_PENDING"] and self.vip_targets:
                # (existing VIP logic continues here)
                vip_assigned = self.assign_vip_elevator(self.vip_targets[0]['entry_floor'])

                if i == vip_assigned:
                    move = self.handle_vip_routing(i)  # <-- This now handles both VIP and VIP_PENDING
                    if move is None:
                        move = 0
                else:
                    move = self.default_elevator_logic(i, mode)
            else:
                move = self.default_elevator_logic(i, mode)

            # ⬆️ Apply movement
            old_position = self.state['elevator_positions'][i]
            new_position = old_position + move

            if 1 <= new_position <= self.num_floors:
                self.state['elevator_positions'][i] = new_position
                energy_consumed += abs(new_position - old_position)

            # 👥 Prevent passenger handling on maintenance elevator
            if not (mode == "MAINTENANCE" and i == self.maintenance_elevator_id):
                self.handle_passenger_movement(i, new_position)

        # 🕒 Advance simulation time and generate new passengers
        self.current_time += pd.Timedelta(seconds=self.time_per_step)
        self.update_passengers()
        
        # ✅ Exit PRESCHEDULE mode if all required elevators have arrived
        if self.preschedule_active and len(self.elevators_arrived) >= self.preschedule_event["num_elevators"]:
            print("✅ PRESCHEDULE MODE COMPLETE — All required elevators have arrived.")
            self.preschedule_active = False
            self.preschedule_event = None
            self.elevator_targets = {}
            self.elevators_arrived = set()
        
        # 🎯 Gather observation and metrics
        obs = self._get_observation()
        done = False

        if self.passenger_wait_times:
            total_wait_time = sum((self.current_time - arrival_time).total_seconds()
                                for arrival_time in self.passenger_wait_times.values())
        else:
            total_wait_time = 0

        info = {
            'mode': mode,
            'energy': energy_consumed,
            'total_wait_time': np.sum(self.wait_times) if self.wait_times else 0,
            'total_service_time': np.sum(self.service_times) if self.service_times else 0,
            'passenger_wait_times': self.passenger_wait_times.copy()
        }

        self.energy_usage.append(energy_consumed)
        return obs, 0, done, info

    def default_elevator_logic(self, i, mode):
        if self.state['elevator_load'][i] > 0:
            return self.move_to_passenger_destination(i)
        
        if mode == "PRESCHEDULE":
            return self.handle_preschedule_routing(i)  # ✅ New dedicated method
        
        elif mode == "RUSH":
            return self.nearest_car_scan(i)
        elif mode == "DYNAMIC-ASSIGN":
            return self.dynamic_assign_routing(i)
        elif mode == "NORMAL":
            return self.energy_efficient_routing(i)
        else:
            return self.energy_efficient_routing_best(i)
        
    def handle_preschedule_routing(self, i):
        if i in self.elevator_targets:
            target_floor = self.elevator_targets[i]
            current_floor = self.state["elevator_positions"][i]

            if current_floor != target_floor:
                return np.sign(target_floor - current_floor)
            else:
                self.elevators_arrived.add(i)
                print(f"✅ Elevator {i} reached Floor {target_floor} for PRESCHEDULE")
                return 0  # Stay at floor once arrived
        else:
            # 🔁 Let others behave as usual
            return self.energy_efficient_routing(i)


    
    def assign_vip_elevator(self, entry_floor):
        """Assign closest completely idle elevator (empty and not moving) to VIP."""
        if self.vip_elevator_id is not None:
            return self.vip_elevator_id

        best_elevator = None
        best_distance = float('inf')

        for i in range(self.num_elevators):
            is_empty = self.state['elevator_load'][i] == 0
            is_not_handling_other_passengers = len(self.state['elevator_passengers'][i]) == 0

            if is_empty and is_not_handling_other_passengers:
                dist = abs(self.state['elevator_positions'][i] - entry_floor)
                if dist < best_distance:
                    best_distance = dist
                    best_elevator = i

        if best_elevator is not None:
            self.vip_elevator_id = best_elevator
            print(f"👑 Elevator {best_elevator} assigned to VIP at Floor {entry_floor}")
        else:
            print(f"⚠️ No idle elevator available for VIP at Floor {entry_floor}")

        return best_elevator
    
    def handle_vip_routing(self, elevator_index):
        """Special movement logic for the VIP-assigned elevator only."""

        # 🚫 Sanity check
        if not self.vip_targets or self.vip_elevator_id != elevator_index:
            return None

        vip = self.vip_targets[0]
        entry = vip['entry_floor']
        dest = vip['destination_floor']
        curr_floor = self.state['elevator_positions'][elevator_index]

        # ✅ If not yet picked up → move to entry floor
        if not vip['picked_up']:
            if curr_floor != entry:
                return np.sign(entry - curr_floor)  # Go to VIP entry floor

            # ✅ At entry floor: clean waiting list to include only VIPs
            vip_prefix = f"VIP_{vip['firebaseUID']}_"

            # ✅ Check if VIPs boarded
            onboard_vips = any(
                str(pid).startswith(vip_prefix)
                for pid, _ in self.state['elevator_passengers'][elevator_index]
            )

            if onboard_vips:
                print(f"✅ VIPs already onboard Elevator {elevator_index}, proceeding to destination")
                vip['picked_up'] = True
                return np.sign(dest - curr_floor)

            # ✅ Still waiting in queue
            vip_waiting = any(
                str(pid).startswith(vip_prefix)
                for pid, _ in self.state['passengers_waiting'][entry]['up']
            )

            if vip_waiting:
                print(f"✅ VIP recognized and picked up at Floor {entry}")
                vip['picked_up'] = True
                return np.sign(dest - curr_floor)

            # ⏳ Still within wait time
            wait_time = (self.current_time - vip['wait_start_time']).total_seconds()
            if wait_time < 60:
                print(f"⏳ Elevator {elevator_index} waiting at Floor {entry} for VIP ({wait_time:.0f}s)")
                return 0

            # ❌ Timeout, release
            print(f"❌ VIP not found within 60 seconds. Releasing VIP elevator.")
            self.vip_targets = []
            self.vip_elevator_id = None
            self.active_reservation_window = None
            return 0

        # ✅ If picked up, go to destination
        if curr_floor == dest:
            print(f"🎉 VIP dropped off at Floor {dest}")
            self.vip_targets = []
            self.vip_elevator_id = None
            self.active_reservation_window = None
            self.handled_vips.add(vip['firebaseUID'])  # Optional redundancy
            return 0

        return np.sign(dest - curr_floor)

        
    def cleanup_expired_reservations(self):
        """Cancel reservation if VIP was not recognized within time window."""
        if self.active_reservation_window:
            start_time = self.active_reservation_window['start_time']
            if (self.current_time - start_time).total_seconds() > 120:
                print(f"❌ VIP No-Show: Releasing reservation for {self.active_reservation_window['firebaseUID']}")
                self.active_reservation_window = None



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

    
    def detect_elevator_mode(self):
        """Determine the operating mode: Energy-Saving, Normal, or Rush Mode."""
        
         # 🔧 Maintenance mode takes top priority
        if not self.maintenance_active:
            current_time_str = self.current_time.strftime("%I:%M:%S %p")
            for ts, info in self.maintenance_schedule.items():
                if info['active'] and current_time_str == ts:
                    self.maintenance_active = True
                    self.maintenance_elevator_id = info['elevator_id']
                    self.maintenance_start_time = self.current_time
                    print(f"🛠️ Maintenance STARTED on Elevator {self.maintenance_elevator_id} at {ts}")
                    return "MAINTENANCE"

        elif self.maintenance_active:
            elapsed = (self.current_time - self.maintenance_start_time).total_seconds()
            if elapsed >= 300:  # 5 minutes passed
                print(f"✅ Maintenance ENDED on Elevator {self.maintenance_elevator_id}")
                self.maintenance_active = False
                self.maintenance_elevator_id = None
                self.maintenance_start_time = None
            else:
                return "MAINTENANCE"
        
        if self.preschedule_active:
            return "PRESCHEDULE"
        
        current_time_str = self.current_time.strftime("%I:%M:%S %p")  # "07:08:35 AM"
        if current_time_str in self.pre_schedule:
            return "PRESCHEDULE"
        
        if self.active_reservation_window:
            elapsed = (self.current_time - self.active_reservation_window['start_time']).total_seconds()
            if elapsed <= 60:
                return "VIP_PENDING"

        if self.vip_targets:
            return "VIP"

        # ✅ Define thresholds for each mode
        rush_passenger_threshold =   10 # 7+ passengers per floor = RUSH
        normal_passenger_threshold = 3  # 4-6 passengers per floor = NORMAL
        # energy_saving_threshold = 3  # ≤3 passengers per floor = ENERGY-SAVING
        
        rush_request_threshold = 30  # More than 10 requests/min = RUSH
        normal_request_threshold = 6  # 6-10 requests/min = NORMAL
        
        rush_occupancy_threshold = 70  # Elevators >70% full = RUSH
        normal_occupancy_threshold = 40  # Elevators 40-70% full = NORMAL
        
        # rush_total_waiting_threshold = 15  # ✅ New Condition: More than 15 passengers waiting → RUSH
        
        # peak_hours = [(8, 10), (17, 19)]  # Rush Time (Morning & Evening)
        
        # ✅ New Condition: If **any single floor** has 10+ waiting passengers → RUSH
        for floor in range(1, self.num_floors + 1):
            total_floor_requests = len(self.state['passengers_waiting'][floor]['up']) + len(self.state['passengers_waiting'][floor]['down'])
            if total_floor_requests > 10:
                return "RUSH"
        
        total_waiting = sum(len(v['up']) + len(v['down']) for v in self.state['passengers_waiting'].values())
        total_requests = len(self.passenger_wait_times)  
        average_occupancy = (sum(self.state['elevator_load']) / (self.num_elevators * self.max_capacity)) * 100
        # elevator_stops = sum(1 for e in self.state['elevator_positions'] if e in self.state['passengers_waiting'])
        
        current_hour = self.current_time.hour
        # is_peak_hour = any(start <= current_hour <= end for start, end in peak_hours)
        
         # ✅ NEW CONDITION: Check if **any elevator is handling passengers**, but a new request appears
        for floor in range(1, self.num_floors + 1):
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']:
                
                # ✅ Find the closest elevator
                closest_elevator = None
                min_distance = float('inf')
                
                for i in range(self.num_elevators):
                    elevator_floor = self.state['elevator_positions'][i]
                    distance = abs(floor - elevator_floor)

                    # ✅ If the elevator is already **handling passengers**, check if another is available
                    if self.state['elevator_load'][i] > 0 and distance < min_distance:
                        closest_elevator = i
                        min_distance = distance

                # ✅ If the closest elevator is **serving passengers**, but another is available → "DYNAMIC-ASSIGN"
                if closest_elevator is not None:
                    for i in range(self.num_elevators):
                        if i != closest_elevator and self.state['elevator_load'][i] == 0:
                            return "DYNAMIC-ASSIGN"

        # ✅ Rush Mode (High Demand)
        if (
            total_waiting > rush_passenger_threshold or
            total_requests > rush_request_threshold or
            average_occupancy > rush_occupancy_threshold 
            # is_peak_hour 
            # total_waiting > rush_total_waiting_threshold  # ✅ NEW CONDITION ADDED
        ):
            return "RUSH"  # 🔴 Rush Mode

        # ✅ Normal Mode (Moderate Traffic)
        if (
            total_waiting > normal_passenger_threshold or
            total_requests > normal_request_threshold or
            average_occupancy > normal_occupancy_threshold
        ):
            return "NORMAL"  # 🟡 Normal Operation

        # ✅ Energy-Saving Mode (Low Demand)
        return "ENERGY-SAVING"  # 🟢 Energy-Saving Mode



    def handle_passenger_movement(self, elevator_index, new_position):
        """Handles passenger drop-off and pick-up at the elevator's new position."""

        # ✅ Track Wait Time when elevator reaches a floor with pending request
        if new_position in self.passenger_wait_times:
            self.track_wait_time(new_position, self.current_time)

        # ✅ Drop off passengers who reached their destination
        exited_passengers = [
            (p_id, dest) for p_id, dest in self.state['elevator_passengers'][elevator_index] if dest == new_position
        ]
        for passenger_id, _ in exited_passengers:
            self.track_service_time(passenger_id, self.current_time)

        # ✅ Remove dropped-off passengers
        self.state['elevator_passengers'][elevator_index] = [
            (p, d) for p, d in self.state['elevator_passengers'][elevator_index] if d != new_position
        ]
        self.state['elevator_load'][elevator_index] = len(self.state['elevator_passengers'][elevator_index])

        # ✅ Pick up passengers if space is available
        if self.state['elevator_load'][elevator_index] < self.max_capacity:
            # Prefer 'up' direction if available
            direction = 'up' if self.state['passengers_waiting'][new_position]['up'] else 'down'
            pickup_queue = self.state['passengers_waiting'][new_position][direction]

            filtered_queue = []

            for pid, dest in pickup_queue:
                is_vip = str(pid).startswith("VIP_")

                # ❌ Non-VIP elevator cannot pick VIP passengers
                if is_vip and elevator_index != self.vip_elevator_id:
                    continue

                # ❌ VIP elevator cannot pick non-VIP passengers
                if not is_vip and elevator_index == self.vip_elevator_id:
                    continue

                filtered_queue.append((pid, dest))

            # 🚪 Board eligible passengers
            for pid, dest in filtered_queue:
                if self.state['elevator_load'][elevator_index] >= self.max_capacity:
                    break

                pickup_queue.remove((pid, dest))  # Remove from queue
                self.passenger_board_times[pid] = self.current_time
                self.state['elevator_passengers'][elevator_index].append((pid, dest))
                self.state['elevator_load'][elevator_index] += 1

                print(f"🚪 Passenger {pid} ENTERED Elevator {elevator_index + 1} at Floor {new_position} (Going to Floor {dest})")

        # ✅ Clean up wait times if floor is now empty
        if not self.state['passengers_waiting'][new_position]['up'] and not self.state['passengers_waiting'][new_position]['down']:
            if new_position in self.passenger_wait_times:
                del self.passenger_wait_times[new_position]


    def dynamic_assign_routing(self, elevator_index):
        """Ensure only the closest available elevator moves to handle new requests in DYNAMIC-ASSIGN mode."""
        
        current_floor = self.state['elevator_positions'][elevator_index]

        # ✅ Find all waiting passengers
        waiting_floors = [
            floor for floor in range(1, self.num_floors + 1)
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']
        ]

        if not waiting_floors:
            return 0  # Stay idle if no passengers are waiting

        # ✅ Find the closest waiting request
        closest_request = min(waiting_floors, key=lambda floor: abs(floor - current_floor))
        closest_distance = abs(closest_request - current_floor)

        # ✅ Find the **closest idle elevator**
        best_elevator = None
        best_distance = float('inf')

        for i in range(self.num_elevators):
            if self.state['elevator_load'][i] == 0:  # ✅ Only consider idle elevators
                elevator_floor = self.state['elevator_positions'][i]
                distance = abs(closest_request - elevator_floor)

                # ✅ Assign the closest available elevator
                if distance < best_distance:
                    best_elevator = i
                    best_distance = distance

        # ✅ If this elevator is the best choice, move towards the request
        if best_elevator == elevator_index:
            return np.sign(closest_request - current_floor)  # Move towards the request

        return 0  # Stay idle if another elevator is closer



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
    

    # def energy_efficient_routing(self, elevator_index):
    #     """Implements Smart Routing for Energy Efficiency."""
    #     current_floor = self.state['elevator_positions'][elevator_index]

    #     # Find nearby requests within the assigned zone
    #     assigned_zone = (elevator_index % 3) + 1  # Assign zones dynamically
    #     zone_start = (assigned_zone - 1) * (self.num_floors // self.num_elevators) + 1
    #     zone_end = assigned_zone * (self.num_floors // self.num_elevators)

    #     for floor in range(zone_start, zone_end + 1):
    #         if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']:
    #             return np.sign(floor - current_floor)  # Move towards request

    #     return 0  # Stay idle if no requests
    
    def energy_efficient_routing(self, elevator_index):
        """Assign the closest available elevator to handle requests efficiently."""
        
        current_floor = self.state['elevator_positions'][elevator_index]

        # ✅ Find all waiting passengers
        waiting_floors = [
            floor for floor in range(1, self.num_floors + 1)
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']
        ]

        if not waiting_floors:
            return 0  # Stay idle if no passengers are waiting

        # ✅ Find the closest waiting request
        closest_request = min(waiting_floors, key=lambda floor: abs(floor - current_floor))
        closest_distance = abs(closest_request - current_floor)

        # ✅ Ensure the closest elevator is assigned to this request
        # Check if another elevator is **closer** to the request
        for i in range(self.num_elevators):
            if i != elevator_index:  # Skip checking itself
                other_elevator_floor = self.state['elevator_positions'][i]
                other_distance = abs(closest_request - other_elevator_floor)

                # **If another elevator is closer, this elevator should stay idle**
                if other_distance < closest_distance:
                    return 0  # ✅ Stay idle since another elevator is closer

        # ✅ If this elevator is the closest, move towards the request
        return np.sign(closest_request - current_floor)
    
    def energy_efficient_routing_best(self, elevator_index):
        """Ensure only one elevator operates in energy-saving mode, choosing the closest one."""
        
        current_floor = self.state['elevator_positions'][elevator_index]

        # ✅ Find all waiting passengers
        waiting_floors = [
            floor for floor in range(1, self.num_floors + 1)
            if self.state['passengers_waiting'][floor]['up'] or self.state['passengers_waiting'][floor]['down']
        ]

        if not waiting_floors:
            return 0  # Stay idle if no passengers are waiting

        # ✅ Find the closest waiting request
        closest_request = min(waiting_floors, key=lambda floor: abs(floor - current_floor))
        closest_distance = abs(closest_request - current_floor)

        # ✅ Ensure only **ONE elevator is assigned** in energy-saving mode
        # **Check if another elevator is already moving towards a request**
        active_elevator = None
        for i in range(self.num_elevators):
            if i != elevator_index and self.state['elevator_load'][i] > 0:
                active_elevator = i
                break  # ✅ Stop searching after finding one active elevator

        # ✅ If another elevator is already serving a request, this elevator stays idle
        if active_elevator is not None:
            return 0  # Stay idle

        # ✅ If no elevator is active, assign the closest one to move
        for i in range(self.num_elevators):
            if i != elevator_index:  # Skip itself
                other_elevator_floor = self.state['elevator_positions'][i]
                other_distance = abs(closest_request - other_elevator_floor)

                # **If another elevator is closer, this elevator stays idle**
                if other_distance < closest_distance:
                    return 0  # ✅ Stay idle since another elevator is closer

        # ✅ If this elevator is the best choice, move towards the request
        return np.sign(closest_request - current_floor)



    def update_passengers(self):
        """Fetch passenger data from MySQL and update the environment (avoids duplication)."""
        try:
            conn = connect_to_db()
            cursor = conn.cursor(dictionary=True)

            # ✅ Fetch passengers with time <= current simulation time
            cursor.execute("SELECT * FROM passengers WHERE STR_TO_DATE(time, '%I:%i:%S %p') <= %s", 
                        (self.current_time.strftime('%I:%M:%S %p'),))
            new_passengers = cursor.fetchall()
            conn.close()

            for row in new_passengers:
                passenger_id = row['passenger_id']
                
                # ✅ Skip if this passenger was already added
                if passenger_id in self.processed_passengers:
                    continue

                floor = int(row['floor'])
                direction = 'up' if row['direction'].strip().lower() == 'up' else 'down'
                destination = int(row['destination_floor'])

                # ✅ Add to waiting list
                self.state['passengers_waiting'][floor][direction].append((passenger_id, destination))
                self.processed_passengers.add(passenger_id)  # ✅ Mark as processed

                # ✅ Optional: track wait start time
                if floor not in self.passenger_wait_times:
                    self.passenger_wait_times[floor] = self.current_time

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
        self.energy_usage = []
        self.service_times = []
        self.wait_times = []
        self.vip_targets = []
        self.vip_elevator_id = None
        self.active_reservation_window = None
        self.passenger_board_times = {}
        self.passenger_wait_times = {}
        return self._get_observation()
    
            
    # def track_wait_time(self, floor, elevator_index, arrival_time):
    #     """Calculate wait time for a request when an elevator reaches the requested floor."""
        
    #     if floor in self.passenger_wait_times:
    #         request_time = self.passenger_wait_times[floor]

    #         # ✅ Corrected Calculation: Consider Time Taken for Elevator to Reach
    #         travel_time = abs(self.state['elevator_positions'][elevator_index] - floor) * self.time_per_floor
    #         wait_time = (arrival_time - request_time).total_seconds() + travel_time

    #         # ✅ Store wait time (for all elevators serving requests)
    #         self.wait_times.append(wait_time)

    #         # ✅ Remove the request from tracking
    #         del self.passenger_wait_times[floor]

    #         # 🔍 Debugging Output
    #         print(f"🟡 Elevator {elevator_index+1} reached Floor {floor} at {arrival_time} (Request appeared at {request_time})")
    #         print(f"⏳ Corrected Wait Time Recorded: {wait_time} seconds")
    
    def track_wait_time(self, floor, arrival_time):
        """Track wait time for all waiting passengers at a floor when an elevator arrives."""
        
        if floor in self.passenger_wait_times:
            request_time = self.passenger_wait_times[floor]

            # ✅ Compute the correct wait time
            wait_time = (arrival_time - request_time).total_seconds()

            # ✅ Store the wait time for every passenger still waiting
            self.wait_times.append(wait_time)

            # 🔍 Debugging Output
            print(f"🟡 Elevator reached Floor {floor} at {arrival_time} (Request appeared at {request_time})")
            print(f"⏳ Corrected Wait Time Recorded: {wait_time:.1f} seconds")

            # ✅ Only remove the request if all passengers at the floor have entered an elevator
            if not self.state['passengers_waiting'][floor]['up'] and not self.state['passengers_waiting'][floor]['down']:
                del self.passenger_wait_times[floor]




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
    
    def render_2d(self):
        """Render the Elevator Environment in Pygame with VIP Visualization."""

        # ✅ Handle Pygame Events to Prevent Freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # ✅ Define UI Layout
        header_height = 100
        total_height = self.screen_height + header_height
        self.screen = pygame.display.set_mode((self.screen_width, total_height))
        self.screen.fill((0, 0, 0))

        font = pygame.font.Font(None, 28)

        # ✅ Display Simulation Time
        time_text = font.render(f"Time: {self.current_time.strftime('%I:%M:%S %p')}", True, (255, 255, 255))
        self.screen.blit(time_text, (20, 10))

        # ✅ Determine Current Mode with Colors
        current_mode = self.detect_elevator_mode()
        mode_colors = {
            "RUSH": (255, 0, 0),
            "NORMAL": (255, 165, 0),
            "ENERGY-SAVING": (0, 255, 0),
            "VIP": (255, 105, 180),
            "PRESCHEDULE": (0, 191, 255)  # Deep Sky Blue
        }
        mode_color = mode_colors.get(current_mode, (255, 255, 255))
        mode_text = font.render(f"MODE: {current_mode}", True, mode_color)
        self.screen.blit(mode_text, (self.screen_width // 2 - 50, 10))
        
        # 🛠️ Maintenance elevator info (only if in MAINTENANCE mode)
        if current_mode == "MAINTENANCE" and self.maintenance_active:
            maint_label = font.render(f"🛠️ Elevator {self.maintenance_elevator_id} Under Maintenance", True, (192, 192, 192))
            self.screen.blit(maint_label, (self.screen_width // 2 - 100, 40))

        # ✅ VIP Status Display
        if self.vip_targets:
            vip_status = "Waiting" if not self.vip_targets[0]['picked_up'] else "In Elevator"
            vip_label = font.render(f"👑 VIP Status: {vip_status}", True, (255, 105, 180))
            self.screen.blit(vip_label, (self.screen_width // 2 - 50, 40))

        # ✅ Total Waiting Passengers
        total_waiting = sum(len(v['up']) + len(v['down']) for v in self.state['passengers_waiting'].values())
        waiting_text = font.render(f"Waiting Passengers: {total_waiting}", True, (255, 255, 0))
        self.screen.blit(waiting_text, (20, 40))

        # ✅ Stats
        total_energy = sum(self.energy_usage) if self.energy_usage else 0
        total_wait_time = sum(self.wait_times) if self.wait_times else 0
        total_service_time = sum(self.service_times) if self.service_times else 0

        stats_text = font.render(
            f"🚶 Wait: {total_wait_time:.1f}s | ⏳ Service: {total_service_time:.1f}s | ⚡ Energy: {total_energy}",
            True, (255, 255, 255)
        )
        self.screen.blit(stats_text, (20, 70))

        # ✅ Draw Floors
        floor_height = (self.screen_height - header_height) // self.num_floors
        for floor in range(1, self.num_floors + 1):
            y = total_height - floor * floor_height

            # Floor line
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_width, y), 2)

            # Floor label
            floor_label = font.render(f"Floor {floor}", True, (255, 255, 255))
            self.screen.blit(floor_label, (10, y + 5))
            
            # PRESCHEDULE marker
            if self.preschedule_active and self.preschedule_event:
                if self.preschedule_event["floor"] == floor:
                    preschedule_marker = font.render("🚀 PRESCHEDULE", True, (0, 191, 255))
                    self.screen.blit(preschedule_marker, (250, y + 5))

            # VIP marker on floor
            for vip in self.vip_targets:
                if vip['entry_floor'] == floor and not vip['picked_up']:
                    vip_marker = font.render("🎯 VIP", True, (255, 105, 180))
                    self.screen.blit(vip_marker, (150, y + 5))

            # Waiting passengers
            up_count = len(self.state['passengers_waiting'][floor]['up'])
            down_count = len(self.state['passengers_waiting'][floor]['down'])

            up_text = font.render(f"⬆ {up_count}", True, (255, 255, 0))
            down_text = font.render(f"⬇ {down_count}", True, (0, 255, 255))
            self.screen.blit(up_text, (self.screen_width - 100, y + 5))
            self.screen.blit(down_text, (self.screen_width - 50, y + 5))

        # ✅ Draw Elevators
        elevator_width = 50
        elevator_spacing = self.screen_width // (self.num_elevators + 1)
        for i, pos in enumerate(self.state['elevator_positions']):
            x = (i + 1) * elevator_spacing - (elevator_width // 2)
            y = total_height - pos * floor_height + 5

            # 🛠️ Maintenance elevator (takes highest priority)
            if self.maintenance_active and i == self.maintenance_elevator_id:
                color = (128, 128, 128)  # Gray
            elif self.vip_elevator_id == i:
                color = (255, 105, 180)  # VIP Pink
            elif self.preschedule_active and i in self.elevator_targets:
                color = (0, 191, 255)  # Preschedule Blue
            elif self.state['elevator_load'][i] > 0:
                color = (255, 0, 0)  # Red for passengers
            elif self.state['elevator_positions'][i] != 1:
                color = (0, 0, 255)  # Blue for moving
            else:
                color = (0, 255, 0)  # Green for idle
                
            pygame.draw.rect(self.screen, color, (x, y, elevator_width, floor_height - 10))

            # Passenger count in elevator
            label = font.render(f"{self.state['elevator_load'][i]}/10", True, (255, 255, 255))
            self.screen.blit(label, (x + 10, y + 20))

        pygame.display.update()
        self.clock.tick(30)
        
        
    def render_3d(self):
        if not hasattr(self, "initialized_3d"):
            
            # Before pygame.init()
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

            pygame.init()
            # pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
            pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)  # or 1920x1080
            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            gluPerspective(45, (800 / 600), 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)
            glTranslatef(0.0, -5.0, -40.0)

            self.elevator_positions_3d = [(i - self.num_elevators // 2) * 6 for i in range(self.num_elevators)]
            self.floor_spacing = 4
            self.initialized_3d = True

        # Handle window events to prevent freezing
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return  # Exit rendering gracefully if window is closed
            
        # ✅ Step 3: Add camera key controls here
        keys = pygame.key.get_pressed()
        if keys[K_w]: self.camera_distance -= 1
        if keys[K_s]: self.camera_distance += 1
        if keys[K_a]: self.camera_angle_y -= 2
        if keys[K_d]: self.camera_angle_y += 2
        if keys[K_q]: self.camera_angle_x += 1
        if keys[K_e]: self.camera_angle_x -= 1
        if keys[K_r]:
            self.camera_distance = 70
            self.camera_angle_x = 10
            self.camera_angle_y = 0
        
        # Clamp camera values
        self.camera_angle_x = max(-89, min(89, self.camera_angle_x))
        self.camera_distance = max(5, min(100, self.camera_distance))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glLoadIdentity()
        gluLookAt(
            self.camera_distance * sin(radians(self.camera_angle_y)),
            self.camera_angle_x,
            self.camera_distance * cos(radians(self.camera_angle_y)),
            0, 10, 0,  # Look-at target (center of scene)
            0, 1, 0    # Up direction
        )
                

        self._draw_floors_3d()
        
        # ✅ Show Floor-Level Markers: 🚀 Preschedule + 👑 VIP
        for floor_number in range(1, self.num_floors + 1):
            y = (floor_number - 1) * self.floor_spacing

            # 🚀 PRESCHEDULE marker
            if self.preschedule_active and self.preschedule_event:
                if self.preschedule_event["floor"] == floor_number:
                    self._draw_text_3d("🚀", 0, y + 2, 6)

            # 👑 VIP marker
            for vip in self.vip_targets:
                if vip['entry_floor'] == floor_number and not vip['picked_up']:
                    self._draw_text_3d("👑", 0, y + 2, -6)
                    
            # ✅ Passenger Waiting Counts
            up_count = len(self.state['passengers_waiting'][floor_number]['up'])
            down_count = len(self.state['passengers_waiting'][floor_number]['down'])

            # Always show Up and Down counts — even if 0
            up_count = len(self.state['passengers_waiting'][floor_number]['up'])
            down_count = len(self.state['passengers_waiting'][floor_number]['down'])

            # Position on right side of building
            up_text = f"Up: {up_count}"
            down_text = f"Down: {down_count}"

            # Display both on the same line with color
            self._draw_text_3d(up_text, 11, y + 0.2, -5, color=(1.0, 1.0, 0.0))   # Yellow
            self._draw_text_3d(down_text, 17, y + 0.2, -5, color=(0.0, 1.0, 1.0)) # Cyan


        mode = self.detect_elevator_mode()
        
        # ✅ Draw simulation status text in 3D (top-left corner of scene)
        total_energy = sum(self.energy_usage) if self.energy_usage else 0
        total_wait_time = sum(self.wait_times) if self.wait_times else 0
        total_service_time = sum(self.service_times) if self.service_times else 0

        # 🏗️ Position above top floor and centered in front
        top_y = self.num_floors * self.floor_spacing + 4 # +3 units above top floor
        x_center = -6                                     # Adjust left/right centering
        z_offset = 0                                      # Slightly in front (optional)

        # # 🕒 Time
        # self._draw_text_3d(f"Time: {self.current_time.strftime('%I:%M:%S %p')}", x_center, top_y, z_offset)

        # # 🔁 Mode
        # self._draw_text_3d(f"Mode: {mode}", x_center, top_y - 1.2, z_offset)
        
        self._draw_text_3d(f"Time: {self.current_time.strftime('%I:%M:%S %p')}", -6, top_y, 0)
        self._draw_text_3d(f"Mode: {mode}", -6, top_y - 1.6, 0)

        # ⏱️ Wait, Service, Energy
        total_wait_time = sum(self.wait_times) if self.wait_times else 0
        total_service_time = sum(self.service_times) if self.service_times else 0
        total_energy = sum(self.energy_usage)

        # self._draw_text_3d(
        #     f"Wait: {total_wait_time:.0f}s | Service: {total_service_time:.0f}s | Energy: {total_energy}",
        #     x_center, top_y - 2.4, z_offset
        # )
        self._draw_text_3d(
            f"Wait: {total_wait_time:.0f}s | Service: {total_service_time:.0f}s | Energy: {total_energy}",
            -6, top_y - 3.2, 0
)


        for i in range(self.num_elevators):
            x = self.elevator_positions_3d[i]
            elevator_height = 4
            y = (self.state['elevator_positions'][i] - 1) * self.floor_spacing + (elevator_height / 2)

            # ✅ Match color logic from render_2d()
            if self.maintenance_active and i == self.maintenance_elevator_id:
                color = (0.5, 0.5, 0.5)  # Gray
            elif self.vip_elevator_id == i:
                color = (1.0, 0.41, 0.71)  # VIP Pink (255,105,180)
            elif self.preschedule_active and i in self.elevator_targets:
                color = (0.0, 0.75, 1.0)  # Preschedule Blue (0,191,255)
            elif self.state['elevator_load'][i] > 0:
                color = (1.0, 0.0, 0.0)  # Red for passengers
            elif self.state['elevator_positions'][i] != 1:
                color = (0.0, 0.0, 1.0)  # Blue for moving
            else:
                color = (0.0, 1.0, 0.0)  # Green for idle

            self._draw_elevator_3d(x, y, 0, color)
            self._draw_text_3d(f"{self.state['elevator_load'][i]}/10", x - 1.5, y + 2.5, 0)


        pygame.display.flip()
        
    def _draw_text_3d(self, text, x, y, z, font=None, color=(1.0, 1.0, 1.0)):


        if font is None:
            font = GLUT_BITMAP_HELVETICA_18

        glColor3f(*color)
        glRasterPos3f(x, y, z)
        for ch in text:
            glutBitmapCharacter(font, ord(ch))


        
    def _draw_floors_3d(self):
        for i in range(self.num_floors):
            y = i * self.floor_spacing

            glColor3f(0.5, 0.5, 0.5)  # ✅ Explicitly set gray before drawing each slab

            glBegin(GL_QUADS)
            glVertex3f(-10, y, -10)
            glVertex3f(10, y, -10)
            glVertex3f(10, y, 10)
            glVertex3f(-10, y, 10)
            glEnd()

            # ✅ Floor label — now stays white
            # self._draw_text_3d(f"Floor {i+1}", -9, y + 0.2, 9)
            self._draw_text_3d(f"Floor {i+1}", -20, y + 0.2, 0)  # Farther to the left



            
    def _draw_elevator_3d(self, x, y, z, color=(0, 1, 0)):

        w, h, d = 3, 4, 3
        w /= 2
        h /= 2
        d /= 2

        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        glBegin(GL_QUADS)

        # Faces
        faces = [
            ((-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d)),  # Front
            ((-w, -h, -d), (-w, h, -d), (w, h, -d), (w, -h, -d)),  # Back
            ((-w, -h, -d), (-w, -h, d), (-w, h, d), (-w, h, -d)),  # Left
            ((w, -h, -d), (w, h, -d), (w, h, d), (w, -h, d)),  # Right
            ((-w, h, -d), (-w, h, d), (w, h, d), (w, h, -d)),  # Top
            ((-w, -h, -d), (w, -h, -d), (w, -h, d), (-w, -h, d))  # Bottom
        ]
        for face in faces:
            for v in face:
                glVertex3f(*v)

        glEnd()
        glPopMatrix()

    def close(self):    
        pygame.quit()
