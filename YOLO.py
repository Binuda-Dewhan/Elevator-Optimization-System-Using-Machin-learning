
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime, timedelta
import random
import time
import os
import threading

import mysql.connector

# ✅ Database Connection Setup
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",   # Change this if MySQL is hosted elsewhere
        user="root",        # Your MySQL username (default is 'root')
        password="root",  # Change this to your MySQL password
        database="elevator_system"  # Database name
    )
    
def clear_passenger_table():
    """Delete all records from the passengers table before inserting new data."""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM passengers")  # Deletes all rows from the table
        conn.commit()
        conn.close()

        print("🗑️ All previous passenger data deleted successfully.")

    except Exception as e:
        print(f"❌ Error deleting data from MySQL: {e}")


def insert_data_into_db(passenger_data):
    conn = connect_to_db()
    cursor = conn.cursor()

    sql = '''
        INSERT INTO passengers (passenger_id, time, floor, direction, destination_floor) 
        VALUES (%s, %s, %s, %s, %s)
    '''
    values = (
        passenger_data["Passenger ID"], 
        passenger_data["Time"], 
        passenger_data["Floor"], 
        passenger_data["Direction (Up/Down)"], 
        passenger_data["Destination Floor"]
    )
    
    cursor.execute(sql, values)
    conn.commit()
    conn.close()




# ✅ Load YOLOv8 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# ✅ Video Paths for Each Floor
video_paths = {
    1: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/2 floor.mp4",
    2: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/3 floor.mp4",
    3: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/4 floor.mp4",
    4: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/5 floor.mp4",
    5: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/6 floor.mp4",
    6: "C:/Users/Binuda Dewhan/Desktop/SCAN +/New folder/7 floor.mp4"
}

# ✅ CSV File for Passenger Data
csv_file = os.path.abspath("C:/Users/Binuda Dewhan/Desktop/SCAN +/passenger_data.csv")
passenger_id = 1  # Unique passenger ID counter
sync_lock = threading.Lock()  # Lock for synchronization

# ✅ Step 1: Check if CSV file exists → DELETE it if it does
if os.path.exists(csv_file):
    os.remove(csv_file)  # ✅ Delete old CSV file

# ✅ Step 2: Create a fresh new CSV with headers
df = pd.DataFrame(columns=["Passenger ID", "Time", "Floor", "Direction (Up/Down)", "Destination Floor"])
df.to_csv(csv_file, index=False)  # ✅ Save new CSV file with column headers

print(f"📁 New CSV file created: {csv_file}")

# ✅ Start Time from 08:00:00 AM
simulated_time = datetime.strptime("08:00:00 AM", "%I:%M:%S %p")
start_real_time = time.time()  # Track real-world time
video_time_sec = 0  # Start video at 0:00
interval = 10 # ✅ Change interval from 30s to 20s

# ✅ Function to Process Video for Each Floor
def process_video(video_path, floor_number):
    global passenger_id, simulated_time, video_time_sec
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video file {video_path}")
        return

    print(f"🚀 Processing Video for Floor {floor_number}...")

    while cap.isOpened():
        with sync_lock:  # Ensure time updates correctly across threads
            real_elapsed_time = int(time.time() - start_real_time)
            video_time_sec = (real_elapsed_time // interval) * interval  # ✅ Align with 20s intervals
            cap.set(cv2.CAP_PROP_POS_MSEC, video_time_sec * 1000)  # Move video to correct time

            # ✅ Sync Simulated Time with Video Time
            simulated_time = datetime.strptime("08:00:00 AM", "%I:%M:%S %p") + timedelta(seconds=video_time_sec)

        ret, frame = cap.read()
        if not ret:
            print(f"❌ ERROR: Could not read frame from {video_path} (End of Video)")
            break  # Stop if video ends

        # ✅ Run YOLOv8 Object Detection
        results = model(frame)

        # ✅ Count Passengers
        person_count = sum(1 for r in results for i in range(len(r.boxes.cls))
                           if r.names[int(r.boxes.cls[i].item())] == "person")

        new_passengers = []
        if person_count > 0:
            for _ in range(person_count):
                destination_floor = random.choice([i for i in range(1, 7) if i != floor_number])
                direction = "Up" if destination_floor > floor_number else "Down"

                passenger_data = {
                    "Passenger ID": passenger_id,
                    "Time": simulated_time.strftime("%I:%M:%S %p"),
                    "Floor": floor_number,
                    "Direction (Up/Down)": direction,
                    "Destination Floor": destination_floor
                }
                new_passengers.append(passenger_data)
                passenger_id += 1  # Increment Passenger ID

        # ✅ Store Data in CSV with Lock
        # ✅ Store Data in CSV with Lock
        # with sync_lock:
        #     if new_passengers:
        #         df = pd.DataFrame(new_passengers)
        #         with open(csv_file, mode='a', newline='') as f:
        #             df.to_csv(f, index=False, header=False)
        #             f.flush()  # ✅ Ensure data is written
        #             os.fsync(f.fileno())  # ✅ Force write to disk
        
        with sync_lock:
            if new_passengers:
                for passenger in new_passengers:
                    insert_data_into_db(passenger)


        # ✅ Print Debugging Information (Formatted Output)
        print(f"\n🕒 {simulated_time.strftime('%I:%M:%S %p')} | 🎥 Video Time: {video_time_sec//60}:{video_time_sec%60:02d}")
        for passenger in new_passengers:
            print(f"  🚶 Passenger {passenger['Passenger ID']} | Floor {passenger['Floor']} → Destination {passenger['Destination Floor']} | Direction: {passenger['Direction (Up/Down)']}")

        time.sleep(interval)  # ✅ Now waits 20 seconds instead of 30

    cap.release()
    print(f"✅ Finished Processing for Floor {floor_number}")

# ✅ Run YOLO for All Floors in Parallel
def run_yolo():
    
    # ✅ Step 1: Clear old passenger data before inserting new records
    clear_passenger_table()
    
    threads = []
    for floor, path in video_paths.items():
        t = threading.Thread(target=process_video, args=(path, floor))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

# ✅ Start YOLO in a Separate Thread
if __name__ == "__main__":
    run_yolo()
