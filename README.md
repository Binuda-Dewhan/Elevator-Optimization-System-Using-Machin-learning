# 🚀 Smart Elevator System Simulator

An intelligent, machine-learning-powered elevator traffic simulator designed for high-rise buildings such as universities and commercial complexes. This system focuses on **dynamic traffic prediction**, **adaptive scheduling**, and **personalized user prioritization** to reduce waiting time, optimize energy usage, and improve the overall elevator experience.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Screenshots](#️-screenshots)
3. [Key Features](#-key-features)
4. [Technologies Used](#-technologies-used)
5. [Folder Structure](#-folder-structure)
6. [How to Run the Project](#-how-to-run-the-project)
7. [Elevator Route Planning Logic](#-elevator-route-planning-logic)
8. [Integration of Components](#-integration-of-components)
9. [Datasets Generated](#-datasets-generated)
10. [License](#-license)
11. [Demo & Contact](#-demo-contact)

---

## 🎯 Project Overview

This project simulates a **real-world smart elevator system** by generating realistic passenger traffic based on class schedules, floor activity, and time-based trends. It supports multiple modules including:
- **Simulator UI** for generating synthetic elevator traffic
- **Machine Learning models** for demand prediction
- **Reinforcement Learning** for personalized prioritization
- **Face recognition** for live user identification
- **Dashboard** for monitoring and anomaly alerts
- **Mobile application** for reservations and feedback

The final system optimizes ride efficiency, reduces wait time, and makes elevator usage smarter and more energy-efficient.

---

## 🖼️ Screenshots

Below are some visual highlights of the simulator in action:

### 🛗 Elevator Simulation UI  
<img width="627" height="375" alt="Screenshot 2025-08-03 155205" src="https://github.com/user-attachments/assets/05c76140-91e0-4d23-944a-91f90a18200b" />

<img width="800" height="600" alt="Screenshot 2025-08-03 155220" src="https://github.com/user-attachments/assets/a865e8d9-28e8-4a37-b81f-85c2fb521413" />


### 📊 Real-Time Performance Metrics  
<img width="720" height="442" alt="Screenshot 2025-05-26 010208" src="https://github.com/user-attachments/assets/4f52be68-58a3-4af7-bcec-c202561943f1" />

<img width="800" height="600" alt="Screenshot 2025-05-27 014652" src="https://github.com/user-attachments/assets/60b17ab1-450e-40f5-8ce7-8ecca7b9ede6" />


### 🧍‍♂️ Live Passenger Detail Generation  
<img width="801" height="494" alt="Screenshot 2024-12-04 194453" src="https://github.com/user-attachments/assets/01624978-28f5-47cb-b39c-57cb256b5fdc" />

---

## 🧠 Key Features

- ⏱️ **Dynamic Traffic Prediction** using LSTM + Attention
- 🧍‍♂️ **User Prioritization** with Actor-Critic RL model
- 🗓️ **Class Schedule Integration** to simulate realistic university traffic
- 📊 **Simulation per 3-minute slots** between 6:00 AM – 9:00 PM
- 🛗 **6-Elevator System** with 10-passenger capacity each
- 👁️ **Face Recognition Module** with reservation fetch
- ⚙️ **Elevator Pre-Scheduling** using prediction model
- 🔋 **Energy Optimization** by toggling elevator idle/active state
- 📈 **Real-time Analytics Dashboard** for monitoring
- 📱 **Flutter App** for personalized user reservations and feedback

---

## ⚙️ Technologies Used
- Main technologies
  
| Category             | Technology                            |
|----------------------|----------------------------------------|
| Simulation           | Python, Pandas, Tkinter, NumPy, Pygame |
| ML Models            | Keras/TensorFlow (LSTM + Attention)    |
| RL Models            | PyTorch (Actor-Critic, PER)            |
| Face Recognition     | OpenCV, DeepFace, RCNN                 |
| Mobile App           | Flutter, Firebase                      |
| Database             | Firebase Realtime DB, AWS S3 (MinIO)   |
| Analytics Dashboard  | Power BI, Node-RED                     |

---

## 📁 Folder Structure
```
├── Dataset Creation/              # Data simulation from schedules
├── Comparing dataset/            # Evaluation datasets for models
├── gui.py                        # Main entry point (Tkinter + Pygame simulator)
├── simulator.py                  # Rule-based elevator simulation engine
├── ENV.py                        # Environment logic and traffic logic
├── ENVsql.py                     # MySQL-linked data ingestion
├── YOLO.py                       # People detection using YOLOv8
├── yolov8n.pt                    # YOLOv8 model weights
├── passengers_01.csv             # Sample synthetic passenger data
├── logo.png                      # App or UI logo
├── gui.spec                      # PyInstaller build file
├── eleTest.py                    # Mini test script for elevators
├── .gitignore                    # Git exclusions

```

---

## 🧪 How to Run the Project
Pre-requisites: Python 3.10, Conda or virtualenv, Firebase key (not public)

### 🔹 Step 1: Create and activate the environment
```
conda create -n elevator_sim_env python=3.10 -y
conda activate elevator_sim_env
```

### 🔹 Step 2: Install dependencies
```
pip install -r requirements.txt
requirements.txt includes: pandas, numpy, pygame, firebase-admin, matplotlib, pillow, gym, opencv-python, torch, yolov8, etc.
```

### 🔹 Step 3: Set up Firebase key (not in repo)
```
# set the environment variable before running
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service.json"  # macOS/Linux
set GOOGLE_APPLICATION_CREDENTIALS=path\to\service.json       # Windows
```

### 🔹 Step 4: Run the simulator
```
python gui.py
```
- The simulator window will open with live traffic simulation, elevator movement, and performance metrics.

---
## 🧠 Elevator Route Planning Logic

This component handles the real-time decision-making and scheduling of multiple elevators within the simulation environment. The logic ensures optimized elevator movement and reduced passenger wait times based on observed traffic patterns and environmental conditions.

### 🔹 System Behavior Highlights:
- Each elevator dynamically selects its actions based on current demand, passenger queues, and occupancy.
- The system adapts to various traffic intensities (e.g., peak hours, low activity periods).
- It intelligently distributes elevator duties to maintain service efficiency and energy savings.


### 🔹 Modes Defined:
- Normal Mode: Default scheduling under moderate traffic

- Rush Mode: Triggered during high traffic; elevators assigned dynamically based on floor queues

- Energy-Saving Mode: During low usage hours, only one elevator remains active

- Dynamic-Assign Mode: Balances load across elevators in real time based on queue stats

### 🔹 Simulation Architecture:
- Custom `ElevatorEnv` modeled in OpenAI Gym-style framework
- Visualized in real-time using a combination of `Tkinter` and `Pygame`
- Core metrics tracked during simulation:
  - **Average Waiting Time**
  - **Service Time**
  - **Energy Consumption**

---

## 🔗 Integration of Components

The simulator acts as the central hub, integrating all smart modules through Firebase's real-time API. Each external component updates relevant data to Firebase Firestore, which the simulator polls or listens to in real time.

### 🔹 Integration Architecture

- **Mobile App**  
  Users can make elevator reservations, submit feedback, or track status — all stored in Firebase. The simulator reads this to adjust elevator scheduling.

- **Face Recognition Module**  
  Identifies users and updates prioritization or reservation triggers in Firebase. The simulator adapts behavior accordingly.

- **Traffic Prediction (LSTM)**  
  Predicts floor-wise demand and schedules. The output is pushed to Firebase and influences elevator pre-scheduling logic.

- **Predictive Maintenance (LSTM + IoT Sensors)**  
  Faults or maintenance alerts are stored in Firebase. The simulator checks for these and disables or reroutes affected elevators.

### 🔹 Real-Time Behavior

At every simulation tick, the environment checks Firebase for:

- 🔧 **Maintenance Events**
- 🧑‍🏫 **Faculty/Student Prioritization**
- 🛗 **Pre-Scheduled Traffic Predictions**
- 📱 **User Reservation Triggers**

Based on this input, the elevator environment dynamically changes its route plan and mode selection — allowing realistic coordination with all modules.

This unified integration enables a comprehensive smart elevator system that mirrors real-world responsiveness and complexity.


---

## 📊 Datasets Generated

This project generates a variety of datasets during simulation and traffic detection phases:

- **Passenger Logs**  
  Synthesized per 3-minute time slot using class schedules to emulate realistic elevator demand.

- **YOLOv8 Detection Logs**  
  Captures real-time people count per floor using video-based detection (via YOLOv8).

- **Elevator State Logs**  
  Records elevator direction, current floor, door status, and system-assigned activity state.

- **Performance Metrics**  
  Computed after each simulation loop to evaluate:
  - Average wait time
  - Service time
  - Energy consumption per elevator movement
 
---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, adapt, or contribute under open-source terms.

---

## 📽️ Demo & Contact

For more information, academic reference, or collaboration inquiries:

- 👤 **Name**: Binuda Dewhan Bandara  
- 📧 **Email**: binudab4@gmail.com  
- 📱 **Phone**: +94 771168665  
- 📄 **Research Paper**: See `/docs` directory or contact the author



---
