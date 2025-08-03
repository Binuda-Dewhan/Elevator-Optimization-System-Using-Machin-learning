import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
from simulator import ElevatorEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from PIL import Image, ImageTk

def show_splash_screen(image_path=r"C:\Users\Binuda Dewhan\Desktop\elevator app\logo.png", delay=3000):
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.configure(bg="white")
    splash.geometry("500x300+500+250")  # Position and size (centered)

    try:
        img = Image.open(image_path)
        img = img.resize((300, 150), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        label = tk.Label(splash, image=logo, bg="white")
        label.image = logo  # prevent garbage collection
        label.pack(pady=20)
    except Exception as e:
        print(f"Error loading splash image: {e}")
        tk.Label(splash, text="Smart Elevator Simulator", font=("Arial", 20, "bold")).pack(pady=60)

    tk.Label(splash, text="Launching Smart Elevator Simulator...", font=("Arial", 12), bg="white").pack(pady=10)
    splash.after(delay, splash.destroy)
    splash.mainloop()



class ElevatorSimulatorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Elevator Simulator")
        self.master.geometry("1000x700")
        self.master.minsize(900, 600)
        self.csv_file = None
        self.render_mode = tk.StringVar(value="3D")
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        # Graph selection flags
        self.show_energy = tk.BooleanVar(value=True)
        self.show_wait_time = tk.BooleanVar(value=True)
        self.show_service_time = tk.BooleanVar(value=True)
        self.show_waiting_passengers = tk.BooleanVar(value=True)
        
        self.is_paused = threading.Event()
        self.is_paused.set()  # Initially not paused

        self.simulation_running = False  # To track simulation state



        self._build_ui()

    def _build_ui(self):
        padding = {'padx': 10, 'pady': 8}
        label_font = ("Arial", 10, "bold")

        # Select Simulator Type
        tk.Label(self.master, text="Select Simulator Type:", font=label_font).grid(row=0, column=0, sticky="w", **padding)
        sim_btn_frame = tk.Frame(self.master)
        sim_btn_frame.grid(row=0, column=1, columnspan=2, sticky="w", **padding)

        self.sim_type = tk.StringVar(value="CSV")
        self.csv_btn = tk.Button(sim_btn_frame, text="CSV-based", width=15, relief="sunken",
                                command=lambda: self.set_sim_type("CSV"))
        self.sql_btn = tk.Button(sim_btn_frame, text="YOLO + SQL-based", width=15, relief="raised",
                                command=lambda: self.set_sim_type("SQL"))
        self.csv_btn.pack(side=tk.LEFT, padx=5)
        self.sql_btn.pack(side=tk.LEFT, padx=5)

        # CSV File Selector
        tk.Label(self.master, text="CSV File:", font=label_font).grid(row=1, column=0, sticky="w", **padding)
        csv_frame = tk.Frame(self.master)
        csv_frame.grid(row=1, column=1, columnspan=2, sticky="w", **padding)
        self.csv_entry = tk.Entry(csv_frame, width=50)
        self.csv_entry.grid(row=0, column=0, padx=(0, 5))
        tk.Button(csv_frame, text="Browse", command=self.browse_file).grid(row=0, column=1)

        # Start Time
        tk.Label(self.master, text="Start Time:", font=label_font).grid(row=2, column=0, sticky="w", **padding)
        time_frame1 = tk.Frame(self.master)
        time_frame1.grid(row=2, column=1, sticky="w")
        self.start_hour = ttk.Combobox(time_frame1, values=[f"{i:02}" for i in range(1, 13)], width=3)
        self.start_min = ttk.Combobox(time_frame1, values=[f"{i:02}" for i in range(0, 60)], width=3)
        self.start_ampm = ttk.Combobox(time_frame1, values=["AM", "PM"], width=3)
        self.start_hour.grid(row=0, column=0)
        tk.Label(time_frame1, text=":").grid(row=0, column=1)
        self.start_min.grid(row=0, column=2)
        self.start_ampm.grid(row=0, column=3)

        # End Time
        tk.Label(self.master, text="End Time:", font=label_font).grid(row=3, column=0, sticky="w", **padding)
        time_frame2 = tk.Frame(self.master)
        time_frame2.grid(row=3, column=1, sticky="w")
        self.end_hour = ttk.Combobox(time_frame2, values=[f"{i:02}" for i in range(1, 13)], width=3)
        self.end_min = ttk.Combobox(time_frame2, values=[f"{i:02}" for i in range(0, 60)], width=3)
        self.end_ampm = ttk.Combobox(time_frame2, values=["AM", "PM"], width=3)
        self.end_hour.grid(row=0, column=0)
        tk.Label(time_frame2, text=":").grid(row=0, column=1)
        self.end_min.grid(row=0, column=2)
        self.end_ampm.grid(row=0, column=3)

        # Render Mode as Toggle Buttons
        tk.Label(self.master, text="Render Mode:", font=label_font).grid(row=4, column=0, sticky="w", **padding)
        self.render_mode = tk.StringVar(value="2D")
        render_frame = tk.Frame(self.master)
        render_frame.grid(row=4, column=1, columnspan=2, sticky="w", **padding)
        self.render_2d_btn = tk.Button(render_frame, text="2D", width=10, relief="sunken",
                                    command=lambda: self.set_render_mode("2D"))
        self.render_3d_btn = tk.Button(render_frame, text="3D", width=10, relief="raised",
                                    command=lambda: self.set_render_mode("3D"))
        self.render_2d_btn.pack(side=tk.LEFT, padx=5)
        self.render_3d_btn.pack(side=tk.LEFT, padx=5)

        # Graph Display Options
        tk.Label(self.master, text="Select Graphs to Display:", font=label_font).grid(row=5, column=0, sticky="w", **padding)
        checkbox_frame = tk.Frame(self.master)
        checkbox_frame.grid(row=5, column=1, columnspan=2, sticky="w", padx=5)
        tk.Checkbutton(checkbox_frame, text="Energy ⚡", variable=self.show_energy).pack(side=tk.LEFT)
        tk.Checkbutton(checkbox_frame, text="Wait Time 🚶", variable=self.show_wait_time).pack(side=tk.LEFT)
        tk.Checkbutton(checkbox_frame, text="Service Time ⏳", variable=self.show_service_time).pack(side=tk.LEFT)
        tk.Checkbutton(checkbox_frame, text="Waiting 👥", variable=self.show_waiting_passengers).pack(side=tk.LEFT)

        # # Run Button
        # self.run_button = tk.Button(self.master, text="Run Simulation", command=self.run_simulation_thread)
        # self.run_button.grid(row=6, column=1, sticky="w", **padding)
        
        # # Pause and Resume Buttons
        # self.pause_button = tk.Button(self.master, text="Pause", command=self.pause_simulation, state="disabled")
        # self.pause_button.grid(row=6, column=2, sticky="w", padx=5)

        # self.resume_button = tk.Button(self.master, text="Resume", command=self.resume_simulation, state="disabled")
        # self.resume_button.grid(row=6, column=2, sticky="e", padx=5)
        
        # Run / Pause / Resume Buttons
        button_frame = tk.Frame(self.master)
        button_frame.grid(row=6, column=1, columnspan=2, sticky="w", padx=10, pady=8)

        self.run_button = tk.Button(button_frame, text="Run Simulation", command=self.run_simulation_thread)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_button = tk.Button(button_frame, text="Pause", command=self.pause_simulation, state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))

        self.resume_button = tk.Button(button_frame, text="Resume", command=self.resume_simulation, state="disabled")
        self.resume_button.pack(side=tk.LEFT)
        
        # 🔍 Data Preview Buttons (Reservation / Pre-Schedule / Maintenance)
        preview_frame = tk.Frame(self.master)
        preview_frame.grid(row=6, column=2, sticky="e", padx=10)

        tk.Button(preview_frame, text="View Reservations", command=self.show_reservations).pack(side=tk.LEFT, padx=2)
        tk.Button(preview_frame, text="View Pre-Schedule", command=self.show_preschedule).pack(side=tk.LEFT, padx=2)
        tk.Button(preview_frame, text="View Maintenance", command=self.show_maintenance).pack(side=tk.LEFT, padx=2)

        # Graph Frame
        self.graph_frame = ttk.LabelFrame(self.master, text="Live Metrics", padding=10)
        self.graph_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        self.master.grid_rowconfigure(7, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # Data tracking
        self.time_points = []
        self.energy_data = []
        self.wait_time_data = []
        self.service_time_data = []
        self.waiting_passenger_data = []

        
    def set_sim_type(self, sim_type):
        self.sim_type.set(sim_type)
        if sim_type == "CSV":
            self.csv_btn.config(relief="sunken")
            self.sql_btn.config(relief="raised")
        else:
            self.csv_btn.config(relief="raised")
            self.sql_btn.config(relief="sunken")
            
    


    def set_render_mode(self, mode):
        self.render_mode.set(mode)
        if mode == "2D":
            self.render_2d_btn.config(relief="sunken")
            self.render_3d_btn.config(relief="raised")
        else:
            self.render_2d_btn.config(relief="raised")
            self.render_3d_btn.config(relief="sunken")

    def pause_simulation(self):
        self.is_paused.clear()
        self.pause_button.config(state="disabled")
        self.resume_button.config(state="normal")

    def resume_simulation(self):
        self.is_paused.set()
        self.resume_button.config(state="disabled")
        self.pause_button.config(state="normal")



    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.csv_file = file_path
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, file_path)
            
    def show_reservations(self):
        print("🔍 show_reservations() called")  # Add this
        env = ElevatorEnv()
        reservations = env.fetch_reservations()

        win = tk.Toplevel(self.master)
        win.title("Reservation List")
        win.geometry("400x300")

        text = tk.Text(win, wrap="word")
        text.pack(fill=tk.BOTH, expand=True)

        if reservations:
            for uid, res in reservations.items():
                text.insert(tk.END, f"User UID: {uid}\n")
                text.insert(tk.END, f"  Entry Floor: {res.get('entryFloor')}\n")
                text.insert(tk.END, f"  Destination Floor: {res.get('destinationFloor')}\n")
                text.insert(tk.END, f"  Time: {res.get('time')}\n")
                text.insert(tk.END, f"  Number of People: {res.get('numberOfPeople')}\n")
                text.insert(tk.END, f"  Urgency Level: {res.get('urgencyLevel')}\n")
                text.insert(tk.END, "-" * 40 + "\n")
        else:
            text.insert(tk.END, "⚠️ No reservation data found.\n")


    def show_preschedule(self):
        env = ElevatorEnv()
        predictions = env.fetch_peak_demand_data_from_firestore()

        win = tk.Toplevel(self.master)
        win.title("Pre-Schedule List")
        win.geometry("400x300")

        text = tk.Text(win, wrap="word")
        text.pack(fill=tk.BOTH, expand=True)

        if predictions:
            for ts, data in predictions.items():
                text.insert(tk.END, f"⏱️ Timestamp: {ts}\n")
                text.insert(tk.END, f"  Time Only: {data.get('time_only')}\n")
                text.insert(tk.END, f"  Floor: {data.get('floor')}\n")
                text.insert(tk.END, f"  Elevators Required: {data.get('num_elevators')}\n")
                text.insert(tk.END, "-" * 40 + "\n")
        else:
            text.insert(tk.END, "⚠️ No traffic prediction data found.\n")


    def show_maintenance(self):
        env = ElevatorEnv()
        data = env.fetch_maintenance_schedule()

        win = tk.Toplevel(self.master)
        win.title("Maintenance List")
        win.geometry("400x300")

        text = tk.Text(win, wrap="word")
        text.pack(fill=tk.BOTH, expand=True)

        if data:
            for ts, info in data.items():
                text.insert(tk.END, f"⏱️ Scheduled at: {ts}\n")
                text.insert(tk.END, f"  Date: {info.get('date')}\n")
                text.insert(tk.END, f"  Elevator ID: {info.get('elevator_id')} | Active: {info.get('active')}\n")
                text.insert(tk.END, f"  Original Time: {info.get('raw_time')}\n")
                text.insert(tk.END, "-" * 40 + "\n")
        else:
            text.insert(tk.END, "⚠️ No maintenance data found.\n")
            
    

    def run_simulation_thread(self):
        if self.sim_type.get() == "CSV":
            thread = threading.Thread(target=self.run_csv_simulation)
            thread.start()
            self.pause_button.config(state="normal")
            self.resume_button.config(state="disabled")
            self.is_paused.set()  # Allow it to run
            self.simulation_running = True

        else:
            self.master.destroy()  # Close the GUI first
            self.run_sql_yolo_simulation()
            
    def run_sql_yolo_simulation(self):
        import subprocess
        print("🚀 Launching YOLO + MySQL Simulator...")

        try:
            subprocess.run(["python", "eleTest.py"], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run SQL-based simulation.\n{e}")


    def run_csv_simulation(self):
        
        # Clear graph data
        self.time_points.clear()
        self.energy_data.clear()
        self.wait_time_data.clear()
        self.service_time_data.clear()
        self.waiting_passenger_data.clear()
        
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Create plot only when simulation starts
        self.fig, self.ax = plt.subplots(figsize=(3, 2.5))  # smaller height
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # 🔥 Stretchable canvas
        # self.canvas.get_tk_widget().pack(padx=10, pady=10, anchor='center')
        self.fig.autofmt_xdate()  # Auto format x-axis time labels

        
        try:
            if not self.csv_file:
                messagebox.showerror("Error", "Please select a CSV file.")
                return

            start_time = pd.to_datetime(f"{self.start_hour.get()}:{self.start_min.get()} {self.start_ampm.get()}", format="%I:%M %p")
            end_time = pd.to_datetime(f"{self.end_hour.get()}:{self.end_min.get()} {self.end_ampm.get()}", format="%I:%M %p")


            env = ElevatorEnv(csv_file=self.csv_file)
            env.passenger_data = env.passenger_data[
                (env.passenger_data["Time"] >= start_time) & (env.passenger_data["Time"] <= end_time)
            ].copy()
            env.current_time = start_time
            env.current_index = 0

            print(f"✅ Running simulation from {start_time.time()} to {end_time.time()}")

            while True:
                self.is_paused.wait()  # ⏸ Wait here if paused
                actions = []
                mode = env.detect_elevator_mode()
                for i in range(env.num_elevators):
                    if mode == "RUSH":
                        action = env.nearest_car_scan(i)
                    elif mode == "DYNAMIC-ASSIGN":
                        action = env.dynamic_assign_routing(i)
                    elif mode == "NORMAL":
                        action = env.energy_efficient_routing(i)
                    else:
                        action = env.energy_efficient_routing_best(i)
                    actions.append(action + 1)

                obs, reward, done, info = env.step(np.array(actions))
                
                # Update graph data
                # self.time_points.append(env.current_time.strftime("%H:%M:%S"))
                self.time_points.append(env.current_time)
                self.energy_data.append(sum(env.energy_usage))
                self.wait_time_data.append(sum(env.wait_times))
                self.service_time_data.append(sum(env.service_times))
                self.waiting_passenger_data.append(
                    sum(len(v['up']) + len(v['down']) for v in env.state['passengers_waiting'].values())
                )
                
                # Clear previous frame
                self.ax.clear()

                # Update matplotlib plot
                if self.show_energy.get():
                    self.ax.plot(self.time_points, self.energy_data, label="Energy ⚡", color="blue")
                if self.show_wait_time.get():
                    self.ax.plot(self.time_points, self.wait_time_data, label="Wait Time 🚶", color="orange")
                if self.show_service_time.get():
                    self.ax.plot(self.time_points, self.service_time_data, label="Service Time ⏳", color="green")
                if self.show_waiting_passengers.get():
                    self.ax.plot(self.time_points, self.waiting_passenger_data, label="Waiting 👥", color="red")


                self.ax.set_title("Live Simulation Metrics")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Value")

                # ✅ Format time on x-axis
                self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M:%S %p"))

                self.fig.autofmt_xdate()  # Auto wrap + rotate
                self.ax.grid(True)
                self.ax.legend(loc="upper left")
                self.fig.tight_layout()
                self.canvas.draw()


                

                if self.render_mode.get() == "3D":
                    env.render_3d()
                else:
                    env.render_2d()

                total_waiting = sum(len(v['up']) + len(v['down']) for v in env.state['passengers_waiting'].values())
                total_in_elevators = sum(obs['elevator_load'])

                if env.current_time >= end_time and total_waiting == 0 and total_in_elevators == 0:
                    break

                time.sleep(0.1)

            env.close()
            messagebox.showinfo("Simulation Complete", "✅ Simulation completed successfully!")
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="disabled")
            self.simulation_running = False


        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    show_splash_screen()  # 👈 Add splash screen first
    root = tk.Tk()
    app = ElevatorSimulatorGUI(root)
    root.mainloop()

