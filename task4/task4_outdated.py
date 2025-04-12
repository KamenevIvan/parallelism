import cv2
import time
import argparse
import logging
import threading
import queue
import os
from typing import Tuple, Optional
import numpy as np
import win32api
import win32process
import win32con


logPath = "log"
os.makedirs(logPath, exist_ok=True)
logging.basicConfig(filename=os.path.join(logPath, 'task4Errors.log'), level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def set_realtime_priority():
    try:
        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
        print("Установлен REAL-TIME приоритет")
    except Exception as e:
        print(f"Не удалось установить приоритет: {e}")

class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")
    
class SensorX(Sensor):
    def __init__(self, delay):
        self._delay = delay
        self._data = 0
        self._next_call = time.perf_counter()  
    
    def get(self):
        now = time.perf_counter()
        if now < self._next_call:
            time.sleep(max(0, self._next_call - now - 0.001))  
        
        self._data += 1
        self._next_call += self._delay  
        return self._data

class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: Tuple[int, int]):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = None
        
        try:
            if camera_name.startswith('/dev/video'):
                camera_index = int(camera_name[10:])
                self.cap = cv2.VideoCapture(camera_index)
            else:
                self.cap = cv2.VideoCapture(camera_name)
                
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {camera_name}")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
        except Exception as e:
            logging.error(f"Error initializing camera {camera_name}: {str(e)}")
            if self.cap:
                self.cap.release()
            raise
    
    def get(self) -> Optional[np.ndarray]:
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame from camera")
                return None
            return frame  
        except Exception as e:
            logging.error(f"Error reading from camera: {str(e)}")
            return None
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class WindowImage:
    def __init__(self, display_frequency: float):
        self.display_frequency = display_frequency
        self.window_name = "Sensor Display"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        self.last_camera_frame = None
        self.last_sensor0_data = 0
        self.last_sensor1_data = 0
        self.last_sensor2_data = 0
    
    def update_display(self, camera_frame, sensor0_data, sensor1_data, sensor2_data):
        if camera_frame is not None and isinstance(camera_frame, np.ndarray):  
            self.last_camera_frame = camera_frame.copy()  
        if sensor0_data is not None:
            self.last_sensor0_data = sensor0_data
        if sensor1_data is not None:
            self.last_sensor1_data = sensor1_data
        if sensor2_data is not None:
            self.last_sensor2_data = sensor2_data
        
        if self.last_camera_frame is not None:
            display_frame = self.last_camera_frame.copy()
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, f"Sensor0: {self.last_sensor0_data}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Sensor1: {self.last_sensor1_data}", (10, 70), font, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Sensor2: {self.last_sensor2_data}", (10, 110), font, 1, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
    
    def __del__(self):
        cv2.destroyWindow(self.window_name)

def sensor0_worker(sensor, data_queue, stop_event):
    from ctypes import windll
    windll.winmm.timeBeginPeriod(1)  
    
    while not stop_event.is_set():
        data = sensor.get()
        try:
            data_queue.put_nowait(('sensor0', data))  
        except queue.Full:
            pass  
    
    windll.winmm.timeEndPeriod(1)   

def sensor_worker(sensor: Sensor, data_queue: queue.Queue, stop_event: threading.Event):
    
    sample_count = 0
    start_time = time.perf_counter()
    last_print_time = start_time
    last_timestamp = start_time
    
    while not stop_event.is_set():
        try:
            timestamp_before = time.perf_counter()
            
            data = sensor.get()
        
            if isinstance(sensor, SensorX):
                current_time = time.perf_counter()
                sample_count += 1

                if current_time - last_print_time >= 2.0:
                    elapsed = current_time - start_time
                    avg_freq = sample_count / elapsed
                    last_interval = current_time - last_timestamp
                    
                    print(
                        f"Sensor{getattr(sensor, '_delay', '?')} frequency: "
                        f"Current={1/last_interval:.1f}Hz, "
                        f"Avg={avg_freq:.1f}Hz, "
                        f"Samples={sample_count}"
                    )
                    
                    sample_count = 0
                    start_time = current_time
                    last_print_time = current_time
                
                last_timestamp = current_time
            
            
            if isinstance(sensor, SensorCam) and data is not None:
                print(f"Camera frame shape: {data.shape}")  
            
            data_queue.put((sensor, data))
            
        except Exception as e:
            logging.error(f"Sensor error: {str(e)}")
            break

def main():
    
    parser = argparse.ArgumentParser(description="Sensor and camera data display")
    parser.add_argument("--camera", type=str, default="/dev/video0", help="Camera device name")
    parser.add_argument("--resolution", type=str, default="640x480", help="Camera resolution (e.g. 1280x720)")
    parser.add_argument("--frequency", type=float, default=30.0, help="Display frequency (Hz)")
    
    args = parser.parse_args()
    
    try:
        
        width, height = map(int, args.resolution.split('x'))
        
        sensor0 = SensorX(0.01)  
        sensor1 = SensorX(0.1)   
        sensor2 = SensorX(1.0)   
        sensor_cam = SensorCam(args.camera, (width, height))
        
        
        cam_queue = queue.Queue(maxsize=1)
        sensor0_queue = queue.Queue(maxsize=1)
        sensor1_queue = queue.Queue(maxsize=1)
        sensor2_queue = queue.Queue(maxsize=1)
        
        stop_event = threading.Event()
        
        threads = [
            threading.Thread(target=sensor_worker, args=(sensor_cam, cam_queue, stop_event)),
            threading.Thread(target=sensor0_worker, args=(sensor0, sensor0_queue, stop_event)),
            threading.Thread(target=sensor_worker, args=(sensor1, sensor1_queue, stop_event)),
            threading.Thread(target=sensor_worker, args=(sensor2, sensor2_queue, stop_event))
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        window = WindowImage(args.frequency)
        
        display_interval = 1.0 / args.frequency
        last_display_time = time.time()

        last_sensor0_value = 0
        last_sensor0_time = time.time()
        last_sensor1_value = 0
        last_sensor1_time = time.time()
        last_sensor2_value = 0
        last_sensor2_time = time.time()
        
        while True:
            current_time = time.time()
            
            if cv2.pollKey() == ord('q'):
                    break
            
            camera_frame = None
            sensor0_data = None
            sensor1_data = None
            sensor2_data = None
            
            try:
                #while not cam_queue.empty():
                    _, camera_frame = cam_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                _, sensor0_data = sensor0_queue.get_nowait()
                #print(f"S0: {sensor0_data}") if sensor0_data % 100 == 0 else None  # Логируем каждые 100 отсчётов
            except queue.Empty:
                pass
            
            try:
                #while not sensor1_queue.empty():
                    _, sensor1_data = sensor1_queue.get_nowait()
                    last_sensor1_time = current_time  
            except queue.Empty:
                if 'last_sensor1_time' in locals():
                    expected_increment = (current_time - last_sensor1_time) / 0.1  
                    sensor1_data = last_sensor1_value + int(expected_increment)
                    
            try:
                #while not sensor2_queue.empty():
                    _, sensor2_data = sensor2_queue.get_nowait()
                    last_sensor2_time = current_time  
            except queue.Empty:
                if 'last_sensor0_time' in locals():
                    expected_increment = (current_time - last_sensor2_time) / 1.0
                    sensor2_data = last_sensor2_value + int(expected_increment)
            
            if sensor0_data is not None:
                last_sensor0_value = sensor0_data
            
            if sensor1_data is not None:
                last_sensor1_value = sensor1_data

            if sensor2_data is not None:
                last_sensor2_value = sensor2_data


            if time.time() - last_display_time >= display_interval:
                #print("\n\n\n!!!!!\n\n\n")
                window.update_display(camera_frame, sensor0_data, sensor1_data, sensor2_data)
                last_display_time = current_time
                #print("\n\n\n&&&&&&&&\n\n\n")
                cv2.waitKey(1)
    
    except Exception as e:
        logging.error(f"Main program error: {str(e)}")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        
        if 'window' in locals():
            del window
        if 'sensor_cam' in locals():
            del sensor_cam

if __name__ == "__main__":
    main()

# python task4.py --camera /dev/video0 --resolution 1280x720 --frequency 24