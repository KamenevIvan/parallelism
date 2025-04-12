import cv2
import time
import argparse
import logging
import threading
import queue
import os
from typing import Tuple, Optional
import numpy as np


logPath = "log"
os.makedirs(logPath, exist_ok=True)
logging.basicConfig(filename=os.path.join(logPath, 'task4Errors.log'), 
                   level=logging.ERROR,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")

class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: Tuple[int, int]):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logging.error(f"Cannot open camera {camera_name}")
            raise RuntimeError(f"Cannot open camera {camera_name}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    def get(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame from camera")
            return None
        return frame
    
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
        if camera_frame is not None:
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

def sensor_worker(sensor: Sensor, data_queue: queue.Queue):
    while True:
        data = sensor.get()
        data_queue.put(data)

def get_latest_from_queue(q: queue.Queue):
    latest = None
    while not q.empty():
        latest = q.get_nowait()
    return latest

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
        
        
        cam_queue = queue.Queue()
        sensor0_queue = queue.Queue()
        sensor1_queue = queue.Queue()
        sensor2_queue = queue.Queue()
        
        threads = [
            threading.Thread(target=sensor_worker, args=(sensor_cam, cam_queue), daemon=True),
            threading.Thread(target=sensor_worker, args=(sensor0, sensor0_queue), daemon=True),
            threading.Thread(target=sensor_worker, args=(sensor1, sensor1_queue), daemon=True),
            threading.Thread(target=sensor_worker, args=(sensor2, sensor2_queue), daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        window = WindowImage(args.frequency)
        
        
        while True:
            frame = get_latest_from_queue(cam_queue)
            s0_data = get_latest_from_queue(sensor0_queue)
            s1_data = get_latest_from_queue(sensor1_queue)
            s2_data = get_latest_from_queue(sensor2_queue)
            
            window.update_display(frame, s0_data, s1_data, s2_data)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    except Exception as e:
        logging.error(f"Main program error: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()