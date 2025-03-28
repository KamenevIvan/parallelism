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
logging.basicConfig(filename=os.path.join(logPath, 'task4Errors.log'), level=logging.ERROR,
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
        if camera_frame is not None and isinstance(camera_frame, np.ndarray):  # Проверяем тип
            self.last_camera_frame = camera_frame.copy()  # Теперь copy() сработает
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

def sensor_worker(sensor: Sensor, data_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            data = sensor.get()
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
            threading.Thread(target=sensor_worker, args=(sensor0, sensor0_queue, stop_event)),
            threading.Thread(target=sensor_worker, args=(sensor1, sensor1_queue, stop_event)),
            threading.Thread(target=sensor_worker, args=(sensor2, sensor2_queue, stop_event))
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        window = WindowImage(args.frequency)
        
        display_interval = 1.0 / args.frequency
        last_display_time = time.time()
        
        while True:
            current_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
                #while not sensor0_queue.empty():
                    _, sensor0_data = sensor0_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                #while not sensor1_queue.empty():
                    _, sensor1_data = sensor1_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                #while not sensor2_queue.empty():
                    _, sensor2_data = sensor2_queue.get_nowait()
            except queue.Empty:
                pass
            
            if current_time - last_display_time >= display_interval:
                window.update_display(camera_frame, sensor0_data, sensor1_data, sensor2_data)
                last_display_time = current_time
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