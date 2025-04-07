import torch
torch.set_num_threads(1)

import argparse
import cv2
cv2.setNumThreads(1)
import time
from ultralytics import YOLO
import concurrent.futures
import numpy as np

class PoseModel:
    
    def __init__(self):
        self.model = YOLO('yolov8s-pose.pt')
    
    def predict(self, frame):
        results = self.model(frame)
        return results[0].plot()

def read_frames(input_path):
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def write_video(output_path, frames, fps):
    if not frames:
        return
    
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  
    
    # Создаем VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise IOError("Не удалось создать видеофайл!")
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Видео успешно сохранено: {output_path}")

def process_single_thread(input_path):
    model = PoseModel()
    frames = read_frames(input_path)
    processed_frames = []
    start_time = time.time()
    for frame in frames:
        processed_frames.append(model.predict(frame))
    print(f"Время обработки: {time.time() - start_time:.2f} секунд")
    write_video("output_single.avi", processed_frames, 60) #, (640, 480)

def init_process():
    global process_model
    process_model = PoseModel()

def process_frame(frame):
    for _ in range(10):  
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return process_model.predict(frame)

def process_multi_thread(input_path, workers):
    frames = read_frames(input_path)
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=init_process) as executor:
        processed_frames = list(executor.map(process_frame, frames))
    print(f"Время обработки: {time.time() - start_time:.2f} секунд")
    write_video("output_multi.avi", processed_frames, 60) #, (640, 480)

def main():
    parser = argparse.ArgumentParser(description='Обработка видео с YOLOv8.')
    parser.add_argument('--input', type=str, required=True, help='e2.mp4')
    parser.add_argument('--mode', choices=['single', 'multi'], required=True, help='Режим выполнения')
    #parser.add_argument('--output', type=str, required=True, help='e2_proccessed')
    parser.add_argument('--workers', type=int, default=4, help='Количество процессов')
    args = parser.parse_args()

    if args.mode == 'single':
        process_single_thread(args.input)
    else:
        process_multi_thread(args.input, args.workers)

if __name__ == "__main__":
    main()