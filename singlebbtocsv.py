import os
import csv
import sys
import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose.pt'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        if not self.yolov8_model.split('-')[-1] == 'pose.pt':
            sys.exit('Model not YOLOv8 pose. Please provide a pose model.')
        self.model = YOLO(self.yolov8_model)

    def extract_keypoint(self, keypoint: np.ndarray) -> dict:
        def valid_point(kp):
            return kp if np.any(kp) else (np.nan, np.nan)

        keypoints = {
            'nose': valid_point(keypoint[self.get_keypoint.NOSE]),
            'left_eye': valid_point(keypoint[self.get_keypoint.LEFT_EYE]),
            'right_eye': valid_point(keypoint[self.get_keypoint.RIGHT_EYE]),
            'left_ear': valid_point(keypoint[self.get_keypoint.LEFT_EAR]),
            'right_ear': valid_point(keypoint[self.get_keypoint.RIGHT_EAR]),
            'left_shoulder': valid_point(keypoint[self.get_keypoint.LEFT_SHOULDER]),
            'right_shoulder': valid_point(keypoint[self.get_keypoint.RIGHT_SHOULDER]),
            'left_elbow': valid_point(keypoint[self.get_keypoint.LEFT_ELBOW]),
            'right_elbow': valid_point(keypoint[self.get_keypoint.RIGHT_ELBOW]),
            'left_wrist': valid_point(keypoint[self.get_keypoint.LEFT_WRIST]),
            'right_wrist': valid_point(keypoint[self.get_keypoint.RIGHT_WRIST]),
            'left_hip': valid_point(keypoint[self.get_keypoint.LEFT_HIP]),
            'right_hip': valid_point(keypoint[self.get_keypoint.RIGHT_HIP]),
            'left_knee': valid_point(keypoint[self.get_keypoint.LEFT_KNEE]),
            'right_knee': valid_point(keypoint[self.get_keypoint.RIGHT_KNEE]),
            'left_ankle': valid_point(keypoint[self.get_keypoint.LEFT_ANKLE]),
            'right_ankle': valid_point(keypoint[self.get_keypoint.RIGHT_ANKLE]),
        }
        return keypoints

    def get_xy_keypoint(self, results) -> dict:
        if results.keypoints is None or results.keypoints.xyn.shape[0] == 0:
            print("Nenhum keypoint detectado.")
            return {}

        result_keypoint = results.keypoints.xyn.cpu().numpy()[0] 
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data

    def detect_keypoints(self, image: np.ndarray):
        results = self.model(image)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print("Nenhuma bounding box detectada.")
            return None

        areas = []
        for i, box in enumerate(results.boxes.xyxy):  # xyxy formato das bounding boxes
            x1, y1, x2, y2 = box.cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            areas.append((i, area))
            print(f"Pessoa {i}: Área da bounding box = {area}")

        max_index, max_area = max(areas, key=lambda x: x[1])
        print(f"Maior bounding box escolhida (Pessoa {max_index}): Área = {max_area}")

        result_keypoint = results.keypoints.xyn.cpu().numpy()[max_index]
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data

def process_directories(directories, output_csv):
    detector = DetectKeypoint('yolov8m-pose.pt')
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'filename', 'pose_category',
            'nose_x', 'nose_y',
            'left_eye_x', 'left_eye_y',
            'right_eye_x', 'right_eye_y',
            'left_ear_x', 'left_ear_y',
            'right_ear_x', 'right_ear_y',
            'left_shoulder_x', 'left_shoulder_y',
            'right_shoulder_x', 'right_shoulder_y',
            'left_elbow_x', 'left_elbow_y',
            'right_elbow_x', 'right_elbow_y',
            'left_wrist_x', 'left_wrist_y',
            'right_wrist_x', 'right_wrist_y',
            'left_hip_x', 'left_hip_y',
            'right_hip_x', 'right_hip_y',
            'left_knee_x', 'left_knee_y',
            'right_knee_x', 'right_knee_y',
            'left_ankle_x', 'left_ankle_y',
            'right_ankle_x', 'right_ankle_y'
        ])
        
        for directory, pose_category in directories:
            for filename in os.listdir(directory):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(directory, filename)
                    img = cv2.imread(img_path)
                    keypoints = detector.detect_keypoints(img)  # Use o método detect_keypoints
                    if keypoints:
                        row = [filename, pose_category] + \
                              [keypoints['nose'][0], keypoints['nose'][1]] + \
                              [keypoints['left_eye'][0], keypoints['left_eye'][1]] + \
                              [keypoints['right_eye'][0], keypoints['right_eye'][1]] + \
                              [keypoints['left_ear'][0], keypoints['left_ear'][1]] + \
                              [keypoints['right_ear'][0], keypoints['right_ear'][1]] + \
                              [keypoints['left_shoulder'][0], keypoints['left_shoulder'][1]] + \
                              [keypoints['right_shoulder'][0], keypoints['right_shoulder'][1]] + \
                              [keypoints['left_elbow'][0], keypoints['left_elbow'][1]] + \
                              [keypoints['right_elbow'][0], keypoints['right_elbow'][1]] + \
                              [keypoints['left_wrist'][0], keypoints['left_wrist'][1]] + \
                              [keypoints['right_wrist'][0], keypoints['right_wrist'][1]] + \
                              [keypoints['left_hip'][0], keypoints['left_hip'][1]] + \
                              [keypoints['right_hip'][0], keypoints['right_hip'][1]] + \
                              [keypoints['left_knee'][0], keypoints['left_knee'][1]] + \
                              [keypoints['right_knee'][0], keypoints['right_knee'][1]] + \
                              [keypoints['left_ankle'][0], keypoints['left_ankle'][1]] + \
                              [keypoints['right_ankle'][0], keypoints['right_ankle'][1]]
                        writer.writerow(row)

if __name__ == "__main__":
    directories = [
        ('defdataset/takeoff', 'takeoff'),
        ('defdataset/land', 'land'),
        ('defdataset/go_up', 'go_up'),
        ('defdataset/go_down', 'go_down'),
        ('defdataset/go_left', 'go_left'),
        ('defdataset/go_right', 'go_right'),
        ('defdataset/go_forward', 'go_forward'),
        ('defdataset/go_back', 'go_back'),
        ('defdataset/follow_me', 'follow_me'),
        ('defdataset/nao_acao', 'nao_acao'),
        
    ]
    process_directories(directories, 'keypointextraction.csv')
