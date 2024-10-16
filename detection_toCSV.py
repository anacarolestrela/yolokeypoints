import os
import csv
import sys
import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

# Define keypoints using a Pydantic model
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

# Classe para detectar keypoints usando o modelo YOLOv8 de pose
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
        keypoints = {
            'nose': keypoint[self.get_keypoint.NOSE],
            'left_eye': keypoint[self.get_keypoint.LEFT_EYE],
            'right_eye': keypoint[self.get_keypoint.RIGHT_EYE],
            'left_ear': keypoint[self.get_keypoint.LEFT_EAR],
            'right_ear': keypoint[self.get_keypoint.RIGHT_EAR],
            'left_shoulder': keypoint[self.get_keypoint.LEFT_SHOULDER],
            'right_shoulder': keypoint[self.get_keypoint.RIGHT_SHOULDER],
            'left_elbow': keypoint[self.get_keypoint.LEFT_ELBOW],
            'right_elbow': keypoint[self.get_keypoint.RIGHT_ELBOW],
            'left_wrist': keypoint[self.get_keypoint.LEFT_WRIST],
            'right_wrist': keypoint[self.get_keypoint.RIGHT_WRIST],
            'left_hip': keypoint[self.get_keypoint.LEFT_HIP],
            'right_hip': keypoint[self.get_keypoint.RIGHT_HIP],
            'left_knee': keypoint[self.get_keypoint.LEFT_KNEE],
            'right_knee': keypoint[self.get_keypoint.RIGHT_KNEE],
            'left_ankle': keypoint[self.get_keypoint.LEFT_ANKLE],
            'right_ankle': keypoint[self.get_keypoint.RIGHT_ANKLE],
        }
        return keypoints

    def get_xy_keypoint(self, results) -> dict:
        if results.keypoints is None or results.keypoints.xyn.shape[0] == 0:
            print("Nenhum keypoint detectado.")
            return {}

        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]  # Normalizado entre 0 e 1
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data

    def __call__(self, image: np.ndarray):
        results = self.model(image)[0]
        return results

def flatten_keypoints(keypoints: dict) -> list:
    flattened = []
    for key in keypoints:
        flattened.extend([keypoints[key][0], keypoints[key][1]])  # Adiciona x, y separadamente
    return flattened

def has_valid_keypoints(keypoints: dict) -> bool:
    """Verifica se todos os keypoints de 0 a 12 têm valores válidos (não são 0 ou NaN)."""
    required_keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip'
    ]
    
    for key in required_keypoints:
        x, y = keypoints[key]
        if x == 0 or y == 0 or np.isnan(x) or np.isnan(y):
            return False
    return True

def process_directories(directories, output_csv):
    detector = DetectKeypoint('yolov8m-pose.pt')
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
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
                    results = detector(img)
                    keypoints = detector.get_xy_keypoint(results)
                    if keypoints and has_valid_keypoints(keypoints):
                        flattened_keypoints = flatten_keypoints(keypoints)
                        row = [filename, pose_category] + flattened_keypoints
                        writer.writerow(row)

# Processar as imagens das pastas e gerar o CSV
if __name__ == "__main__":
    directories = [
        ('dataset_ofc/takeoff', 'takeoff'),
        ('dataset_ofc/land', 'land'),
        ('dataset_ofc/go_up', 'go_up'),
        ('dataset_ofc/go_down', 'go_down'),
        ('dataset_ofc/go_left', 'go_left'),
        ('dataset_ofc/go_right', 'go_right'),
        ('dataset_ofc/go_forward', 'go_forward'),
        ('dataset_ofc/go_back', 'go_back'),
        ('dataset_ofc/nao_acao', 'nao_acao'),
        
    ]
    process_directories(directories, 'kpdataset.csv')
