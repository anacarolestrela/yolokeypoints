import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

# Define keypoints usando um modelo Pydantic
class GetKeypoint(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16

class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose.pt'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        self.model = YOLO(self.yolov8_model)

    def extract_keypoint(self, keypoint: np.ndarray) -> dict:
        keypoints = {
            'nose': keypoint[self.get_keypoint.NOSE] if self.get_keypoint.NOSE < len(keypoint) else (None, None),
            'left_eye': keypoint[self.get_keypoint.LEFT_EYE] if self.get_keypoint.LEFT_EYE < len(keypoint) else (None, None),
            'right_eye': keypoint[self.get_keypoint.RIGHT_EYE] if self.get_keypoint.RIGHT_EYE < len(keypoint) else (None, None),
            'left_ear': keypoint[self.get_keypoint.LEFT_EAR] if self.get_keypoint.LEFT_EAR < len(keypoint) else (None, None),
            'right_ear': keypoint[self.get_keypoint.RIGHT_EAR] if self.get_keypoint.RIGHT_EAR < len(keypoint) else (None, None),
            'left_shoulder': keypoint[self.get_keypoint.LEFT_SHOULDER] if self.get_keypoint.LEFT_SHOULDER < len(keypoint) else (None, None),
            'right_shoulder': keypoint[self.get_keypoint.RIGHT_SHOULDER] if self.get_keypoint.RIGHT_SHOULDER < len(keypoint) else (None, None),
            'left_elbow': keypoint[self.get_keypoint.LEFT_ELBOW] if self.get_keypoint.LEFT_ELBOW < len(keypoint) else (None, None),
            'right_elbow': keypoint[self.get_keypoint.RIGHT_ELBOW] if self.get_keypoint.RIGHT_ELBOW < len(keypoint) else (None, None),
            'left_wrist': keypoint[self.get_keypoint.LEFT_WRIST] if self.get_keypoint.LEFT_WRIST < len(keypoint) else (None, None),
            'right_wrist': keypoint[self.get_keypoint.RIGHT_WRIST] if self.get_keypoint.RIGHT_WRIST < len(keypoint) else (None, None),
            'left_hip': keypoint[self.get_keypoint.LEFT_HIP] if self.get_keypoint.LEFT_HIP < len(keypoint) else (None, None),
            'right_hip': keypoint[self.get_keypoint.RIGHT_HIP] if self.get_keypoint.RIGHT_HIP < len(keypoint) else (None, None),
            'left_knee': keypoint[self.get_keypoint.LEFT_KNEE] if self.get_keypoint.LEFT_KNEE < len(keypoint) else (None, None),
            'right_knee': keypoint[self.get_keypoint.RIGHT_KNEE] if self.get_keypoint.RIGHT_KNEE < len(keypoint) else (None, None),
            'left_ankle': keypoint[self.get_keypoint.LEFT_ANKLE] if self.get_keypoint.LEFT_ANKLE < len(keypoint) else (None, None),
            'right_ankle': keypoint[self.get_keypoint.RIGHT_ANKLE] if self.get_keypoint.RIGHT_ANKLE < len(keypoint) else (None, None),
        }
        return keypoints

    def get_xy_keypoint(self, results) -> dict:
        if results.keypoints is None or results.keypoints.xyn.shape[0] == 0:
            print("No keypoints detected.")
            return {}

        result_keypoint = results.keypoints.xyn.cpu().numpy()
        if len(result_keypoint) == 0:
            print("No keypoints in result.")
            return {}

        keypoint_data = self.extract_keypoint(result_keypoint[0])
        return keypoint_data

    def __call__(self, image: np.ndarray):
        results = self.model(image)[0]
        return results

def process_video(video_path):
    detector = DetectKeypoint('yolov8n-pose.pt')
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector(frame)
        keypoints = detector.get_xy_keypoint(results)
        
        if keypoints:
            for name, (x, y) in keypoints.items():
                if x is not None and y is not None:
                    x_int = int(x * frame.shape[1])
                    y_int = int(y * frame.shape[0])
                    cv2.circle(frame, (x_int, y_int), 5, (0, 255, 0), -1)
                    cv2.putText(frame, name, (x_int, y_int - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
if __name__ == "__main__":
    process_video('videotest.mp4')
