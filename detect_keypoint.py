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
        # Verifica se o modelo é de pose, caso contrário, encerra o programa
        if not self.yolov8_model.split('-')[-1] == 'pose.pt':
            sys.exit('Model not YOLOv8 pose. Please provide a pose model.')
        # Carrega o modelo YOLOv8
        self.model = YOLO(self.yolov8_model)

    # Função para extrair e nomear os keypoints
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

    # Função para obter as coordenadas X, Y dos keypoints
    def get_xy_keypoint(self, results) -> dict:
        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]  # Normalizado entre 0 e 1
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data

    # Função para realizar a inferência
    def __call__(self, image: np.ndarray):
        results = self.model(image)[0]
        return results

# Exemplo de uso da classe:
if __name__ == "__main__":
    # Inicializa o detector de keypoints
    detector = DetectKeypoint('yolov8m-pose.pt')

    # Carrega uma imagem de teste
    img = cv2.imread('bus.jpg')

    # Faz a inferência e obtém os resultados
    results = detector(img)

    # Extrai e exibe os keypoints nomeados
    keypoints = detector.get_xy_keypoint(results)
    print(keypoints)
