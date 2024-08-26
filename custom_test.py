from ultralytics import YOLO
import cv2

# Carregar o modelo YOLOv8 de pose pré-treinado
model = YOLO('yolov8m-pose.pt')


# Capturar o vídeo
video_path = 'videotest.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi carregado corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Definir a taxa de processamento (por exemplo, processar 1 a cada 5 quadros)
process_rate = 10
frame_count = 0

# Loop para processar o vídeo
while True:
    ret, frame = cap.read()  # Ler um quadro do vídeo
    if not ret:
        break  # Se não houver mais quadros, sair do loop

    # Processar apenas 1 a cada `process_rate` quadros
    if frame_count % process_rate == 0:
        results = model(frame, conf=0.3)  # Realizar inferência no quadro atual

        # Iterar sobre os resultados e renderizar as detecções
        for result in results:
            annotated_frame = result.plot()  # Desenhar as detecções no quadro

        # Exibir o quadro anotado
        cv2.imshow('YOLOv8 Pose Detection', annotated_frame)

    frame_count += 1

    # Parar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
