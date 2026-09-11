import cv2
from ultralytics import YOLO

# 1. Cargar el modelo preentrenado YOLOv8 (versión n=nano, muy rápida)
model = YOLO('yolov8n.pt')

# 2. Iniciar la webcam (0=cam por defecto)
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Pulsa la tecla 'q' en la ventana de video para salir.")

while True:
    # 3. Leer cada frame en tiempo real
    exito, frame = cap.read()
    if not exito:
        print("No se pudo acceder a la cámara.")
        break

    # 4. Pasar el frame de la a YOLO para que detecte objetos
    resultados = model(frame)

    # 5. Pedirle a YOLO que dibuje las cajas de colores sobre los objetos detectados
    frame_anotado = resultados[0].plot()

    # 6. Mostrar el resultado en una ventana
    cv2.imshow("Mi primer detector - YOLOv8", frame_anotado)

    # 7. Si pulsas 'q', se rompe el bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Limpiar y apagar la cámara al terminar
cap.release()
cv2.destroyAllWindows()