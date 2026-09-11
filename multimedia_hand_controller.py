import os
import time
import urllib.request
import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

# ---------------------------------------------------------
# Configuración del modelo MediaPipe Hand Landmarker
# ---------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Descargando modelo 'hand_landmarker.task'...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Modelo descargado correctamente.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

detector = HandLandmarker.create_from_options(options)

# ---------------------------------------------------------
# Control de intervalos y anti-spam global
# ---------------------------------------------------------
tiempo_ultima_accion = 0
ultimo_gesto_ejecutado = None

COOLDOWN_GENERAL = 1.0       # Margen estricto obligatorio entre cualquier gesto distinto
INTERVALO_VOLUMEN = 0.15     # Repetición fluida permitida únicamente al sostener el volumen


def clasificar_gesto(landmarks):
    """
    Analiza la geometría integral de la mano. Exige un cierre real del puño 
    y posiciones relativas estrictas del pulgar para evitar activaciones erróneas.
    """
    # 1. Verificación estricta del cierre de los 4 dedos principales
    indice_cerrado = landmarks[8].y > landmarks[6].y
    corazon_cerrado = landmarks[12].y > landmarks[10].y
    anular_cerrado = landmarks[16].y > landmarks[14].y
    menique_cerrado = landmarks[20].y > landmarks[18].y
    puño_cerrado = indice_cerrado and corazon_cerrado and anular_cerrado and menique_cerrado

    # Verificación de dedos extendidos para control multimedia
    indice_arriba = landmarks[8].y < landmarks[6].y
    corazon_arriba = landmarks[12].y < landmarks[10].y
    anular_arriba = landmarks[16].y < landmarks[14].y
    menique_arriba = landmarks[20].y < landmarks[18].y

    # 2. Posicionamiento avanzado del pulgar validado con el contexto de la mano
    # Pulgar Arriba: El puño está cerrado, el pulgar apunta hacia arriba y supera con margen la base del índice
    pulgar_arriba = (
        puño_cerrado
        and landmarks[4].y < landmarks[3].y 
        and landmarks[3].y < landmarks[2].y 
        and landmarks[4].y < landmarks[5].y
        and (landmarks[5].y - landmarks[4].y) > 0.04
    )
    
    # Pulgar Abajo: El puño está cerrado y la punta desciende claramente por debajo de la muñeca
    pulgar_abajo = (
        puño_cerrado
        and landmarks[4].y > landmarks[3].y 
        and landmarks[3].y > landmarks[2].y 
        and landmarks[4].y > landmarks[0].y
        and (landmarks[4].y - landmarks[0].y) > 0.04
    )

    # 3. Mapeo de estados y gestos
    if puño_cerrado:
        if pulgar_arriba:
            return "VOL_UP"
        elif pulgar_abajo:
            return "VOL_DOWN"
        return "REPOSO"

    if indice_arriba and not corazon_arriba and not anular_arriba and not menique_arriba:
        return "PLAY_PAUSA"

    if indice_arriba and corazon_arriba and not anular_arriba and not menique_arriba:
        return "SIGUIENTE"

    if indice_arriba and corazon_arriba and anular_arriba and menique_arriba:
        return "ANTERIOR"

    return "DESCONOCIDO"


def ejecutar_accion(gesto):
    acciones = {
        "PLAY_PAUSA": ("playpause", "[ACCIÓN] Play / Pausa"),
        "SIGUIENTE": ("nexttrack", "[ACCIÓN] Siguiente canción"),
        "ANTERIOR": ("prevtrack", "[ACCIÓN] Anterior canción"),
        "VOL_UP": ("volumeup", "[ACCIÓN] Subir volumen"),
        "VOL_DOWN": ("volumedown", "[ACCIÓN] Bajar volumen"),
    }
    
    if gesto in acciones:
        tecla, descripcion = acciones[gesto]
        print(descripcion)
        pyautogui.press(tecla)


# ---------------------------------------------------------
# Bucle Principal de Captura
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

print("\n==================================================")
print("       CONTROLADOR MULTIMEDIA POR GESTOS")
print("==================================================")
print(" ✊ Puño cerrado    -> REPOSO (Sin acción)")
print(" ☝️ Índice solo     -> Play / Pausa")
print(" ✌️ Victoria        -> Siguiente canción")
print(" 🖐️ Mano abierta    -> Anterior canción")
print(" 👍 Pulgar arriba   -> Subir volumen")
print(" 👎 Pulgar abajo    -> Bajar volumen")
print("--------------------------------------------------")
print(" Presione 'q' en la ventana de video para salir.\n")

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        print("[ERROR] No se pudo leer el fotograma de la cámara.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    texto_interfaz = "Buscando mano..."
    color_texto = (180, 180, 180)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            gesto_actual = clasificar_gesto(hand_landmarks)
            tiempo_actual = time.time()

            if gesto_actual == "REPOSO":
                texto_interfaz = "Estado: REPOSO ✊"
                color_texto = (120, 120, 120)
                ultimo_gesto_ejecutado = None
            elif gesto_actual != "DESCONOCIDO":
                texto_interfaz = f"Gesto: {gesto_actual}"

                # Control anti-spam estricto y gestión de transiciones rápidas
                if gesto_actual in ["VOL_UP", "VOL_DOWN"]:
                    if gesto_actual == ultimo_gesto_ejecutado:
                        # Permite repetición fluida si mantienes el mismo gesto de volumen
                        if (tiempo_actual - tiempo_ultima_accion) > INTERVALO_VOLUMEN:
                            ejecutar_accion(gesto_actual)
                            tiempo_ultima_accion = tiempo_actual
                            color_texto = (0, 255, 0)
                    else:
                        # Exige cooldown general si cambias de gesto a volumen
                        if (tiempo_actual - tiempo_ultima_accion) > COOLDOWN_GENERAL:
                            ejecutar_accion(gesto_actual)
                            tiempo_ultima_accion = tiempo_actual
                            ultimo_gesto_ejecutado = gesto_actual
                            color_texto = (0, 255, 0)
                else:
                    # Bloqueo total ante cualquier cambio rápido entre acciones distintas
                    if (tiempo_actual - tiempo_ultima_accion) > COOLDOWN_GENERAL:
                        ejecutar_accion(gesto_actual)
                        tiempo_ultima_accion = tiempo_actual
                        ultimo_gesto_ejecutado = gesto_actual
                        color_texto = (0, 255, 0)

    # Mostrar información en la interfaz
    cv2.putText(
        frame,
        texto_interfaz,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_texto,
        2,
    )
    
    cv2.imshow("Controlador Multimedia", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cerrar recursos y limpiar
detector.close()
cap.release()
cv2.destroyAllWindows()
print("\nApagando cacharros... ¡Chao!")