import math
import os
import time
import urllib.request
from collections import deque
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuración de PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
ANCHO_PANTALLA, ALTO_PANTALLA = pyautogui.size()

# Márgenes de trabajo en la cámara (zona interactiva)
MARGEN_X = 200
MARGEN_TOP = 60
MARGEN_BOTTOM = 220

# Umbrales para el clic (distancia relativa a la palma)
PRE_PINCH = 0.38   # Congela el cursor justo antes de hacer clic
CLICK_DOWN = 0.28  # Dispara el clic
CLICK_UP = 0.40    # Libera el clic

# Conexiones de los puntos de la mano para dibujar el esqueleto
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),    # Corazón
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17)                                # Palma
]


class FiltroEMA:
    """Suaviza el movimiento del cursor mediante media móvil exponencial."""
    def __init__(self, alpha=0.22):
        self.alpha = alpha
        self.x, self.y = 0, 0
        self.inicializado = False

    def actualizar(self, target_x, target_y):
        if not self.inicializado:
            self.x, self.y = target_x, target_y
            self.inicializado = True
        else:
            self.x += self.alpha * (target_x - self.x)
            self.y += self.alpha * (target_y - self.y)
        return int(self.x), int(self.y)


class EstabilizadorGesto:
    """Evita falsos positivos requiriendo N frames consecutivos del mismo gesto."""
    def __init__(self, tamano_buffer=4):
        self.buffer = deque(maxlen=tamano_buffer)

    def actualizar(self, gesto_raw):
        self.buffer.append(gesto_raw)
        if len(self.buffer) == self.buffer.maxlen and len(set(self.buffer)) == 1:
            return self.buffer[0]
        return None


def descargar_modelo():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Descargando modelo 'hand_landmarker.task'...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Modelo descargado correctamente.")
        except Exception as e:
            print(f"Error al descargar el modelo: {e}")
            exit(1)
    return model_path


def detectar_gesto(lm):
    """Determina el gesto actual según la posición de los dedos."""
    i_up = lm[8].y < lm[6].y
    m_up = lm[12].y < lm[10].y
    r_up = lm[16].y < lm[14].y
    p_up = lm[20].y < lm[18].y

    if i_up and not m_up and not r_up and not p_up:
        return "RATON"
    if i_up and m_up and not r_up and not p_up:
        return "SCROLL"
    if i_up and m_up and r_up and p_up:
        return "VOLUMEN"
    return "REPOSO"


def dibujar_esqueleto(frame, lm, w, h):
    """Dibuja las conexiones y puntos clave de la mano."""
    puntos = [(int(p.x * w), int(p.y * h)) for p in lm]
    
    for p1, p2 in CONEXIONES_MANO:
        cv2.line(frame, puntos[p1], puntos[p2], (0, 200, 255), 1)
    
    for p in puntos:
        cv2.circle(frame, p, 3, (0, 255, 255), -1)


def main():
    model_path = descargar_modelo()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    filtro_raton = FiltroEMA(alpha=0.22)
    estabilizador = EstabilizadorGesto(tamano_buffer=4)

    haciendo_clic = False
    ultimo_tiempo_vol = 0
    modo_actual = "REPOSO"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    nombre_ventana = "Control por Gestos"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    print("\n--- Control por Gestos ---")
    print(" ☝️  Índice: Mover cursor (juntar con pulgar para clic)")
    print(" ✌️  Índice + Corazón: Scroll arriba / abajo")
    print(" 🖐️  Mano abierta: Control de volumen por altura")
    print(" ✊  Puño / Otros: Pausa")
    print("\n Presiona 'q' o cierra la ventana para salir.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            tiempo_actual = time.time()
            resultado = detector.detect_for_video(mp_image, int(tiempo_actual * 1000))

            gesto_raw = "REPOSO"
            texto_estado = "Esperando mano..."
            color_hud = (128, 128, 128)

            if resultado.hand_landmarks:
                lm = resultado.hand_landmarks[0]

                # Tamaño de referencia de la palma (muñeca a base de dedo corazón)
                muneca = (int(lm[0].x * w), int(lm[0].y * h))
                base_corazon = (int(lm[9].x * w), int(lm[9].y * h))
                tamano_palma = max(10, math.hypot(base_corazon[0] - muneca[0], base_corazon[1] - muneca[1]))

                idx_pos = (int(lm[8].x * w), int(lm[8].y * h))
                pulgar_pos = (int(lm[4].x * w), int(lm[4].y * h))

                gesto_raw = detectar_gesto(lm)
                modo_confirmado = estabilizador.actualizar(gesto_raw)
                if modo_confirmado is not None:
                    modo_actual = modo_confirmado

                if modo_actual == "RATON":
                    color_hud = (0, 255, 128)

                    dist_pinch = math.hypot(idx_pos[0] - pulgar_pos[0], idx_pos[1] - pulgar_pos[1])
                    ratio_pinch = dist_pinch / tamano_palma

                    if ratio_pinch > PRE_PINCH:
                        target_x = np.interp(idx_pos[0], [MARGEN_X, w - MARGEN_X], [0, ANCHO_PANTALLA])
                        target_y = np.interp(idx_pos[1], [MARGEN_TOP, h - MARGEN_BOTTOM], [0, ALTO_PANTALLA])
                        sm_x, sm_y = filtro_raton.actualizar(target_x, target_y)
                        pyautogui.moveTo(sm_x, sm_y)
                    else:
                        # Congela el cursor en la posición antes de hacer clic
                        cv2.circle(frame, idx_pos, 8, (0, 255, 255), -1)

                    if ratio_pinch < CLICK_DOWN:
                        if not haciendo_clic:
                            pyautogui.click()
                            haciendo_clic = True
                        texto_estado = "Clic"
                        cv2.circle(frame, idx_pos, 14, (0, 255, 0), -1)
                    elif ratio_pinch > CLICK_UP:
                        haciendo_clic = False
                        texto_estado = "Moviendo cursor"

                    cv2.line(frame, idx_pos, pulgar_pos, (255, 0, 255), 2)

                elif modo_actual == "SCROLL":
                    color_hud = (255, 200, 0)
                    centro_y = MARGEN_TOP + (h - MARGEN_TOP - MARGEN_BOTTOM) // 2
                    dy = idx_pos[1] - centro_y

                    cv2.line(frame, (MARGEN_X, centro_y), (w - MARGEN_X, centro_y), (0, 255, 255), 1)

                    if abs(dy) > 30:
                        velocidad = int(-dy / 15)
                        pyautogui.scroll(velocidad)
                        texto_estado = f"Scroll ({velocidad})"
                    else:
                        texto_estado = "Scroll (centro)"

                elif modo_actual == "VOLUMEN":
                    color_hud = (0, 200, 255)
                    vol_target = int(np.interp(idx_pos[1], [MARGEN_TOP, h - MARGEN_BOTTOM], [100, 0]))

                    if tiempo_actual - ultimo_tiempo_vol > 0.12:
                        if vol_target > 60:
                            pyautogui.press("volumeup")
                        elif vol_target < 40:
                            pyautogui.press("volumedown")
                        ultimo_tiempo_vol = tiempo_actual

                    texto_estado = f"Volumen objetivo: {vol_target}%"

                elif modo_actual == "REPOSO":
                    texto_estado = "Pausa"

                dibujar_esqueleto(frame, lm, w, h)

            # Guía del marco activo
            cv2.rectangle(frame, (MARGEN_X, MARGEN_TOP), (w - MARGEN_X, h - MARGEN_BOTTOM), (0, 255, 255), 1)

            # Panel superior de estado
            cv2.rectangle(frame, (10, 10), (350, 75), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (350, 75), color_hud, 2)
            cv2.putText(frame, f"Modo: {modo_actual}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Estado: {texto_estado}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_hud, 1)

            cv2.imshow(nombre_ventana, frame)

            # Salida con 'q' o cerrando la ventana desde la X
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()