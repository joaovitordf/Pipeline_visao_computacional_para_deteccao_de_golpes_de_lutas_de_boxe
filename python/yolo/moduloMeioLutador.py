import numpy as np
import cv2

def draw_rectangle(image, center, width, height, color=(0, 255, 0), thickness=2):
    x1 = int(center[0] - width / 2)
    y1 = int(center[1] - height / 2)

    x2 = int(center[0] + width / 2)
    y2 = int(center[1] + height / 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image

def ponto_central(A, B, C, D):
    pontos_quadrilatero = [A, B, C, D]

    pontos_array = np.array(pontos_quadrilatero)

    centroide = np.mean(pontos_array, axis=0)

    return centroide

def moduloMeioLutadores(frame_count, lutador1, lutador2):
    # Ombro esquerdo 5
    #continua daqui
    #print(lutador1.nose)

    # Ombro direito 6

    # Cintura esquerda 11

    # Cintura direita 12

    #print(lut1)
    pass
