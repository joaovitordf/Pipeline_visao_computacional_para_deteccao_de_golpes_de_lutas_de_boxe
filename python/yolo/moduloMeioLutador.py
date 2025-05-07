import numpy as np
import cv2

def draw_circle(image, center, radius, color=(0, 255, 0), thickness=2):
    center = (int(center[0]), int(center[1]))
    cv2.circle(image, center, radius, color, thickness)
    return image

def draw_line_between_centers(image, center1, center2, color=(0, 255, 0), thickness=2):
    # Converte as coordenadas dos centros para inteiros
    pt1 = (int(center1[0]), int(center1[1]))
    pt2 = (int(center2[0]), int(center2[1]))

    # Desenha a linha entre os dois centros
    cv2.line(image, pt1, pt2, color, thickness)
    return image

def calcular_distancia(center1, center2):
    ponto1 = np.array(center1)
    ponto2 = np.array(center2)

    distancia = np.linalg.norm(ponto1 - ponto2)
    return distancia

def ponto_central(A, B, C, D):
    pontos_quadrilatero = [A, B, C, D]

    pontos_array = np.array(pontos_quadrilatero)

    centroide = np.mean(pontos_array, axis=0)

    return centroide

def moduloMeioLutadores(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count):
    # Calcula o ponto central do lutador 1
    center1 = ponto_central(lutador1.left_shoulder, lutador1.right_shoulder, lutador1.left_hip, lutador1.right_hip)
    frame = draw_circle(frame, center1, radius=4, color=(0, 255, 0), thickness=8)  # Círculo com raio 10

    # Calcula o ponto central do lutador 2
    center2 = ponto_central(lutador2.left_shoulder, lutador2.right_shoulder, lutador2.left_hip, lutador2.right_hip)
    frame = draw_circle(frame, center2, radius=4, color=(0, 255, 0), thickness=8)  # Círculo com raio 10

    # Desenha a linha entre os dois centros
    frame = draw_line_between_centers(frame, center1, center2, color=(0, 255, 255), thickness=3)

    # Calcula a distância entre os pontos centrais
    distancia = calcular_distancia(center1, center2)

    lutador1.distancia = distancia
    lutador2.distancia = distancia

    # Calcula ponto médio da linha
    mid_x = int((center1[0] + center2[0]) / 2)
    mid_y = int((center1[1] + center2[1]) / 2)

    # Formata o texto da distância (por ex. "123.4")
    texto = f"{distancia:.1f}"

    # Desenha o texto no ponto médio
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(texto, font, font_scale, thickness)

    # Ajusta para centralizar o texto no ponto médio
    text_w, text_h = text_size
    text_org = (mid_x - text_w // 2, mid_y - text_h // 2)

    # Cor de fundo para legibilidade (opcional)
    cv2.rectangle(frame,
                  (text_org[0] - 2, text_org[1] + 2),
                  (text_org[0] + text_w + 2, text_org[1] - text_h - 2),
                  (0, 0, 0),
                  thickness=cv2.FILLED)

    # Finalmente desenha o texto em branco
    cv2.putText(frame, texto, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    frame_lutador[frame_count].update({'lutador_1': lutador1})
    frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame
