import numpy as np
import cv2


def draw_circle(image, center, radius, color=(0, 255, 0), thickness=2):
    center = (int(center[0]), int(center[1]))
    cv2.circle(image, center, radius, color, thickness)
    return image


def draw_line_between_centers(image, center1, center2, color=(0, 255, 0), thickness=2):
    pt1 = (int(center1[0]), int(center1[1]))
    pt2 = (int(center2[0]), int(center2[1]))
    cv2.line(image, pt1, pt2, color, thickness)
    return image


def ponto_central(A, B, C, D):
    for p in (A, B, C, D):
        if p is None:
            return None
    pts = np.array([A, B, C, D], dtype=float)
    return tuple(np.mean(pts, axis=0))


def calcular_distancia(center1, center2):
    if center1 is None or center2 is None:
        return 0.0
    return float(np.linalg.norm(np.array(center1) - np.array(center2)))


def colisao(roi1, roi2, distancia_max_ponto=20.0):
    if roi1 is None or roi2 is None:
        return False

    # Se tiver 4 valores, trato como bbox
    if hasattr(roi1, '__len__') and len(roi1) == 4 and len(roi2) == 4:
        x11, y11, x12, y12 = roi1
        x21, y21, x22, y22 = roi2
        return not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)

    # Senao, trato como ponto
    dist = calcular_distancia(roi1, roi2)
    return dist <= distancia_max_ponto


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

    texto = f"{distancia:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(texto, font, font_scale, thickness)
    padding = 8
    text_org = (mid_x - text_w // 2, mid_y + text_h + padding)

    cv2.putText(frame, texto, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    frame_lutador[frame_count].update({'lutador_1': lutador1})
    frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame


def moduloMeioLutadoresOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count):
    frame_lutador.setdefault(frame_count, {})

    c1 = ponto_central(
        lutador1.left_shoulder, lutador1.right_shoulder,
        lutador1.left_hip, lutador1.right_hip
    )
    c2 = ponto_central(
        lutador2.left_shoulder, lutador2.right_shoulder,
        lutador2.left_hip, lutador2.right_hip
    )

    if c1 is not None:
        frame = draw_circle(frame, c1, radius=4, color=(0, 255, 0), thickness=8)
    if c2 is not None:
        frame = draw_circle(frame, c2, radius=4, color=(0, 255, 0), thickness=8)
    if c1 is not None and c2 is not None:
        frame = draw_line_between_centers(frame, c1, c2, color=(0, 255, 255), thickness=3)

    dist = calcular_distancia(c1, c2)
    lutador1.distancia = dist
    lutador2.distancia = dist

    if c1 is not None and c2 is not None:
        mid_x = (c1[0] + c2[0]) / 2
        mid_y = (c1[1] + c2[1]) / 2
        txt = f"{dist:.1f}"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, thh), _ = cv2.getTextSize(txt, font, fs, th)
        org = (int(mid_x - tw / 2), int(mid_y + thh / 2))
        cv2.rectangle(frame,
                      (org[0] - 2, org[1] + 2),
                      (org[0] + tw + 2, org[1] - thh - 2),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, txt, org, font, fs, (255, 255, 255), th, cv2.LINE_AA)

    frame_lutador[frame_count]['lutador_1'] = lutador1
    frame_lutador[frame_count]['lutador_2'] = lutador2

    return frame
