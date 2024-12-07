import numpy as np
import cv2

def draw_circle(image, center, radius, color=(0, 255, 0), thickness=2):
    """
    Desenha um círculo no ponto central dado.
    :param image: Imagem onde o círculo será desenhado.
    :param center: Centro do círculo (x, y).
    :param radius: Raio do círculo.
    :param color: Cor do círculo (B, G, R).
    :param thickness: Espessura da borda. Use -1 para círculo sólido.
    :return: Imagem com o círculo desenhado.
    """
    center = (int(center[0]), int(center[1]))  # Converte as coordenadas para inteiros
    cv2.circle(image, center, radius, color, thickness)
    return image

def draw_line_between_centers(image, center1, center2, color=(0, 255, 0), thickness=2):
    """
    Desenha uma linha reta entre os pontos centrais de dois lutadores.
    :param image: Imagem onde a linha será desenhada.
    :param center1: Coordenadas (x, y) do centro do lutador 1.
    :param center2: Coordenadas (x, y) do centro do lutador 2.
    :param color: Cor da linha (B, G, R).
    :param thickness: Espessura da linha.
    :return: Imagem com a linha desenhada.
    """
    # Converte as coordenadas dos centros para inteiros
    pt1 = (int(center1[0]), int(center1[1]))
    pt2 = (int(center2[0]), int(center2[1]))

    # Desenha a linha entre os dois centros
    cv2.line(image, pt1, pt2, color, thickness)
    return image

def calcular_distancia(center1, center2):
    """
    Calcula a distância euclidiana entre dois pontos centrais.
    :param center1: Coordenadas (x, y) do primeiro ponto central.
    :param center2: Coordenadas (x, y) do segundo ponto central.
    :return: Distância entre os dois pontos.
    """
    # Converte os pontos para arrays NumPy para facilitar o cálculo
    ponto1 = np.array(center1)
    ponto2 = np.array(center2)

    # Calcula a distância euclidiana
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

    frame_lutador[frame_count].update({'lutador_1': lutador1})
    frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame
