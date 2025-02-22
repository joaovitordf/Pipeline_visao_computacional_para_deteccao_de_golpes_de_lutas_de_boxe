import cv2
import os

from ultralytics import YOLO
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from skimage.exposure import histogram, cumulative_distribution
from skimage.filters import threshold_otsu

from python.yolo.moduloMeioLutador import moduloMeioLutadores

model = YOLO("pesos/yolov8x-seg.pt")

def colisao(r1, r2):
    # r1 e r2 são tuplas (x1, y1, x2, y2)
    x1_min, y1_min, x1_max, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2

    # Verifica se um retângulo está à esquerda do outro
    if x1_max < x2_min or x2_max < x1_min:
        return False

    # Verifica se um retângulo está acima do outro
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True

def segmentar_imagem(imagem, conf):
    """
    Segmenta uma imagem usando YOLOv8 e retorna a imagem segmentada.

    :param imagem: Imagem de entrada (numpy array).
    :param conf: Limite de confiança para detecção.
    :return: Imagem segmentada (numpy array com máscara binária).
    """
    # Cria uma máscara preta com o mesmo tamanho da imagem
    background = np.zeros_like(imagem)

    # Realiza as previsões de segmentação
    results = model.predict(imagem, conf=conf, verbose=True, device="cuda")

    # Preenche os objetos detectados com a cor branca
    white_color = (255, 255, 255)
    for result in results:
        for mask in result.masks.xy:
            points = np.int32([mask])
            cv2.fillPoly(background, points, white_color)

    return background

def automatoColisao(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count, frame_original):

    # Atravessou roi da cabeca ou corpo

    # Recebe as coordenadas se forem validas continua na funcao
    frame = moduloMeioLutadores(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)

    #keypoints = results[0].keypoints

    if lutador1.distancia is not None and lutador1.distancia > 220:

        # ----------------------------- Possivel ataque do lutador 1 -----------------------------
        r1 = None
        r2 = None

        # Golpe de mao esquerda do lutador 1 no lutador 2
        if lutador1.roi_mao_esquerda is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_esquerda
            r1 = x1, y1, x2, y2

        if lutador2.roi_cabeca is not None:
            (x1, y1), (x2, y2) = lutador2.roi_cabeca
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador1.roi_mao_esquerdaCabeca = True

                x1, y1, x2, y2 = map(int, r1)
                recorte = frame_original[y1:y2, x1:x2]

                """sail_hsv = rgb2hsv(recorte)

                # Definindo um valor de limiar no canal de saturação ou valor (V)
                th = 0.4
                sail_gray_bw = sail_hsv[..., 1] < th  # Usando o canal de saturação (S) para a comparação

                # Converter a imagem binária (True/False) para 0/255 para exibição no OpenCV
                sail_gray_bw_display = (sail_gray_bw * 255).astype('uint8')

                # Mostrar a imagem binária
                cv2.imshow("Recorte e Recorte Segmentado", sail_gray_bw_display)
                #cv2.waitKey(0)"""

                """recorte_segmentado = segmentar_imagem(recorte, conf=0.2)

                altura = max(recorte.shape[0], recorte_segmentado.shape[0])
                largura = max(recorte.shape[1], recorte_segmentado.shape[1])

                recorte_redimensionado = cv2.resize(recorte, (largura, altura))
                recorte_segmentado_redimensionado = cv2.resize(recorte_segmentado, (largura, altura))

                imagem_combinada = np.vstack((recorte_redimensionado, recorte_segmentado_redimensionado))

                cv2.imshow("Recorte e Recorte Segmentado", imagem_combinada)
                cv2.waitKey(0)"""

            if not colisao(r1, r2) and lutador1.roi_mao_esquerdaCabeca:
                lutador1.roi_mao_esquerdaCabeca = False
                print(lutador1.distancia)
                lutador1.soco()

        r1 = None

        # Golpe de mao direita do lutador 1 no lutador 2
        if lutador1.roi_mao_direita is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_direita
            r1 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador1.roi_mao_direitaCabeca = True
            if not colisao(r1, r2) and lutador1.roi_mao_direitaCabeca:
                lutador1.roi_mao_direitaCabeca = False
                print(lutador1.distancia)
                lutador1.soco()

        # ----------------------------------------------------------------------------------------

        # ----------------------------- Possivel ataque do lutador 2 -----------------------------
        r1 = None
        r2 = None

        if lutador2.roi_mao_esquerda is not None:
            (x1, y1), (x2, y2) = lutador2.roi_mao_esquerda
            r1 = x1, y1, x2, y2

        if lutador1.roi_cabeca is not None:
            (x1, y1), (x2, y2) = lutador1.roi_cabeca
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador2.roi_mao_esquerdaCabeca = True
            if not colisao(r1, r2) and lutador2.roi_mao_esquerdaCabeca:
                lutador2.roi_mao_esquerdaCabeca = False
                print(lutador1.distancia)
                lutador2.soco()

        r1 = None

        if lutador2.roi_mao_direita is not None:
            (x1, y1), (x2, y2) = lutador2.roi_mao_direita
            r1 = x1, y1, x2, y2

        if lutador1.roi_cabeca is not None:
            (x1, y1), (x2, y2) = lutador1.roi_cabeca
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador2.roi_mao_direitaCabeca = True
            if not colisao(r1, r2) and lutador2.roi_mao_direitaCabeca:
                lutador2.roi_mao_direitaCabeca = False
                print(lutador1.distancia)
                lutador2.soco()

        r1 = None
        r2 = None

        # ----------------------------------------------------------------------------------------

        frame_lutador[frame_count].update({'lutador_1': lutador1})
        frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame