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

from python.yolo.moduloMeioLutador import moduloMeioLutadores, moduloMeioLutadoresOtimizado, calcular_distancia

model = YOLO("pesos/yolov8x-seg.pt")

def colisao(roi1, roi2, distancia_max_ponto=20.0):

    if roi1 is None or roi2 is None:
        return False

    # caso 1: bbox como (x1,y1,x2,y2)
    if hasattr(roi1, '__len__') and len(roi1) == 4 and len(roi2) == 4:
        x11, y11, x12, y12 = roi1
        x21, y21, x22, y22 = roi2
        result = not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)
        return result

    # caso 2: bbox como dois pontos (top-left, bottom-right)
    if (hasattr(roi1, '__len__') and len(roi1) == 2
        and hasattr(roi1[0], '__len__') and len(roi1[0]) == 2):
        x11, y11 = roi1[0]
        x12, y12 = roi1[1]
        x21, y21 = roi2[0]
        x22, y22 = roi2[1]
        result = not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)
        return result

    # caso 3: compara como pontos
    dist = calcular_distancia(roi1, roi2)
    result = dist <= distancia_max_ponto
    return result

def segmentar_imagem(imagem, conf):
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

    if lutador1.distancia is not None and lutador1.distancia:

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

        # ----------------------------- Possível ataque no tronco -----------------------------
        r1 = None
        r2 = None

        # Golpe de mão esquerda do lutador 1 no tronco do lutador 2
        if lutador1.roi_mao_esquerda is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_esquerda
            r1 = x1, y1, x2, y2

        if lutador2.roi_tronco is not None:
            (x1, y1), (x2, y2) = lutador2.roi_tronco
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador1.roi_mao_esquerdaTronco = True
            if not colisao(r1, r2) and lutador1.roi_mao_esquerdaTronco:
                lutador1.roi_mao_esquerdaTronco = False
                print(lutador1.distancia)
                lutador1.soco()

        r1 = None
        r2 = None

        # Golpe de mão direita do lutador 1 no tronco do lutador 2
        if lutador1.roi_mao_direita is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_direita
            r1 = x1, y1, x2, y2

        if lutador2.roi_tronco is not None:
            (x1, y1), (x2, y2) = lutador2.roi_tronco
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador1.roi_mao_direitaTronco = True
            if not colisao(r1, r2) and lutador1.roi_mao_direitaTronco:
                lutador1.roi_mao_direitaTronco = False
                print(lutador1.distancia)
                lutador1.soco()

        r1 = None
        r2 = None

        # Golpe de mão esquerda do lutador 2 no tronco do lutador 1
        if lutador2.roi_mao_esquerda is not None:
            (x1, y1), (x2, y2) = lutador2.roi_mao_esquerda
            r1 = x1, y1, x2, y2

        if lutador1.roi_tronco is not None:
            (x1, y1), (x2, y2) = lutador1.roi_tronco
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador2.roi_mao_esquerdaTronco = True
            if not colisao(r1, r2) and lutador2.roi_mao_esquerdaTronco:
                lutador2.roi_mao_esquerdaTronco = False
                print(lutador1.distancia)
                lutador2.soco()

        r1 = None
        r2 = None

        # Golpe de mão direita do lutador 2 no tronco do lutador 1
        if lutador2.roi_mao_direita is not None:
            (x1, y1), (x2, y2) = lutador2.roi_mao_direita
            r1 = x1, y1, x2, y2

        if lutador1.roi_tronco is not None:
            (x1, y1), (x2, y2) = lutador1.roi_tronco
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador2.roi_mao_direitaTronco = True
            if not colisao(r1, r2) and lutador2.roi_mao_direitaTronco:
                lutador2.roi_mao_direitaTronco = False
                print(lutador1.distancia)
                lutador2.soco()
        # ----------------------------------------------------------------------------------------

        frame_lutador[frame_count].update({'lutador_1': lutador1})
        frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame


def automatoColisaoOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count, frame_original=None):
    frame = moduloMeioLutadoresOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count)

    ataques = [
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_cabeca', 'esquerda_cabeca', lutador1.soco),
        (lutador1, 'roi_mao_direita', lutador2, 'roi_cabeca', 'direita_cabeca', lutador1.soco),
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_tronco', 'esquerda_tronco', lutador1.soco),
        (lutador1, 'roi_mao_direita', lutador2, 'roi_tronco', 'direita_tronco', lutador1.soco),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_cabeca', 'esquerda_cabeca', lutador2.soco),
        (lutador2, 'roi_mao_direita', lutador1, 'roi_cabeca', 'direita_cabeca', lutador2.soco),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_tronco', 'esquerda_tronco', lutador2.soco),
        (lutador2, 'roi_mao_direita', lutador1, 'roi_tronco', 'direita_tronco', lutador2.soco),
    ]

    for atacante, mao_attr, defensor, parte_attr, flag_suffix, metodo_soco in ataques:
        r1 = getattr(atacante, mao_attr, None)
        r2 = getattr(defensor, parte_attr, None)
        if r1 is None or r2 is None:
            continue

        flag_name = f"{mao_attr}_{flag_suffix}"
        prev_flag = getattr(atacante, flag_name, False)

        if colisao(r1, r2):
            setattr(atacante, flag_name, True)
        else:
            if prev_flag:
                metodo_soco()
            setattr(atacante, flag_name, False)

    return frame
