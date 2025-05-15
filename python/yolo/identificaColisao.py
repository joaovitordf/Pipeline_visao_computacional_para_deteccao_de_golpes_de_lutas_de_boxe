import cv2
from ultralytics import YOLO
import numpy as np


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


def verificaGolpeIrregular(roi_golpe, roi_linha_cintura):
    if roi_golpe is None or roi_linha_cintura is None:
        return False

    # Coordenadas do golpe
    x1, y1, x2, y2 = roi_golpe

    # Coordenadas da região da linha de cintura
    (x_left, y_top), (x_right, y_bottom) = roi_linha_cintura

    # Verifica se há interseção entre os dois retângulos
    horizontal_overlap = not (x2 < x_left or x1 > x_right)
    vertical_overlap   = not (y2 < y_top  or y1 > y_bottom)

    return horizontal_overlap and vertical_overlap


def automatoColisao(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count, frame_original):
    frame = moduloMeioLutadores(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)

    if lutador1.distancia is not None and lutador1.distancia:

        # ----------------------------- Possivel ataque do lutador 1 -----------------------------
        r1 = r2 = None

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

            if not colisao(r1, r2) and lutador1.roi_mao_esquerdaCabeca:
                lutador1.roi_mao_esquerdaCabeca = False
                lutador1.soco()

        r1 = r2 = None

        # Golpe de mao direita do lutador 1 no lutador 2
        if lutador1.roi_mao_direita is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_direita
            r1 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if colisao(r1, r2):
                lutador1.roi_mao_direitaCabeca = True
            if not colisao(r1, r2) and lutador1.roi_mao_direitaCabeca:
                lutador1.roi_mao_direitaCabeca = False
                lutador1.soco()

        # ----------------------------------------------------------------------------------------

        # ----------------------------- Possivel ataque do lutador 2 -----------------------------
        r1 = r2 = None

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
                lutador2.soco()

        r1 = r2 = None

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
                lutador2.soco()

        # ----------------------------------------------------------------------------------------

        # ----------------------------- Possível ataque no tronco -----------------------------
        r1 = r2 = None

        # Golpe de mão esquerda do lutador 1 no tronco do lutador 2
        if lutador1.roi_mao_esquerda is not None:
            (x1, y1), (x2, y2) = lutador1.roi_mao_esquerda
            r1 = x1, y1, x2, y2

        if lutador2.roi_tronco is not None:
            (x1, y1), (x2, y2) = lutador2.roi_tronco
            r2 = x1, y1, x2, y2

        if r1 is not None and r2 is not None:
            if not colisao(r1, r2) and lutador1.roi_mao_esquerdaTronco:
                lutador1.roi_mao_esquerdaTronco = False
                lutador1.soco()

        r1 = r2 = None

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
            elif lutador1.roi_mao_direitaTronco:
                # contato terminou
                lutador1.roi_mao_direitaTronco = False
                lutador1.soco()

        r1 = r2 = None

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
            elif lutador2.roi_mao_esquerdaTronco:
                # contato terminou
                lutador2.roi_mao_esquerdaTronco = False
                lutador2.soco()

        r1 = r2 = None

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
            elif lutador2.roi_mao_direitaTronco:
                # contato terminou
                lutador2.roi_mao_direitaTronco = False
                lutador2.soco()
        # ----------------------------------------------------------------------------------------

        # ----------------------------- Possível golpe irregular -----------------------------

        checa_golpe_irregular(
            lutador1, lutador2,
            roi_mao_attr="roi_mao_direita",
            roi_alvo_attr="roi_linha_cintura"
        )

        checa_golpe_irregular(
            lutador1, lutador2,
            roi_mao_attr="roi_mao_esquerda",
            roi_alvo_attr="roi_linha_cintura"
        )

        checa_golpe_irregular(
            lutador2, lutador1,
            roi_mao_attr="roi_mao_direita",
            roi_alvo_attr="roi_linha_cintura"
        )

        checa_golpe_irregular(
            lutador2, lutador1,
            roi_mao_attr="roi_mao_esquerda",
            roi_alvo_attr="roi_linha_cintura"
        )

        # ----------------------------------------------------------------------------------------

        frame_lutador[frame_count].update({'lutador_1': lutador1})
        frame_lutador[frame_count].update({'lutador_2': lutador2})

    return frame


def checa_golpe_irregular(atacante, defensor, roi_mao_attr, roi_alvo_attr):
    roi_mao = getattr(atacante, roi_mao_attr, None)
    roi_alvo = getattr(defensor, roi_alvo_attr, None)
    if roi_mao is None or roi_alvo is None:
        return

    (ax1, ay1), (ax2, ay2) = roi_mao
    (dx1, dy1), (dx2, dy2) = roi_alvo
    r_atk = (ax1, ay1, ax2, ay2)
    r_def = (dx1, dy1, dx2, dy2)

    # colisão detectada
    if colisao(r_atk, r_def):
        # -- esquerda --
        if roi_mao_attr == 'roi_mao_esquerda':
            if not getattr(atacante, 'golpe_irregularEsquerda', False):
                atacante.golpe_irregularEsquerda = True
                print(f"Golpe irregular de lutador {atacante.identificador}! (esquerda)")
                atacante.falta()

        # -- direita --
        elif roi_mao_attr == 'roi_mao_direita':
            if not getattr(atacante, 'golpe_irregularDireita', False):
                atacante.golpe_irregularDireita = True
                print(f"Golpe irregular de lutador {atacante.identificador}! (direita)")
                atacante.falta()

    else:
        # quando termina o contato, limpamos as flags
        if roi_mao_attr == 'roi_mao_esquerda' and getattr(atacante, 'golpe_irregularEsquerda', False):
            atacante.golpe_irregularEsquerda = False

        elif roi_mao_attr == 'roi_mao_direita' and getattr(atacante, 'golpe_irregularDireita', False):
            atacante.golpe_irregularDireita = False


def automatoColisaoOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count, frame_original=None):
    frame = moduloMeioLutadoresOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count)

    ataques = [
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_cabeca', 'esquerda_cabeca'),
        (lutador1, 'roi_mao_direita',  lutador2, 'roi_cabeca', 'direita_cabeca'),
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_tronco','esquerda_tronco'),
        (lutador1, 'roi_mao_direita',  lutador2, 'roi_tronco','direita_tronco'),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_cabeca', 'esquerda_cabeca'),
        (lutador2, 'roi_mao_direita',  lutador1, 'roi_cabeca', 'direita_cabeca'),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_tronco','esquerda_tronco'),
        (lutador2, 'roi_mao_direita',  lutador1, 'roi_tronco','direita_tronco'),
    ]

    for atacante, mao_attr, defensor, parte_attr, flag_suffix in ataques:
        r1 = getattr(atacante, mao_attr, None)
        r2 = getattr(defensor, parte_attr, None)
        if not (r1 and r2):
            continue

        flag_name = f"{mao_attr}_{flag_suffix}"
        prev_flag = getattr(atacante, flag_name, False)

        if colisao(r1, r2):
            # contato continua
            setattr(atacante, flag_name, True)
        elif prev_flag:
            # contato terminou: decide soco
            atacante.soco()
            setattr(atacante, flag_name, False)

    # ----------------------------- Golpe irregular abaixo da cintura -----------------------------
    checa_golpe_irregular(lutador1, lutador2, 'roi_mao_esquerda', 'roi_linha_cintura')
    checa_golpe_irregular(lutador1, lutador2, 'roi_mao_direita',  'roi_linha_cintura')

    checa_golpe_irregular(lutador2, lutador1, 'roi_mao_esquerda', 'roi_linha_cintura')
    checa_golpe_irregular(lutador2, lutador1, 'roi_mao_direita',  'roi_linha_cintura')
    # ---------------------------------------------------------------------------------------------

    # atualiza histórico de quadros
    frame_lutador[frame_count]['lutador_1'] = lutador1
    frame_lutador[frame_count]['lutador_2'] = lutador2

    return frame
