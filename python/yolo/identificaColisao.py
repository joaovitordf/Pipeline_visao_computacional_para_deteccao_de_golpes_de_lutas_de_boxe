import cv2
from ultralytics import YOLO
import numpy as np

from python.yolo.moduloMeioLutador import moduloMeioLutadores, moduloMeioLutadoresOtimizado, calcular_distancia


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
    vertical_overlap = not (y2 < y_top or y1 > y_bottom)

    return horizontal_overlap and vertical_overlap


def automatoColisao(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count, frame_original):
    frame = moduloMeioLutadores(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)

    h, w = frame.shape[:2]
    # Distancia aceita para contar os golpes (baseado na resolucao da tela)
    limiar = w * 0.12  # 12% da largura da tela
    # print(limiar)
    if lutador1.distancia is not None and lutador1.distancia > limiar:
        # ----------------------------- Possivel ataque do lutador 1 -----------------------------
        # dentro do trecho de ataque do lutador1:
        processa_colisao_mao(
            atacante=lutador1,
            defensor=lutador2,
            roi_mao_attr='roi_mao_esquerda'
        )
        processa_colisao_mao(
            atacante=lutador1,
            defensor=lutador2,
            roi_mao_attr='roi_mao_direita'
        )

        # ----------------------------------------------------------------------------------------

        # ----------------------------- Possivel ataque do lutador 2 -----------------------------
        processa_colisao_mao(
            atacante=lutador2,
            defensor=lutador1,
            roi_mao_attr='roi_mao_esquerda'
        )
        processa_colisao_mao(
            atacante=lutador2,
            defensor=lutador1,
            roi_mao_attr='roi_mao_direita'
        )

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


def processa_colisao_mao(atacante, defensor, roi_mao_attr):
    roi_mao = getattr(atacante, roi_mao_attr, None)
    if roi_mao is None:
        return

    # regiões alvo
    alvos = [
        ('cabeca', defensor.roi_cabeca),
        ('tronco', defensor.roi_tronco),
    ]

    for parte, roi_alvo in alvos:
        if roi_alvo is None:
            continue

        (ax1, ay1), (ax2, ay2) = roi_mao
        (bx1, by1), (bx2, by2) = roi_alvo
        if colisao((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)):
            flag_attr = f"{roi_mao_attr}{parte.capitalize()}"
            # só dispara uma vez por contato
            if not getattr(atacante, flag_attr, False):
                setattr(atacante, flag_attr, True)
                atacante.soco(parte)
            return

    # quando termina o contato, resetamos flags
    for parte in ('Cabeca', 'Tronco'):
        flag_attr = f"{roi_mao_attr}{parte}"
        if getattr(atacante, flag_attr, False):
            setattr(atacante, flag_attr, False)


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
        # Calcula o ponto médio vertical da mão (metade superior)
        mid_y = (ay1 + ay2) / 2
        # Região de interseção
        inter_x1 = max(ax1, dx1)
        inter_y1 = max(ay1, dy1)
        inter_x2 = min(ax2, dx2)
        inter_y2 = min(ay2, dy2)

        # Só considera golpe irregular se a interseção estiver na metade superior da mão
        if inter_y1 < mid_y:
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
    frame = moduloMeioLutadoresOtimizado(
        frame, results, lutador1, lutador2, frame_lutador, frame_count
    )

    ataques = [
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_cabeca', 'esquerda_cabeca', 'cabeca'),
        (lutador1, 'roi_mao_direita', lutador2, 'roi_cabeca', 'direita_cabeca', 'cabeca'),
        (lutador1, 'roi_mao_esquerda', lutador2, 'roi_tronco', 'esquerda_tronco', 'tronco'),
        (lutador1, 'roi_mao_direita', lutador2, 'roi_tronco', 'direita_tronco', 'tronco'),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_cabeca', 'esquerda_cabeca', 'cabeca'),
        (lutador2, 'roi_mao_direita', lutador1, 'roi_cabeca', 'direita_cabeca', 'cabeca'),
        (lutador2, 'roi_mao_esquerda', lutador1, 'roi_tronco', 'esquerda_tronco', 'tronco'),
        (lutador2, 'roi_mao_direita', lutador1, 'roi_tronco', 'direita_tronco', 'tronco'),
    ]

    for atacante, mao_attr, defensor, parte_attr, flag_suf, parte in ataques:
        r1 = getattr(atacante, mao_attr, None)
        r2 = getattr(defensor, parte_attr, None)
        if not (r1 and r2):
            continue

        flag_name = f"{mao_attr}_{flag_suf}"
        prev_flag = getattr(atacante, flag_name, False)

        if colisao(r1, r2):
            # ainda em contato: seta flag
            setattr(atacante, flag_name, True)

        elif prev_flag:
            # contato terminou: dispara soco com a parte correta
            atacante.soco(parte)
            setattr(atacante, flag_name, False)

    # ----------------------------- Golpe irregular abaixo da cintura -----------------------------
    checa_golpe_irregular(lutador1, lutador2, 'roi_mao_esquerda', 'roi_linha_cintura')
    checa_golpe_irregular(lutador1, lutador2, 'roi_mao_direita', 'roi_linha_cintura')

    checa_golpe_irregular(lutador2, lutador1, 'roi_mao_esquerda', 'roi_linha_cintura')
    checa_golpe_irregular(lutador2, lutador1, 'roi_mao_direita', 'roi_linha_cintura')
    # ---------------------------------------------------------------------------------------------

    # atualiza histórico de quadros
    frame_lutador[frame_count]['lutador_1'] = lutador1
    frame_lutador[frame_count]['lutador_2'] = lutador2

    return frame
