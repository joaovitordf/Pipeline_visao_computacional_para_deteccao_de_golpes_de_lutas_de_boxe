import cv2

from python.yolo.moduloMeioLutador import moduloMeioLutadores


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


def automatoColisao(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count):

    # Atravessou roi da cabeca ou corpo

    # Recebe as coordenadas se forem validas continua na funcao
    moduloMeioLutadores(frame_count, lutador1, lutador2)

    keypoints = results[0].keypoints

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
        if not colisao(r1, r2) and lutador1.roi_mao_esquerdaCabeca:
            lutador1.roi_mao_esquerdaCabeca = False
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
            lutador2.soco()

    r1 = None
    r2 = None

    # ----------------------------------------------------------------------------------------

    frame_lutador[frame_count].update({'lutador_1': lutador1})
    frame_lutador[frame_count].update({'lutador_2': lutador2})
