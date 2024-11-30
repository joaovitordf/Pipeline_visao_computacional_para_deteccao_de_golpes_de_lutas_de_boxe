import cv2
from yolo.roiParts import cabecaCoordenadas, troncoCoordenadas, maoEsquerda, maoDireita


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
    lut1 = frame_lutador[frame_count]["lutador_1"]
    lut2 = frame_lutador[frame_count]["lutador_2"]
    keypoints = results[0].keypoints

    # ----------------------------- Possivel ataque do lutador 1 -----------------------------
    r1 = None
    r2 = None

    # Golpe de mao esquerda do lutador 1 no lutador 2
    if lut1.maoEsquerdaCoord is not None:
        (x1, y1), (x2, y2) = lut1.maoEsquerdaCoord
        r1 = x1, y1, x2, y2

    if lut2.cabeca is not None:
        (x1, y1), (x2, y2) = lut2.cabeca
        r2 = x1, y1, x2, y2

    if r1 is not None and r2 is not None:
        if colisao(r1, r2):
            lut1.maoEsquerdaCabeca = True
        if not colisao(r1, r2) and lut1.maoEsquerdaCabeca:
            lut1.maoEsquerdaCabeca = False
            lut1.soco()

    r1 = None

    # Golpe de mao direita do lutador 1 no lutador 2
    if lut1.maoDireitaCoord is not None:
        (x1, y1), (x2, y2) = lut1.maoDireitaCoord
        r1 = x1, y1, x2, y2

    if r1 is not None and r2 is not None:
        if colisao(r1, r2):
            lut1.maoDireitaCabeca = True
        if not colisao(r1, r2) and lut1.maoDireitaCabeca:
            lut1.maoDireitaCabeca = False
            lut1.soco()

    # ----------------------------------------------------------------------------------------

    # ----------------------------- Possivel ataque do lutador 2 -----------------------------
    r1 = None
    r2 = None

    if lut2.maoEsquerdaCoord is not None:
        (x1, y1), (x2, y2) = lut2.maoEsquerdaCoord
        r1 = x1, y1, x2, y2

    if lut1.cabeca is not None:
        (x1, y1), (x2, y2) = lut1.cabeca
        r2 = x1, y1, x2, y2

    if r1 is not None and r2 is not None:
        if colisao(r1, r2):
            lut2.maoEsquerdaCabeca = True
        if not colisao(r1, r2) and lut2.maoEsquerdaCabeca:
            lut2.maoEsquerdaCabeca = False
            lut2.soco()

    r1 = None

    if lut2.maoDireitaCoord is not None:
        (x1, y1), (x2, y2) = lut2.maoDireitaCoord
        r1 = x1, y1, x2, y2

    if lut1.cabeca is not None:
        (x1, y1), (x2, y2) = lut1.cabeca
        r2 = x1, y1, x2, y2

    if r1 is not None and r2 is not None:
        if colisao(r1, r2):
            lut2.maoDireitaCabeca = True
        if not colisao(r1, r2) and lut2.maoDireitaCabeca:
            lut2.maoDireitaCabeca = False
            lut2.soco()

    r1 = None
    r2 = None

    # ----------------------------------------------------------------------------------------

    frame_lutador[frame_count].update({'lutador_1': lut1})
    frame_lutador[frame_count].update({'lutador_2': lut2})
