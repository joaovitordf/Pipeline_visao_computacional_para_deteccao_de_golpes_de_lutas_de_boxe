import numpy as np
import cv2

from python.yolo.moduloDefineCoordenadas import nose_coordenadas


def roi_cabeca(imagem, keypoints_numpy):
    (x, y) = nose_coordenadas(imagem, keypoints_numpy)
    # nose: 0

    if x != 0 and y != 0:

        start_point = (x-30, y-30)
        end_point = (x+30, y+30)

        roiCabeca = (start_point, end_point)
        return roiCabeca
    else:
        return None


def roi_mao_esquerda(imagem, keypoints_numpy):
    # left-writs: 9
    x = int(keypoints_numpy[9][0] * imagem.shape[1])

    y = int(keypoints_numpy[9][1] * imagem.shape[0])

    if x != 0 and y != 0:

        start_point = (x-50, y-30)
        end_point = (x+50, y+30)

        maoCoord = (start_point, end_point)

        return maoCoord
    else:
        return None


def roi_mao_direita(imagem, keypoints_numpy):
    # right-writs: 10
    x = int(keypoints_numpy[10][0] * imagem.shape[1])

    y = int(keypoints_numpy[10][1] * imagem.shape[0])

    if x != 0 and y != 0:
        start_point = (x - 50, y - 30)
        end_point = (x + 50, y + 30)

        maoCoord = (start_point, end_point)

        return maoCoord
    else:
        return None

def roi_linha_cintura(imagem, keypoints_numpy):
    # left-hip: 11
    # right-hip: 12
    x1 = int(keypoints_numpy[11][0] * imagem.shape[1])

    y1 = int(keypoints_numpy[11][1] * imagem.shape[0])

    x2 = int(keypoints_numpy[12][0] * imagem.shape[1])

    y2 = int(keypoints_numpy[12][1] * imagem.shape[0])

    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
        # Modifiquei o X pois em alguns frames a linha fica muito pequena
        aux_min_x = min(x1, x2)

        aux_min_x -= 80

        aux_max_x = max(x1, x2)

        aux_max_x += 80

        start_point = (aux_min_x, min(y1, y2))
        end_point = (aux_max_x, max(y1, y2))



        roi_linha_cinturaCoord = (start_point, end_point)

        return roi_linha_cinturaCoord
    else:
        return None

def roi_tronco(imagem, keypoints_numpy):
    # right-shoulder: 6
    # left-hip: 11
    x1 = int(keypoints_numpy[6][0] * imagem.shape[1])

    y1 = int(keypoints_numpy[6][1] * imagem.shape[0])

    x2 = int(keypoints_numpy[11][0] * imagem.shape[1])

    y2 = int(keypoints_numpy[11][1] * imagem.shape[0])

    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
        aux_min_x = min(x1, x2)

        aux_max_x = max(x1, x2)

        # Preciso ter um tamanho minimo
        if aux_max_x - aux_min_x <= 40:
            aux_max_x += 30

        start_point = (aux_min_x, min(y1, y2))
        end_point = (aux_max_x, max(y1, y2))

        troncoCoord = (start_point, end_point)

        return troncoCoord
    else:
        return None