import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from yolo.clusteriza import DominantColors, clusterizaFunction
from yolo.identificaColisao import colisao
from yolo.identificaGolpe import golpe

from yolo.roiParts import linhaCintura, maoEsquerda, maoDireita, troncoCoordenadas

from python.yolo.roiParts import cabecaCoordenadas


def pernaCoordenadas(imagem, keypoints):
    x1 = int(keypoints[12][0] * imagem.shape[1])

    y1 = int(keypoints[12][1] * imagem.shape[0])

    x2 = int(keypoints[13][0] * imagem.shape[1])

    y2 = int(keypoints[13][1] * imagem.shape[0])

    if x1 > x2:
        coordenada_start_x = x2
        coordenada_end_x = x1
    else:
        coordenada_start_x = x1
        coordenada_end_x = x2

    if y1 > y2:
        coordenada_start_y = y2
        coordenada_end_y = y1
    else:
        coordenada_start_y = y1
        coordenada_end_y = y2

    #print(coordenada_start_y, coordenada_end_y)
    #print(coordenada_start_x, coordenada_end_x)

    return [coordenada_start_x, coordenada_end_x, coordenada_start_y, coordenada_end_y]


def define_lutador(lutador1, lutador2, cor, tolerancia=50):
    lut1 = np.abs(cor - lutador1.cor)
    lut2 = np.abs(cor - lutador2.cor)

    media_lut1 = (lut1[0] + lut1[1] + lut1[2]) / 3
    media_lut2 = (lut2[0] + lut2[1] + lut2[2]) / 3

    if media_lut1 <= tolerancia:
        return 1
    elif media_lut2 <= tolerancia:
        return 2
    else:
        return None


def boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count):
    if lutador1.cor is None and lutador2.cor is None and len(cores) == 2:
        lutador1.cor = cores[0]
        lutador2.cor = cores[1]

    coordenada_corte = list()
    keypoints = results[0].keypoints
    for pessoa in keypoints:
        keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
        #draw_boundingBox(imagem, keypoints_numpy)
        coord = pernaCoordenadas(frame, keypoints_numpy)
        coordenada_corte.append(coord)
        recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
        teste = DominantColors(recorte, 1)
        cor = teste.dominantColors()
        identifica_lutador = define_lutador(lutador1, lutador2, cor[0])
        if identifica_lutador == 1:
            lutador1.identificador = 1
            cabeca = cabecaCoordenadas(frame, keypoints_numpy)
            tronco = troncoCoordenadas(frame, keypoints_numpy)
            linhasCinturas = linhaCintura(frame, keypoints_numpy)
            maoEsquerdaCoord = maoEsquerda(frame, keypoints_numpy)
            maoDireitaCoord = maoDireita(frame, keypoints_numpy)
            lutador1.cabeca = cabeca
            lutador1.tronco = tronco
            lutador1.linhasCinturas = linhasCinturas
            lutador1.maoEsquerdaCoord = maoEsquerdaCoord
            lutador1.maoDireitaCoord = maoDireitaCoord
            frame_lutador[frame_count].update({'lutador_1': lutador1})
        elif identifica_lutador == 2:
            lutador2.identificador = 2
            cabeca = cabecaCoordenadas(frame, keypoints_numpy)
            tronco = troncoCoordenadas(frame, keypoints_numpy)
            linhasCinturas = linhaCintura(frame, keypoints_numpy)
            maoEsquerdaCoord = maoEsquerda(frame, keypoints_numpy)
            maoDireitaCoord = maoDireita(frame, keypoints_numpy)
            lutador2.cabeca = cabeca
            lutador2.tronco = tronco
            lutador2.linhasCinturas = linhasCinturas
            lutador2.maoEsquerdaCoord = maoEsquerdaCoord
            lutador2.maoDireitaCoord = maoDireitaCoord
            lutador2.coordenadas = keypoints_numpy
            frame_lutador[frame_count].update({'lutador_2': lutador2})

    annotated_frame = results[0].plot(boxes=False)
    for r in results:
        annotator = Annotator(annotated_frame)
        boxes = r.boxes
        contador = 0
        areaLutador = list()

        for box in boxes:
            #print(keypoints)
            coord = coordenada_corte[contador]
            #print(x)
            #print(y)
            recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
            teste = DominantColors(recorte, 1)
            cor = teste.dominantColors()

            identifica_lutador = define_lutador(lutador1, lutador2, cor[0])
            if identifica_lutador == 1:
                lutador1.identificador = 1
                lutador1.box = box
                frame_lutador[frame_count].update({'lutador_1': lutador1})
            elif identifica_lutador == 2:
                lutador2.identificador = 2
                lutador2.box = box
                frame_lutador[frame_count].update({'lutador_2': lutador2})
            #frame_lutador[frame_count].update({'lutador_id': identifica_lutador, 'coordenada':})

            #print(identifica_lutador)
            b = box.xyxy[0]
            box = b.tolist()
            areaLutador.append(box)

            label_lutador = "Lutador " + str(identifica_lutador)
            annotator.box_label(b, label_lutador, color=(0, 255, 0))

            if lutador1.cabeca is not None:
                start, end = lutador1.cabeca
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador1.tronco is not None:
                start, end = lutador1.tronco
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador1.linhasCinturas is not None:
                start, end = lutador1.linhasCinturas
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador1.maoEsquerdaCoord is not None:
                start, end = lutador1.maoEsquerdaCoord
                annotator.im = cv2.rectangle(annotator.im, start, end, (0, 0, 255), 5)
            if lutador1.maoDireitaCoord is not None:
                start, end = lutador1.maoDireitaCoord
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)

            if lutador2.cabeca is not None:
                start, end = lutador2.cabeca
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador2.tronco is not None:
                start, end = lutador2.tronco
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador2.linhasCinturas is not None:
                start, end = lutador2.linhasCinturas
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
            if lutador2.maoEsquerdaCoord is not None:
                start, end = lutador2.maoEsquerdaCoord
                annotator.im = cv2.rectangle(annotator.im, start, end, (0, 0, 255), 5)
            if lutador2.maoDireitaCoord is not None:
                start, end = lutador2.maoDireitaCoord
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)

            contador += 1

        #verifica_colisao(keypoints, r, frame, areaLutador, coordenada_corte, lutador1, lutador2)
        """cab = cabeca(frame, results)
        for c in cab:
            (start, end) = c
            annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)
        tronco = troncoCoordenadas(frame, results)
        for tron in tronco:
            (start, end) = tron
            annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 5)

        linhasCinturas = linhaCintura(frame, results)
        for linha in linhasCinturas:
            (start, end) = linha
            annotator.im = cv2.rectangle(annotator.im, start, end, (0, 0, 255), 10)
        maoEsquerdaCoord = maoEsquerda(frame, results)
        for maoEsq in maoEsquerdaCoord:
            (start, end) = maoEsq
            annotator.im = cv2.rectangle(annotator.im, start, end, (0, 0, 255), 5)
        maoDireitaCoord = maoDireita(frame, results)
        for maoDir in maoDireitaCoord:
            (start, end) = maoDir
            annotator.im = cv2.rectangle(annotator.im, start, end, (0, 0, 255), 5)"""
        return annotator


def verifica_colisao(keypoints, r, frame, areaLutador, coordenada_corte, lutador1, lutador2):
    # xyxy
    retangulo1 = areaLutador[0]

    retangulo2 = areaLutador[1]

    if colisao(retangulo1, retangulo2):
        golpe(keypoints, r, frame, coordenada_corte, lutador1, lutador2)
    else:
        #print("Sem colisao.")
        pass
