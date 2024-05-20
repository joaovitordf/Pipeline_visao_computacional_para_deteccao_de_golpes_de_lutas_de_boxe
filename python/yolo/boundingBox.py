import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from yolo.clusteriza import DominantColors, clusterizaFunction

def troncoCoordenadas(imagem, keypoints):

    x1 = int(keypoints[6][0]*imagem.shape[1])

    y1 = int(keypoints[6][1]*imagem.shape[0])

    x2 = int(keypoints[11][0]*imagem.shape[1])

    y2 = int(keypoints[11][1]*imagem.shape[0])

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

    return [coordenada_start_x, coordenada_end_x,coordenada_start_y, coordenada_end_y]

def pernaCoordenadas(imagem, keypoints):

    x1 = int(keypoints[12][0]*imagem.shape[1])

    y1 = int(keypoints[12][1]*imagem.shape[0])

    x2 = int(keypoints[13][0]*imagem.shape[1])

    y2 = int(keypoints[13][1]*imagem.shape[0])

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

    return [coordenada_start_x, coordenada_end_x,coordenada_start_y, coordenada_end_y]

def define_lutador(lutador1, lutador2, cor, tolerancia=50):
    lut1 = np.abs(cor - lutador1.cor)
    lut2 = np.abs(cor - lutador2.cor)

    media_lut1 = (lut1[0] + lut1[1] + lut1[2])/3
    media_lut2 = (lut2[0] + lut2[1] + lut2[2])/3

    if media_lut1 <= tolerancia:
        return 1
    elif media_lut2 <= tolerancia:
        return 2
    else:
        return None

def boundingBox(frame, results, cores, lutador1, lutador2):
    if lutador1.cor is None and lutador2.cor is None and len(cores) == 2:
        lutador1.cor = cores[0]
        lutador2.cor = cores[1]

    coordenada_corte = list()
    keypoints = results[0].keypoints
    for pessoa in keypoints:
        keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
        #draw_boundingBox(imagem, keypoints_numpy)
        coordenada_corte.append(pernaCoordenadas(frame, keypoints_numpy))

    annotated_frame = results[0].plot(boxes = False)
    for r in results:
        annotator = Annotator(annotated_frame)
        boxes = r.boxes

        contador = 0

        for box in boxes:
            
            coord = coordenada_corte[contador]
            #print(x)
            #print(y)
            recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
            teste = DominantColors(recorte, 1)
            cor = teste.dominantColors()

            identifica_lutador = define_lutador(lutador1, lutador2, cor[0])

            b = box.xyxy[0] 

            label_lutador = "Lutador " + str(identifica_lutador)
            annotator.box_label(b, label_lutador)
            contador += 1
    
    return annotator