import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from yolo.clusteriza import DominantColors, clusterizaFunction
from yolo.identificaColisao import colisao

from python.yolo.moduloDefineCoordenadas import *

from ultralytics.utils.plotting import Annotator
from python.yolo.roiParts import roi_cabeca, roi_tronco, roi_linha_cintura, roi_mao_esquerda, roi_mao_direita

CLASS_LABELS = {0: 'vermelho', 1: 'azul'}
CLASS_COLORS = {0: (0, 0, 255), 1: (255, 0, 0)}

SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


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

    # print(coordenada_start_y, coordenada_end_y)
    # print(coordenada_start_x, coordenada_end_x)

    return [coordenada_start_x, coordenada_end_x, coordenada_start_y, coordenada_end_y]


def define_lutador(lutador1, lutador2, cor, tolerancia=80):
    cor = np.array(cor, dtype=float)
    c1 = np.array(lutador1.cor, dtype=float)
    c2 = np.array(lutador2.cor, dtype=float)

    # calcula distância euclidiana
    dist1 = np.linalg.norm(cor - c1)
    dist2 = np.linalg.norm(cor - c2)

    if dist1 <= tolerancia:
        return 1
    elif dist2 <= tolerancia:
        return 2
    else:
        return None


def boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count):
    # Garante que exista um dicionário para este frame
    frame_lutador.setdefault(frame_count, {})

    if lutador1.cor is None and lutador2.cor is None and len(cores) == 2:
        lutador1.cor = cores[0]
        lutador2.cor = cores[1]

    coordenada_corte = list()
    keypoints = results[0].keypoints
    for pessoa in keypoints:
        keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
        # draw_boundingBox(imagem, keypoints_numpy)
        coord = pernaCoordenadas(frame, keypoints_numpy)
        coordenada_corte.append(coord)
        recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
        teste = DominantColors(recorte, 1)
        cor = teste.dominantColors()
        if cor.size == 0:
            # print(f"[boundingBox] recorte vazio em pessoa, pulando define_lutador.")
            continue
        identifica_lutador = define_lutador(lutador1, lutador2, cor[0])
        if identifica_lutador == 1:
            lutador1.identificador = 1
            roiCabeca = roi_cabeca(frame, keypoints_numpy)
            roiTronco = roi_tronco(frame, keypoints_numpy)
            roiLinhaCintura = roi_linha_cintura(frame, keypoints_numpy)
            roiMaoEsquerda = roi_mao_esquerda(frame, keypoints_numpy)
            roiMaoDireita = roi_mao_direita(frame, keypoints_numpy)
            lutador1.roi_cabeca = roiCabeca
            lutador1.roi_tronco = roiTronco
            lutador1.roi_linha_cintura = roiLinhaCintura
            lutador1.roi_mao_esquerda = roiMaoEsquerda
            lutador1.roi_mao_direita = roiMaoDireita

            # COORDENADAS DE CADA KEYPOINT DOS LUTADORES SENDO ARMAZENADO
            lutador1.nose = nose_coordenadas(frame, keypoints_numpy)
            lutador1.left_eye = left_eye_coordenadas(frame, keypoints_numpy)
            lutador1.right_eye = right_eye_coordenadas(frame, keypoints_numpy)
            lutador1.left_ear = left_ear_coordenadas(frame, keypoints_numpy)
            lutador1.right_ear = right_ear_coordenadas(frame, keypoints_numpy)
            lutador1.left_shoulder = left_shoulder_coordenadas(frame, keypoints_numpy)
            lutador1.right_shoulder = right_shoulder_coordenadas(frame, keypoints_numpy)
            lutador1.left_elbow = left_elbow_coordenadas(frame, keypoints_numpy)
            lutador1.right_elbow = right_elbow_coordenadas(frame, keypoints_numpy)
            lutador1.left_wrist = left_wrist_coordenadas(frame, keypoints_numpy)
            lutador1.right_wrist = right_wrist_coordenadas(frame, keypoints_numpy)
            lutador1.left_hip = left_hip_coordenadas(frame, keypoints_numpy)
            lutador1.right_hip = right_hip_coordenadas(frame, keypoints_numpy)
            lutador1.left_knee = left_knee_coordenadas(frame, keypoints_numpy)
            lutador1.right_knee = right_knee_coordenadas(frame, keypoints_numpy)
            lutador1.left_ankle = left_ankle_coordenadas(frame, keypoints_numpy)
            lutador1.right_ankle = right_ankle_coordenadas(frame, keypoints_numpy)

            frame_lutador[frame_count].update({'lutador_1': lutador1})
        elif identifica_lutador == 2:
            lutador2.identificador = 2
            roiCabeca = roi_cabeca(frame, keypoints_numpy)
            roiTronco = roi_tronco(frame, keypoints_numpy)
            roiLinhaCintura = roi_linha_cintura(frame, keypoints_numpy)
            roiMaoEsquerda = roi_mao_esquerda(frame, keypoints_numpy)
            roiMaoDireita = roi_mao_direita(frame, keypoints_numpy)
            lutador2.roi_cabeca = roiCabeca
            lutador2.roi_tronco = roiTronco
            lutador2.roi_linha_cintura = roiLinhaCintura
            lutador2.roi_mao_esquerda = roiMaoEsquerda
            lutador2.roi_mao_direita = roiMaoDireita
            lutador2.coordenadas = keypoints_numpy

            # COORDENADAS DE CADA KEYPOINT DOS LUTADORES SENDO ARMAZENADO
            lutador2.nose = nose_coordenadas(frame, keypoints_numpy)
            lutador2.left_eye = left_eye_coordenadas(frame, keypoints_numpy)
            lutador2.right_eye = right_eye_coordenadas(frame, keypoints_numpy)
            lutador2.left_ear = left_ear_coordenadas(frame, keypoints_numpy)
            lutador2.right_ear = right_ear_coordenadas(frame, keypoints_numpy)
            lutador2.left_shoulder = left_shoulder_coordenadas(frame, keypoints_numpy)
            lutador2.right_shoulder = right_shoulder_coordenadas(frame, keypoints_numpy)
            lutador2.left_elbow = left_elbow_coordenadas(frame, keypoints_numpy)
            lutador2.right_elbow = right_elbow_coordenadas(frame, keypoints_numpy)
            lutador2.left_wrist = left_wrist_coordenadas(frame, keypoints_numpy)
            lutador2.right_wrist = right_wrist_coordenadas(frame, keypoints_numpy)
            lutador2.left_hip = left_hip_coordenadas(frame, keypoints_numpy)
            lutador2.right_hip = right_hip_coordenadas(frame, keypoints_numpy)
            lutador2.left_knee = left_knee_coordenadas(frame, keypoints_numpy)
            lutador2.right_knee = right_knee_coordenadas(frame, keypoints_numpy)
            lutador2.left_ankle = left_ankle_coordenadas(frame, keypoints_numpy)
            lutador2.right_ankle = right_ankle_coordenadas(frame, keypoints_numpy)

            frame_lutador[frame_count].update({'lutador_2': lutador2})

    annotated_frame = results[0].plot(boxes=False)
    for r in results:
        annotator = Annotator(annotated_frame)
        boxes = r.boxes
        contador = 0
        areaLutador = list()

        for box in boxes:
            # print(keypoints)
            coord = coordenada_corte[contador]
            # print(x)
            # print(y)
            recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
            teste = DominantColors(recorte, 1)
            cor = teste.dominantColors()
            if cor.size == 0:
                # print(f"[boundingBox] recorte vazio em pessoa, pulando define_lutador.")
                continue
            identifica_lutador = define_lutador(lutador1, lutador2, cor[0])
            if identifica_lutador == 1:
                lutador1.identificador = 1
                lutador1.box = box
                frame_lutador[frame_count].update({'lutador_1': lutador1})
            elif identifica_lutador == 2:
                lutador2.identificador = 2
                lutador2.box = box
                frame_lutador[frame_count].update({'lutador_2': lutador2})
            # frame_lutador[frame_count].update({'lutador_id': identifica_lutador, 'coordenada':})

            # print(identifica_lutador)
            b = box.xyxy[0]
            box = b.tolist()
            areaLutador.append(box)

            if identifica_lutador == 1:
                label_lutador = (
                    f"Lutador 1: {lutador1.socos}|"
                    f"{lutador1.irregular}"
                )
            elif identifica_lutador == 2:
                label_lutador = (
                    f"Lutador 2: {lutador2.socos}|"
                    f"{lutador2.irregular}"
                )
            else:
                label_lutador = "Lutador desconhecido"
            annotator.box_label(b, label_lutador, color=(0, 0, 0))

            if lutador1.roi_cabeca is not None:
                start, end = lutador1.roi_cabeca
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador1.roi_tronco is not None:
                start, end = lutador1.roi_tronco
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador1.roi_linha_cintura is not None:
                start, end = lutador1.roi_linha_cintura
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador1.roi_mao_esquerda is not None:
                start, end = lutador1.roi_mao_esquerda
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador1.roi_mao_direita is not None:
                start, end = lutador1.roi_mao_direita
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)

            if lutador2.roi_cabeca is not None:
                start, end = lutador2.roi_cabeca
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador2.roi_tronco is not None:
                start, end = lutador2.roi_tronco
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador2.roi_linha_cintura is not None:
                start, end = lutador2.roi_linha_cintura
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador2.roi_mao_esquerda is not None:
                start, end = lutador2.roi_mao_esquerda
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)
            if lutador2.roi_mao_direita is not None:
                start, end = lutador2.roi_mao_direita
                annotator.im = cv2.rectangle(annotator.im, start, end, (255, 0, 0), 2)

            contador += 1

        return annotator


def boundingBoxOtimizado(frame, results, lutador1, lutador2, frame_lutador, frame_count):
    annotated = frame.copy()
    annotator = Annotator(annotated)
    frame_lutador.setdefault(frame_count, {})
    h, w = frame.shape[:2]

    for r in results:
        boxes = r.boxes
        kps = r.keypoints.xyn.cpu().numpy() if hasattr(r.keypoints, 'xyn') else None

        for i, box in enumerate(boxes):
            cls_idx = int(box.cls[0])
            base_label = CLASS_LABELS.get(cls_idx, 'desconhecido')
            color = CLASS_COLORS.get(cls_idx, (0, 255, 0))

            # decide qual lutador e pega contagem de socos
            if cls_idx == 0:
                lut = lutador1
            else:
                lut = lutador2
            count_socos = getattr(lut, 'socos', 0)
            count_irregulares = getattr(lut, 'irregular', 0)

            # monta label com contagem
            label_text = (
                f"{base_label}("
                f"{count_socos}|"
                f"{count_irregulares})"
            )

            # desenha bbox + label
            xyxy = list(map(int, box.xyxy[0].tolist()))
            annotator.box_label(xyxy, label_text, color=color)

            # atualiza lutador e frame_lutador
            lut.identificador = 1 if cls_idx == 0 else 2
            lut.box = xyxy
            frame_lutador[frame_count][f'lutador_{lut.identificador}'] = lut

            # se houver pose, desenha pontos, esqueleto e ROIs...
            if kps is not None and i < len(kps):
                person_kp = kps[i]
                pts = []
                # filtra e marca keypoints inválidos como None
                for x_rel, y_rel in person_kp:
                    if x_rel == 0 and y_rel == 0:
                        pts.append(None)
                    else:
                        x, y = int(x_rel * w), int(y_rel * h)
                        pts.append((x, y))
                        cv2.circle(annotator.im, (x, y), 3, color, -1)

                # desenha somente as arestas cujo par de pontos existam
                for a, b in SKELETON:
                    pa, pb = pts[a], pts[b]
                    if pa is not None and pb is not None:
                        cv2.line(annotator.im, pa, pb, color, 2)

                # ROIs
                head = roi_cabeca(frame, person_kp)
                trunk = roi_tronco(frame, person_kp)
                waist = roi_linha_cintura(frame, person_kp)
                left_hand = roi_mao_esquerda(frame, person_kp)
                right_hand = roi_mao_direita(frame, person_kp)
                for roi in [head, trunk, waist, left_hand, right_hand]:
                    if roi is not None:
                        start, end = roi
                        cv2.rectangle(annotator.im, start, end, color, 2)

                # atualiza atributos de ROI no objeto lutador
                lut.roi_cabeca = head
                lut.roi_tronco = trunk
                lut.roi_linha_cintura = waist
                lut.roi_mao_esquerda = left_hand
                lut.roi_mao_direita = right_hand

    return annotator
