import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from meioQuadrado import ponto_central
import time
import utils

from clusteriza import DominantColors
from lutador import Lutador

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = utils.retorna_diretorio("pose_landmarker_heavy.task")

video_source = utils.retorna_diretorio("videocompletocorte.mp4")

num_poses = 2
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7


def draw_landmarks_on_image(rgb_image, detection_result, imagem_original):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    coordenada_corte_perna = list()

    #print(len(pose_landmarks_list))
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])

        coordenada_corte_perna.append(draw_legs(annotated_image, pose_landmarks))

        draw_boundingBox(annotated_image, pose_landmarks, imagem_original, coordenada_corte_perna)

        # TESTANDO
        """Ax = pose_landmarks[11].x
        Ay = pose_landmarks[11].y
        A = (Ax, Ay)

        Bx = pose_landmarks[12].x
        By = pose_landmarks[12].y
        B = (Bx, By)

        Cx = pose_landmarks[23].x
        Cy = pose_landmarks[23].y
        C = (Cx, Cy)

        Dx = pose_landmarks[24].x
        Dy = pose_landmarks[24].y
        D = (Dx, Dy)

        centro_quadrilatero = ponto_central(A, B, C, D)

        x = centro_quadrilatero[0]
        y = centro_quadrilatero[1]
        
        radius = 20

        # Desenha um circulo no centro da pessoa para visualização
        #annotated_image = cv2.circle(annotated_image, (int(x * annotated_image.shape[1]), int(y * annotated_image.shape[0])), radius, (255, 0, 0), -1)
        """
        # ---------------------
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

def draw_legs(image, pose_landmarks):
    quadril_esquerdo = (pose_landmarks[23].x,pose_landmarks[23].y)

    quadril_direito = (pose_landmarks[24].x,pose_landmarks[24].y)

    joelho_esquerdo = (pose_landmarks[25].x,pose_landmarks[25].y)

    joelho_direito = (pose_landmarks[26].x,pose_landmarks[26].y)

    min_x = min(quadril_esquerdo[0], quadril_direito[0], joelho_esquerdo[0], joelho_direito[0])
    max_x = max(quadril_esquerdo[0], quadril_direito[0], joelho_esquerdo[0], joelho_direito[0])
    min_y = min(quadril_esquerdo[1], quadril_direito[1], joelho_esquerdo[1], joelho_direito[1])
    max_y = max(quadril_esquerdo[1], quadril_direito[1], joelho_esquerdo[1], joelho_direito[1])

    min_x *= image.shape[1]
    max_x *= image.shape[1]
    min_y *= image.shape[0]
    max_y *= image.shape[0]

    coordenada_corte_perna = list()

    if(int(min_x) > 0 and int(min_y) > 0 and int(max_x) > 0 and int(max_y) > 0):
        min_point = (int(min_x), int(min_y))
        max_point = (int(max_x), int(max_y))
        
        cv2.rectangle(image, min_point, max_point, (0, 0, 255), 2)
        coordenada_corte_perna.append(min_point)
        coordenada_corte_perna.append(max_point)
        # Pega o canto inferior esquerdo e o canto superior direito do retangulo e recorta a imagem nesse intervalo
        #imagem_cortada = image[min_point[1]:max_point[1], min_point[0]:max_point[0]]

        #avg_color_per_row = np.average(imagem_cortada, axis=0)
        #avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)
    return coordenada_corte_perna



def draw_boundingBox(image, pose_landmarks, imagem_original, coordenada_corte_perna):
    if len(coordenada_corte_perna) > 0:
        min_x = min(landmark.x for landmark in pose_landmarks)
        max_x = max(landmark.x for landmark in pose_landmarks)
        min_y = min(landmark.y for landmark in pose_landmarks)
        max_y = max(landmark.y for landmark in pose_landmarks)
        
        min_x *= image.shape[1]
        max_x *= image.shape[1]
        min_y *= image.shape[0]
        max_y *= image.shape[0]
        
        min_point = (int(min_x), int(min_y))
        max_point = (int(max_x), int(max_y))
        
        cv2.rectangle(image, min_point, max_point, (255, 0, 0), 2)
        
        # Faz uma verificacao em cada tupla existente nas coordenadas para que nao de erro caso a pessoa saia da tela
        if all(len(sub_list) > 0 for sub_list in coordenada_corte_perna):
            colors = clusteriza_imagem_perna(imagem_original, coordenada_corte_perna)
            #print(colors)
            if lutador1.cor is not None and lutador2.cor is not None and colors is not None:
                #print(lutador1.cor)
                #print(lutador2.cor)
                #print(coordenada_corte_perna)
                # Calcula as coordenadas do texto (acima do bounding box)
                text_position = (int(min_x), int(min_y - 10))  # 10 pixels acima da coordenada y do bounding box
                identifica_lutador = define_lutador(colors)
                #print(identifica_lutador)
                
                # Adicione o texto acima do bounding box
                cv2.putText(image, "Lutador " + str(identifica_lutador), text_position, cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
    else:
        print("CHEGUEI AQUI")
        pass

def define_lutador(colors, tolerancia=30):
    # Pega sempre o ultimo valor do vetor
    # Pois vai ser sempre o ultimo lutador a ser detectado
    cor_lutador_atual = colors[-1]
    #print(lutador2.cor)
    if np.all(np.abs(cor_lutador_atual - lutador1.cor) <= tolerancia):
        return 1
    elif np.all(np.abs(cor_lutador_atual - lutador2.cor) <= tolerancia):
        return 2
    else:
        return None

def clusteriza_imagem_perna(imagem_original, coordenada_corte_perna):
    # TODO: Caso seja mais de 2 pessoas, o que fazer
    if len(coordenada_corte_perna) <= 2 and len(coordenada_corte_perna) > 0:
        colors = list()
        for pernas_pessoa in coordenada_corte_perna:
            min_point = pernas_pessoa[0]
            max_point = pernas_pessoa[1]
            # Pega o canto inferior esquerdo e o canto superior direito do retangulo e recorta a imagem nesse intervalo
            imagem_cortada = imagem_original[min_point[1]:max_point[1], min_point[0]:max_point[0]]

            clusters = 1
            dc = DominantColors(imagem_cortada, clusters)
            #print(lutador1.cor)
            #print(lutador2.cor)
            # Metodo 1
            colors.append(dc.dominantColors())
            #print(colors)
            # Metodo 2
            #avg_color_per_row = np.average(imagem_cortada, axis=0)
            #avg_color = np.average(avg_color_per_row, axis=0)
            #print(avg_color)
            # Metodo 3
            # Pensei em diminuir ainda mais a regiao de interesse, para diminuir a quantidade de pixeis a serem analisados

            #dc.plotHistogram()
            #cv2.imshow("ts", imagem_cortada)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #print(colors)
            if len(colors) == 2:
                primeira_execucao = np.concatenate(colors)
                if lutador1.primeira_execucao == 1 and lutador2.primeira_execucao == 0:
                    lutador2.cor = primeira_execucao[1]
                    lutador2.primeira_execucao = 1

                if lutador1.primeira_execucao == 0 and lutador2.primeira_execucao == 0:
                    lutador1.cor = primeira_execucao[0]
                    lutador1.primeira_execucao = 1
        # Concatene todas as listas de cores em um único array
        # Pega as cores de ambas as pernas detectadas e junta elas em um array
        concatenated_colors = np.concatenate(colors)
        return concatenated_colors
    else:
        return None


to_window = None
last_timestamp_ms = 0

def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    global lutador
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    # print("pose landmarker result: {}".format(detection_result))
    imagem_original = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)

    # Desenha landmarks na imagem de saída
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), detection_result, imagem_original)
    
    to_window = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    #clusteriza_imagem_perna(imagem_original, coordenada_corte_perna)
    #cv2.imshow("seila", imagem_original)
    #cv2.waitKey(0)




base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    output_segmentation_masks=False,
    result_callback=print_result
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    lutador1 = Lutador(1,None,0)
    lutador2 = Lutador(2,None,0)
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        #image = cv2.resize(image, (540, 960))

        # Converta o quadro recebido do OpenCV para um objeto Image do MediaPipe.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if to_window is not None:
            cv2.imshow("MediaPipe Pose Landmark", to_window)
            time.sleep(0.2)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()