import asyncio
import math

import websockets
import base64
import cv2
import numpy as np
from websockets.legacy.server import WebSocketServerProtocol

from ultralytics import YOLO
import utils
from yolo.identificaColisao import automatoColisao, automatoColisaoOtimizado
from yolo.boundingBox import boundingBox, boundingBoxOtimizado
from yolo.clusteriza import clusterizaFunction
from lutador import Lutador
from clusteriza import DominantColors

import time
from collections import deque

global_context = None


def create_context(otimizado):
    if otimizado == 0:
        model_path = utils.retorna_diretorio("pesos/yolov8m-pose.pt")
        model = YOLO(model_path)
        lutador1 = Lutador(1, None, 0, 0, None, None)
        lutador2 = Lutador(2, None, 0, 0, None, None)
        frame_lutador = {}
    else:
        model_path = r"C:\Users\xjoao\PycharmProjects\TCC\python\runs\pose\train13\weights\best.pt"
        model = YOLO(model_path)
        lutador1 = Lutador(1, None, 0, 0, None, None)
        lutador2 = Lutador(2, None, 0, 0, None, None)
        frame_lutador = {}
    return {'model': model, 'lutador1': lutador1, 'lutador2': lutador2, 'frame_lutador': frame_lutador}


def process_yolo(frame, frame_id, context):
    if frame is None or frame.size == 0:
        print(f"Frame {frame_id} vazio, ignorando processamento YOLO.")
        return frame

    model = context['model']
    lutador1 = context['lutador1']
    lutador2 = context['lutador2']
    frame_lutador = context['frame_lutador']

    frame_original = frame.copy()
    results = model(frame, verbose=False, device="cuda")

    try:
        frame_lutador[frame_id] = {'frame': frame_id, 'lutador_1': lutador1, 'lutador_2': lutador2}
        cores = clusterizaFunction(frame, results, lutador1, lutador2, frame_lutador, frame_id)
        annotator = boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_id)
        annotated_frame = annotator.result()
        annotated_frame = automatoColisao(
            annotated_frame, results, cores, lutador1, lutador2, frame_lutador, frame_id, frame_original
        )
        return annotated_frame
    except Exception as e:
        print(f"Erro inesperado no processamento do frame {frame_id}: {e}")
        return frame


def process_yolo_otimizado(frame, frame_id, context):
    if frame is None or frame.size == 0:
        print(f"Frame {frame_id} vazio, ignorando processamento YOLO.")
        return frame

    results = context['model'](frame, verbose=False, device="cuda")

    try:
        annotator = boundingBoxOtimizado(
            frame=frame,
            results=results,
            lutador1=context['lutador1'],
            lutador2=context['lutador2'],
            frame_lutador=context['frame_lutador'],
            frame_count=frame_id
        )
        annotated_frame = annotator.im

        _ = automatoColisaoOtimizado(
            frame=annotated_frame,
            results=results,
            lutador1=context['lutador1'],
            lutador2=context['lutador2'],
            frame_lutador=context['frame_lutador'],
            frame_count=frame_id,
            frame_original=frame
        )

        return annotated_frame

    except Exception as e:
        print(f"Erro inesperado no processamento do frame {frame_id}: {e}")
        return frame


def process_yolo_pose(frame, frame_id, context):
    if frame is None or frame.size == 0:
        print(f"Frame {frame_id} vazio, ignorando processamento YOLO.")
        return frame

    model = context['model']
    results = model(frame, verbose=False, device="cuda")

    try:
        annotated_frame = frame.copy()
        for result in results:
            if hasattr(result, 'keypoints'):
                if hasattr(result.keypoints, 'data'):
                    kp_tensor = result.keypoints.data
                    kp_np = kp_tensor.cpu().numpy() if hasattr(kp_tensor, 'cpu') else np.array(kp_tensor)
                else:
                    kp_np = np.array(result.keypoints)

                if kp_np.ndim == 3:
                    for points in kp_np:
                        for point in points:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                elif kp_np.ndim == 2:
                    for point in kp_np:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        return annotated_frame
    except Exception as e:
        print(f"Erro inesperado no processamento do frame {frame_id}: {e}")
        return frame


def process_yolo_pose_trace(frame, frame_id, context):
    if frame is None or frame.size == 0:
        print(f"Frame {frame_id} vazio, ignorando processamento YOLO.")
        return frame

    model = context['model']
    results = model(frame, verbose=False, device="cuda")

    try:
        annotated_frame = frame.copy()
        for result in results:
            if hasattr(result, 'keypoints'):
                if hasattr(result.keypoints, 'data'):
                    kp_tensor = result.keypoints.data
                    kp_np = kp_tensor.cpu().numpy() if hasattr(kp_tensor, 'cpu') else np.array(kp_tensor)
                else:
                    kp_np = np.array(result.keypoints)

                skeleton = [
                    (0, 1), (1, 3), (0, 2), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12),
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                ]

                if kp_np.ndim == 3:
                    for points in kp_np:
                        valid_points = []
                        for point in points:
                            x, y = int(point[0]), int(point[1])
                            if x == 0 and y == 0:
                                valid_points.append(None)
                            else:
                                valid_points.append((x, y))
                                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                        for connection in skeleton:
                            pt1 = valid_points[connection[0]]
                            pt2 = valid_points[connection[1]]
                            if pt1 is not None and pt2 is not None:
                                cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)
                elif kp_np.ndim == 2:
                    valid_points = []
                    for point in kp_np:
                        x, y = int(point[0]), int(point[1])
                        if x == 0 and y == 0:
                            valid_points.append(None)
                        else:
                            valid_points.append((x, y))
                            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    for connection in skeleton:
                        pt1 = valid_points[connection[0]]
                        pt2 = valid_points[connection[1]]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)
        return annotated_frame
    except Exception as e:
        print(f"Erro inesperado no processamento do frame {frame_id}: {e}")
        return frame


def extrai_canal_red(image):
    red_image = np.zeros_like(image)
    red_image[:, :, 2] = image[:, :, 2]
    return red_image


def extrai_canal_green(image):
    green_image = np.zeros_like(image)
    green_image[:, :, 1] = image[:, :, 1]
    return green_image


def extrai_canal_blue(image):
    blue_image = np.zeros_like(image)
    blue_image[:, :, 0] = image[:, :, 0]
    return blue_image

# ------------------ Modo Imagem (DEBUG) ------------------


def main_image():
    context = create_context(0)
    image_path = utils.retorna_diretorio("videos/extracao3.png")
    frame = cv2.imread(image_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None or frame.size == 0:
        print("Imagem não encontrada ou inválida.")
        return

    blue_channel = extrai_canal_blue(frame)

    dc = DominantColors(rgb, clusters=1)
    cores = dc.dominantColors()

    dc.plotClusters()


# ------------------ Modo Vídeo ------------------

def main_video(otimizado):
    if otimizado == 0:
        context = create_context(0)
    else:
        context = create_context(1)
    video_source = utils.retorna_diretorio("videos/fim2.mp4")
    cap = cv2.VideoCapture(video_source)

    tamanho_lista = 10
    fps_lista = deque(maxlen=tamanho_lista)

    while cap.isOpened():
        start = time.time()
        success, frame = cap.read()
        if not success:
            break

        if otimizado == 0:
            resize = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            processed_frame = process_yolo(resize, 0, context)
        else:
            resize = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            processed_frame = process_yolo_otimizado(resize, 0, context)

        end = time.time()
        fps = 1.0 / (end - start)
        fps_lista.append(fps)
        media_fps = sum(fps_lista) / len(fps_lista)

        cv2.putText(processed_frame, f"{media_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)

        cv2.imshow("Janela", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ Modo Servidor ------------------

async def handler(websocket: WebSocketServerProtocol) -> None:
    frame_count = 0
    async for message in websocket:
        try:
            img_data = base64.b64decode(message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Frame inválido recebido.")
            else:
                processed_frame = process_yolo(frame, frame_count, global_context)
                print(f"Frame {frame_count} processado com sucesso.")
                frame_count += 1
                cv2.imshow("Frame Processado", cv2.resize(
                    processed_frame,
                    (int(processed_frame.shape[1]), int(processed_frame.shape[0]))
                ))
                cv2.waitKey(1)
        except Exception as e:
            print(f"Erro ao processar o frame: {e}")

async def main_server():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Servidor WebSocket rodando na porta 8765")
        await asyncio.Future()  # Mantém o servidor ativo indefinidamente


# ------------------ Escolha do Modo ------------------

if __name__ == "__main__":
    global_context = create_context(0)
    escolha = input(
        "Escolha uma opção:\n"
        "0: Usar vídeo já gravado\n"
        "1: Usar vídeo já gravado (otimizado)\n"
        "2: Usar servidor\n"
        "3: Usar imagem\n"
        "Digite 0, 1, 2 ou 3: "
    )
    if escolha == "0":
        main_video(0)
    elif escolha == "1":
        main_video(1)
    elif escolha == "2":
        asyncio.run(main_server())
    elif escolha == "3":
        main_image()
    else:
        print("Opção inválida!")
