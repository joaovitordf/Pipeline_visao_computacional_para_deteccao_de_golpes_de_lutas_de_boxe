import asyncio

import websockets
import base64
import cv2
import numpy as np
from websockets.legacy.server import WebSocketServerProtocol
import json

from ultralytics import YOLO
import utils
from yolo.identificaColisao import automatoColisao, automatoColisaoOtimizado
from yolo.boundingBox import boundingBox, boundingBoxOtimizado
from yolo.clusteriza import clusterizaFunction
from lutador import Lutador

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


# ------------------ Modo Imagem (DEBUG) ------------------


def main_image():
    context = create_context(0)
    image_path = utils.retorna_diretorio("videos/extracao3.png")
    frame = cv2.imread(image_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None or frame.size == 0:
        print("Imagem não encontrada ou inválida.")
        return


# ------------------ Modo Vídeo ------------------

def main_video(otimizado):
    if otimizado == 0:
        context = create_context(0)
    else:
        context = create_context(1)

    lutador1 = context["lutador1"]
    lutador2 = context["lutador2"]

    video_source = utils.retorna_diretorio("videos/fim2.mp4")
    cap = cv2.VideoCapture(video_source)

    tamanho_lista = 10
    fps_lista = deque(maxlen=tamanho_lista)

    total_frames = 0
    media_fps = 0.0
    tempo_inicio_execucao = time.time()

    while cap.isOpened():
        start = time.time()
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1

        resize = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
        if otimizado == 0:
            processed_frame = process_yolo(resize, 0, context)
        else:
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

    tempo_total_execucao = time.time() - tempo_inicio_execucao

    cap.release()
    cv2.destroyAllWindows()

    """# monta json com estatísticas
    dados = {
        "total_frames": total_frames,
        "media_fps": round(media_fps, 2),
        "golpes": {
            "lutador_1": lutador1.socos,
            "lutador_2": lutador2.socos
        },
        "golpes_irregulares": {
            "lutador_1": lutador1.irregular,
            "lutador_2": lutador2.irregular
        },
        "tempo_total_execucao_segundos": round(tempo_total_execucao, 2)
    }

    with open("estatisticas_video_otimizado.txt", "w") as f:
        json.dump(dados, f, indent=4)"""


# ------------------ Modo Servidor ------------------

async def handler(websocket: WebSocketServerProtocol) -> None:
    frame_count = 0
    fps_deque = deque(maxlen=10)
    async for message in websocket:
        try:
            start_time = time.time()

            img_data = base64.b64decode(message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Frame inválido recebido.")
                continue

            frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

            processed = process_yolo(frame_resized, frame_count, global_context)

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)

            text = f"FPS: {avg_fps:.1f}"
            cv2.putText(
                processed, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Frame Processado", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                await websocket.close()
                break

            frame_count += 1

        except Exception as e:
            print(f"Erro ao processar o frame: {e}")


async def main_server():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Servidor WebSocket rodando na porta 8765")
        await asyncio.Future()


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
