import asyncio
import websockets
import base64
import cv2
import numpy as np
from typing import Dict, Any
from websockets.legacy.server import WebSocketServerProtocol

from ultralytics import YOLO
import utils
from yolo.identificaColisao import automatoColisao
from yolo.boundingBox import boundingBox
from yolo.clusteriza import clusterizaFunction
from lutador import Lutador


# Global context with type annotation
global_context: Dict[str, Any] = {}

def create_context() -> Dict[str, Any]:
    # ... [same as before] ...
    model_path = utils.retorna_diretorio("pesos/yolov8m-pose.pt")
    model = YOLO(model_path)
    lutador1 = Lutador(1, None, 0, None, None)
    lutador2 = Lutador(2, None, 0, None, None)
    frame_lutador = {}
    return {'model': model, 'lutador1': lutador1, 'lutador2': lutador2, 'frame_lutador': frame_lutador}


def process_yolo(frame, frame_id, context):
    """
    Processa um frame utilizando o modelo YOLO e as funções de pós-processamento.

    Parâmetros:
      frame: imagem (numpy array) a ser processada
      frame_id: identificador do frame
      context: dicionário com o modelo e demais objetos necessários

    Retorna:
      O frame processado (com anotações e colisões detectadas) ou o frame original em caso de erro.
    """
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


# ------------------ Modo Vídeo ------------------

def main_video():
    """
    Inicializa o contexto e processa um vídeo gravado, chamando process_yolo para cada frame.
    """
    context = create_context()
    video_source = utils.retorna_diretorio("videos/vid1.mp4")
    cap = cv2.VideoCapture(video_source)

    frame_count_local = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_yolo(frame, frame_count_local, context)
        # Redimensiona a imagem para visualização (ajuste o fator conforme necessário)
        cv2.imshow("Vídeo Processado", cv2.resize(
            processed_frame,
            (int(processed_frame.shape[1]), int(processed_frame.shape[0]))
        ))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count_local += 1

    cap.release()
    cv2.destroyAllWindows()


# ------------------ Modo Servidor ------------------

frame_count = 0  # Contador global para o modo servidor

async def handler(websocket: WebSocketServerProtocol) -> None:
    global frame_count, global_context
    assert global_context, "global_context must be initialized!"
    async for message in websocket:
        try:
            # Decode the base64 message into an image
            img_data = base64.b64decode(message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Frame inválido recebido.")
            else:
                processed_frame = process_yolo(frame, frame_count, global_context)
                print(f"Frame {frame_count} processado com sucesso.")
                frame_count += 1
                # Display the processed frame (resized for visualization)
                cv2.imshow("Frame Processado", cv2.resize(
                    processed_frame,
                    (int(processed_frame.shape[1]), int(processed_frame.shape[0]))
                ))
                cv2.waitKey(1)
        except Exception as e:
            print(f"Erro ao processar o frame: {e}")

async def main_server():
    global global_context
    global_context = create_context()
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Servidor WebSocket rodando na porta 8765")
        await asyncio.Future()  # Mantém o servidor ativo indefinidamente


# ------------------ Escolha do Modo ------------------

if __name__ == "__main__":
    escolha = input(
        "Escolha uma opção:\n"
        "1: Usar vídeo já gravado\n"
        "2: Usar servidor\n"
        "Digite 1 ou 2: "
    )
    if escolha == "1":
        main_video()
    elif escolha == "2":
        asyncio.run(main_server())
    else:
        print("Opção inválida!")
