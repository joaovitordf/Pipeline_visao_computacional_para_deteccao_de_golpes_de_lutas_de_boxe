import asyncio
import websockets
import base64
import cv2
import numpy as np
import json

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import utils
from yolo.identificaColisao import automatoColisao
from yolo.boundingBox import boundingBox
from yolo.clusteriza import clusterizaFunction
from yolo.yoloImagem import yoloImagem, yoloAnguloJuntas
from lutador import Lutador

frame_count = 0

def process_with_yolo(frame, frame_id):
    """
    Função para processar o frame utilizando YOLO.
    """
    if frame is None or frame.size == 0:
        print(f"Frame {frame_id} vazio, ignorando processamento YOLO.")
        return frame
    else:
        model_path = utils.retorna_diretorio("pesos/yolov8m-pose.pt")
        model = YOLO(model_path)

        lutador1 = Lutador(1, None, 0, None, None)
        lutador2 = Lutador(2, None, 0, None, None)
        frame_lutador = dict()

        frame_original = frame.copy()
        results = model(frame, verbose=False, device="cuda")
        # Realiza o processamento com YOLO
        try:
            frame_lutador[frame_count] = {'frame': frame_count, 'lutador_1': lutador1, 'lutador_2': lutador2}
            # lutador1.socos += 1
            cores = clusterizaFunction(frame, results, lutador1, lutador2, frame_lutador, frame_count)
            annotator = boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)
            annotated_frame = annotator.result()
            annotated_frame = automatoColisao(annotated_frame, results, cores, lutador1, lutador2, frame_lutador,
                                              frame_count, frame_original)
            return annotated_frame
        except Exception as e:
            print(f"Erro inesperado no processamento do frame {frame_id}: {e}")
            return frame

async def handler(websocket, path=None):
    global frame_count
    async for message in websocket:
        try:
            # Imprime os primeiros 50 caracteres da mensagem e o frame count
            print(f"Mensagem recebida para frame {frame_count}: {message[:50]}...")
            # Decodifica a mensagem base64 para imagem
            img_data = base64.b64decode(message)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Frame inválido recebido.")
            else:
                # Processa o frame utilizando YOLO
                processed_frame = process_with_yolo(frame, frame_count)
                print(f"Frame {frame_count} processado com sucesso.")
                frame_count += 1

                # Exibe o frame processado (redimensionado)
                cv2.imshow("Frame Processado", cv2.resize(processed_frame, (int(processed_frame.shape[1] * 3), int(processed_frame.shape[0] * 3))))
                cv2.waitKey(1)
        except Exception as e:
            print(f"Erro ao processar o frame: {e}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Servidor WebSocket rodando na porta 8765")
        await asyncio.Future()  # Mantém o servidor ativo indefinidamente


if __name__ == "__main__":
    asyncio.run(main())

"""def main():
    model_path = utils.retorna_diretorio("pesos/yolov8m-pose.pt")

    video_source = utils.retorna_diretorio("videos/videocompletocorte.mp4")

    imagem_source = utils.retorna_diretorio("videos/imagemBoxe.png")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)

    lutador1 = Lutador(1, None, 0, None, None)
    lutador2 = Lutador(2, None, 0, None, None)

    frame_lutador = dict()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_original = frame
        if success:
            results = model(frame, verbose=False, device="cuda")
            try:
                frame_lutador[frame_count] = {'frame': frame_count, 'lutador_1': lutador1, 'lutador_2': lutador2}
                #lutador1.socos += 1
                cores = clusterizaFunction(frame, results, lutador1, lutador2, frame_lutador, frame_count)
                annotator = boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)
                annotated_frame = annotator.result()
                annotated_frame = automatoColisao(annotated_frame, results, cores, lutador1, lutador2, frame_lutador, frame_count, frame_original)
                altura, largura, _ = np.shape(annotated_frame)
                frame = cv2.resize(annotated_frame, (int((largura * 0.6)), int((altura * 0.6))))
                cv2.imshow("teste", frame)
            except Exception as e:
                pass
                #print(f"Erro ao processar o frame: {e}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
        #print(frame_lutador[frame_count]['lutador_1'])
        #print(frame_lutador[frame_count]['lutador_2'])
        frame_count += 1
        #time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    #yoloImagem(model_path, imagem_source)
    #yoloAnguloJuntas(model, imagem_source)"""
