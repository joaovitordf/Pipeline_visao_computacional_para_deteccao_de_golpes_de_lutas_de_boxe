import utils
from yolo.identificaColisao import automatoColisao

from yolo.boundingBox import boundingBox
from yolo.clusteriza import clusterizaFunction
from yolo.yoloImagem import yoloImagem, yoloAnguloJuntas
import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from lutador import Lutador


def main():
    model_path = utils.retorna_diretorio("/pesos/yolov8m-pose.pt")

    video_source = utils.retorna_diretorio("/videos/videocompletocorte2.mp4")

    imagem_source = utils.retorna_diretorio("/videos/imagemBoxe.png")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)

    lutador1 = Lutador(1, None, 0, None, None)
    lutador2 = Lutador(2, None, 0, None, None)

    frame_lutador = dict()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, verbose=False)
            try:
                frame_lutador[frame_count] = {'frame': frame_count, 'lutador_1': lutador1, 'lutador_2': lutador2}
                #lutador1.socos += 1
                cores = clusterizaFunction(frame, results)
                annotator = boundingBox(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)
                colisao = automatoColisao(frame, results, cores, lutador1, lutador2, frame_lutador, frame_count)
                annotated_frame = annotator.result()
                altura, largura, _ = np.shape(annotated_frame)
                frame = cv2.resize(annotated_frame, (int((largura * 0.6)), int((altura * 0.6))))
                cv2.imshow("teste", frame)
            except Exception as e:
                print(f"Erro ao processar o frame: {e}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
        #print(frame_lutador[frame_count]['lutador_1'])
        #print(frame_lutador[frame_count]['lutador_2'])
        frame_count += 1
        time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()

    #yoloImagem(model_path, imagem_source)
    #yoloAnguloJuntas(model, imagem_source)


if __name__ == "__main__":
    main()
