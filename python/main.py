import utils

from yolo.boundingBox import boundingBox
from yolo.clusteriza import clusterizaFunction
from yolo.yoloImagem import yoloImagem
import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from lutador import Lutador

def main():
    model_path = utils.retorna_diretorio("/pesos/yolov8m-pose.pt")

    video_source = utils.retorna_diretorio("/videos/videoteste.mp4")

    imagem_source = utils.retorna_diretorio("/videos/imagemBoxe.png")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source)

    lutador1 = Lutador(1,None,0)
    lutador2 = Lutador(2,None,0)

    while (cap.isOpened()):
        success, frame = cap.read()

        if success:
            results = model(frame, verbose = False)
            try:
                cores = clusterizaFunction(frame, results)
                annotator = boundingBox(frame, results, cores, lutador1, lutador2)
                annotated_frame = annotator.result()
                altura, largura, _ = np.shape(annotated_frame)
                frame = cv2.resize(annotated_frame, (int((largura*0.6)),int((altura*0.6))))
                cv2.imshow("teste", frame)
            except Exception as e:
                print(f"Erro ao processar o frame: {e}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

        time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()

    #yoloImagem(model_path, imagem_source)

if __name__ == "__main__":
    main()