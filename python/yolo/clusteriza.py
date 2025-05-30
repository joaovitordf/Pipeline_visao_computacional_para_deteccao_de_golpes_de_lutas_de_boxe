import cv2
from sklearn.cluster import KMeans
import numpy as np


class DominantColors:
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image  # Espera imagem RGB
        self.COLORS = None
        self.LABELS = None

    def dominantColors(self):
        # converte em lista de pixels: (N,3)
        pixels = self.IMAGE.reshape(-1, 3)

        # Protege contra array vazio
        if pixels.shape[0] == 0:
            return np.zeros((0, 3), dtype=int)

        # Ajusta clusters se houver menos pixels que clusters
        n_clusters = min(self.CLUSTERS, pixels.shape[0])

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)

        self.COLORS = kmeans.cluster_centers_.astype(int)
        self.LABELS = kmeans.labels_
        return self.COLORS


def pernaCoordenadas(imagem, keypoints):
    x1 = int(keypoints[12][0] * imagem.shape[1])
    y1 = int(keypoints[12][1] * imagem.shape[0])
    x2 = int(keypoints[13][0] * imagem.shape[1])
    y2 = int(keypoints[13][1] * imagem.shape[0])

    start_x, end_x = sorted((x1, x2))
    start_y, end_y = sorted((y1, y2))

    recorte = imagem[start_y:end_y, start_x:end_x]
    return recorte


def clusterizaFunction(imagem, results, lutador1, lutador2, frame_lutador, frame_count):
    cores = []

    # Protege contra resultados vazios
    if not results or len(results) == 0:
        print(f"[clusterizaFunction] nenhum resultado do YOLO no frame {frame_count}.")
        return cores

    # Protege contra keypoints inválidos
    kp_list = getattr(results[0], 'keypoints', None)
    if not kp_list:
        print(f"[clusterizaFunction] keypoints inválidos no frame {frame_count}.")
        return cores

    # 1) recorta perna de cada pessoa
    recortes = []
    for pessoa in kp_list:
        arr = pessoa.xyn.cpu().numpy()
        # Protege contra qualquer array sem pixels/keypoints:
        if arr.size == 0:
            continue

        keyp = arr[0]
        rec = pernaCoordenadas(imagem, keyp)
        if rec.size > 0:
            recortes.append(rec)

    # 2) clusteriza cada recorte
    for rec in recortes:
        dominante = DominantColors(rec, clusters=1)
        cores_dom = dominante.dominantColors()
        if cores_dom.shape[0] > 0:
            cores.append(cores_dom[0])

    return cores
