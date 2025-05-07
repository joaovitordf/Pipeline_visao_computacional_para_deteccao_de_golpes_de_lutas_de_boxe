import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# FONTE: https://medium.com/buzzrobot/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036

class DominantColors:
    def __init__(self, image, clusters=3, tolerancia=90):
        self.CLUSTERS = clusters
        self.IMAGE = image  # Expect RGB image
        self.COLORS = None
        self.LABELS = None
        self.tolerancia = tolerancia

    def dominantColors(self):
        # reshape to list of pixels
        pixels = self.IMAGE.reshape(-1, 3)
        self.IMAGE = pixels

        # run k-means
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(pixels)
        self.COLORS = kmeans.cluster_centers_.astype(int)
        self.LABELS = kmeans.labels_
        return self.COLORS

    """def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % tuple(rgb)

    def plotClusters(self):
        colors_hex = [self.rgb_to_hex(c) for c in self.COLORS]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.COLORS[:, 0], self.COLORS[:, 1], self.COLORS[:, 2],
            c=colors_hex,
            s=200,
            marker='o',
            edgecolors='k'
        )

        # desenhar esfera de tolerância ao redor de cada cor dominante
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        for center in self.COLORS:
            x = self.tolerancia * np.cos(u) * np.sin(v) + center[0]
            y = self.tolerancia * np.sin(u) * np.sin(v) + center[1]
            z = self.tolerancia * np.cos(v) + center[2]
            ax.plot_wireframe(x, y, z, color='black', alpha=0.3)

        # Forçar os eixos para o range RGB
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        plt.tight_layout()
        plt.show(block=True)"""

def roi_tronco(imagem, keypoints):
    x1 = int(keypoints[6][0] * imagem.shape[1])

    y1 = int(keypoints[6][1] * imagem.shape[0])

    x2 = int(keypoints[11][0] * imagem.shape[1])

    y2 = int(keypoints[11][1] * imagem.shape[0])

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

    recorte = imagem[coordenada_start_y: coordenada_end_y, coordenada_start_x:coordenada_end_x]

    return recorte


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

    #print(coordenada_start_y, coordenada_end_y)
    #print(coordenada_start_x, coordenada_end_x)

    recorte = imagem[coordenada_start_y: coordenada_end_y, coordenada_start_x:coordenada_end_x]

    return recorte


def clusterizaFunction(imagem, results, lutador1, lutador2, frame_lutador, frame_count):
    cores = []
    #print(lutador1.distancia)
    if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
        # Lista para armazenar as imagens recortadas
        imagens_pessoa = []
        # Itera sobre os keypoints detectados
        for pessoa in results[0].keypoints:
            keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
            #draw_boundingBox(imagem, keypoints_numpy)
            # Obtém o recorte da imagem com base nos keypoints
            recorte = pernaCoordenadas(imagem, keypoints_numpy)
            #cv2.imshow("recorte", recorte)
            #cv2.waitKey(0)
            if recorte.size == 0:
                #print("Recorte vazio, pulando processamento.")
                continue

            imagens_pessoa.append(recorte)

        # Processa os recortes para obter cores dominantes
        for pessoa in imagens_pessoa:
            try:
                dominante = DominantColors(pessoa, 1)
                cor = dominante.dominantColors()
                cores.append(cor[0])
            except ValueError as e:
                pass
                #print(f"Erro ao processar a imagem: {e}")
    else:
        print("Keypoints não encontrados ou inválidos no resultado.")

    return cores
