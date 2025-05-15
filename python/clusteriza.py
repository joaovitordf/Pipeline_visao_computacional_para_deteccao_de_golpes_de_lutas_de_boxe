import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# FONTE: https://medium.com/buzzrobot/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036

class DominantColors:
    def __init__(self, image, clusters=3, tolerancia=100):
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

    def rgb_to_hex(self, rgb):
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
        plt.show(block=True)