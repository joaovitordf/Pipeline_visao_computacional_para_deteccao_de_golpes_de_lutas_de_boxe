import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# FONTE: https://medium.com/buzzrobot/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
        #read image
        img = self.IMAGE
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        """# Criar uma imagem em branco para exibir as cores
        color_display = np.zeros((100, len(self.COLORS) * 100, 3), dtype=np.uint8)

        # Preencher a imagem com as cores dominantes
        for i, color in enumerate(self.COLORS):
            color_display[:, i * 100:(i + 1) * 100] = color

        cv2.imshow("cores dominantes", color_display)
        cv2.waitKey(0)"""
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    """def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()"""

def troncoCoordenadas(imagem, keypoints):

    x1 = int(keypoints[6][0]*imagem.shape[1])

    y1 = int(keypoints[6][1]*imagem.shape[0])

    x2 = int(keypoints[11][0]*imagem.shape[1])

    y2 = int(keypoints[11][1]*imagem.shape[0])

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

    x1 = int(keypoints[12][0]*imagem.shape[1])

    y1 = int(keypoints[12][1]*imagem.shape[0])

    x2 = int(keypoints[13][0]*imagem.shape[1])

    y2 = int(keypoints[13][1]*imagem.shape[0])

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

def clusterizaFunction(imagem, results):
    if hasattr(results[0], 'keypoints'):
        # Access the keypoints for the first detected object
        keypoints = results[0].keypoints
        # Convert keypoints to numpy array and access the keypoints for the first detected object
        imagens_pessoa = list()
        for pessoa in keypoints:
            keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
            #draw_boundingBox(imagem, keypoints_numpy)
            imagens_pessoa.append(pernaCoordenadas(imagem, keypoints_numpy))

        cores = list()
        for pessoa in imagens_pessoa:
            teste = DominantColors(pessoa, 1)
            cor = teste.dominantColors()
            cores.append(cor[0])

        return cores