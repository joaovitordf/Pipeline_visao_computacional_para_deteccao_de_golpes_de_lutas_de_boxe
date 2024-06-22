import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
from sklearn.cluster import KMeans
import numpy as np
import math

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
        
        # Criar uma imagem em branco para exibir as cores
        color_display = np.zeros((100, len(self.COLORS) * 100, 3), dtype=np.uint8)

        # Preencher a imagem com as cores dominantes
        for i, color in enumerate(self.COLORS):
            color_display[:, i * 100:(i + 1) * 100] = color
            print(color)
        
        cv2.imshow("cores dominantes", color_display)
        cv2.waitKey(0)
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
def draw_boundingBox(image, keypoints):
    # [:, 0] Seleciona todos os elementos da primeira coluna
    x = keypoints[keypoints[:, 0] != 0, 0]
    y = keypoints[keypoints[:, 1] != 0, 1]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    largura = image.shape[1]
    altura = image.shape[0]
    
    min_x *= largura
    max_x *= largura
    min_y *= altura
    max_y *= altura
    
    recorte = image[int(min_y):int(max_y), int(min_x):int(max_x)]

    return recorte

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

    print(coordenada_start_y, coordenada_end_y)
    print(coordenada_start_x, coordenada_end_x)
    recorte = imagem[coordenada_start_y: coordenada_end_y, coordenada_start_x:coordenada_end_x]

    cv2.imshow("imagem", recorte)
    cv2.waitKey(0)

    return recorte

def yoloImagem(model, caminho_imagem):
    model = YOLO(model)
    imagem = cv2.imread(caminho_imagem)
    results = model(imagem, verbose = False)
    if hasattr(results[0], 'keypoints'):
        # Access the keypoints for the first detected object
        keypoints = results[0].keypoints
        # Convert keypoints to numpy array and access the keypoints for the first detected object
        imagens_pessoa = list()
        for pessoa in keypoints:
            keypoints_numpy = pessoa.xyn.cpu().numpy()[0]
            #draw_boundingBox(imagem, keypoints_numpy)
            imagens_pessoa.append(troncoCoordenadas(imagem, keypoints_numpy))

        for pessoa in imagens_pessoa:
            teste = DominantColors(pessoa, 1)
            teste.dominantColors()
        """cv2.imshow("imagem", imagem)
        cv2.waitKey(0)"""

    """
        min_point = (int(min_x), int(min_y))
        max_point = (int(max_x), int(max_y))
        
        cv2.rectangle(image, min_point, max_point, (0, 0, 255), 2)"""

    
    """for r in results:
        annotator = Annotator(imagem)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            annotator.box_label(b, "teste")
          
    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)     
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    """# Check if the keypoints attribute is present in the results
    if hasattr(results[0], 'keypoints'):
        # Access the keypoints for the first detected object
        keypoints = results[0].keypoints
        # Convert keypoints to numpy array and access the keypoints for the first detected object
        #print(len(keypoints))
        keypoints_numpy = keypoints.xyn.cpu().numpy()[0]
        #print(keypoints_numpy)
        x = int(keypoints_numpy[10][0] * imagem.shape[1])  # Converte para coordenadas da imagem
        y = int(keypoints_numpy[10][1] * imagem.shape[0])
        centro = (x, y)
        raio = 5  # Raio do círculo
        cor = (0, 0, 255)  # Cor do círculo em BGR 
        espessura = 10
        #print(results)
        #print(keypoints_numpy)

        annotated_frame = results[0].plot(boxes = True)
        cv2.circle(annotated_frame, centro, raio, cor, espessura)

        cv2.imshow("imagem", annotated_frame)
        cv2.waitKey(0)
    else:
        print("No keypoints attribute found in the results.")
        """
    
def yoloAnguloJuntas(model, caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    results = model(imagem, verbose = False)
    if hasattr(results[0], 'keypoints'):
        # Access the keypoints for the first detected object
        keypoints = results[0].keypoints
        # Convert keypoints to numpy array and access the keypoints for the first detected object
        imagens_pessoa = list()
        for pessoa in keypoints:
            keypoints_numpy = pessoa.xyn.cpu().numpy()[0]

            x_6 = keypoints_numpy[6][0]
            y_6 = keypoints_numpy[6][1]

            # Quero o angulo de 8
            x_8 = keypoints_numpy[8][0]
            y_8 = keypoints_numpy[8][1]

            x_10 = keypoints_numpy[10][0]
            y_10 = keypoints_numpy[10][1]

            pontos = np.array([[x_6, y_6],[x_8, y_8],[x_10, y_10]])
            A = (pontos[0][0], pontos[0][1])
            B = (pontos[1][0], pontos[1][1])
            C = (pontos[2][0], pontos[2][1])

            print(A,B,C)

            printAngle(A,B,C)
            

            annotated_frame = results[0].plot(boxes = False)

            """annotated_frame = desenhaCirculo(annotated_frame, results, x_6, y_6)
            annotated_frame = desenhaCirculo(annotated_frame, results, x_8, y_8)
            annotated_frame = desenhaCirculo(annotated_frame, results, x_10, y_10)"""

            cv2.imshow("imagem", annotated_frame)
            cv2.waitKey(0)

def desenhaCirculo(annotated_frame, results, x,y):
    centro = (x, y)
    raio = 5  # Raio do círculo
    cor = (0, 0, 255)  # Cor do círculo em BGR 
    espessura = 10
    #print(results)
    #print(keypoints_numpy)

    cv2.circle(annotated_frame, centro, raio, cor, espessura)

    return annotated_frame

def printAngle(A, B, C):  
      
    # Square of lengths be a2, b2, c2  
    a2 = lengthSquare(B, C)  
    b2 = lengthSquare(A, C)  
    c2 = lengthSquare(A, B)  
  
    # length of sides be a, b, c  
    a = math.sqrt(a2);  
    b = math.sqrt(b2);  
    c = math.sqrt(c2);  
  
    # From Cosine law  
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c));  
    betta = math.acos((a2 + c2 - b2) / 
                         (2 * a * c));  
    gamma = math.acos((a2 + b2 - c2) / 
                         (2 * a * b));  
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi;  
    betta = betta * 180 / math.pi;  
    gamma = gamma * 180 / math.pi;  
  
    # printing all the angles  
    print("alpha : %f" %(alpha))  
    print("betta : %f" %(betta)) 
    print("gamma : %f" %(gamma)) 

# returns square of distance b/w two points  
def lengthSquare(X, Y):  
    xDiff = X[0] - Y[0]  
    yDiff = X[1] - Y[1]  
    return xDiff * xDiff + yDiff * yDiff 