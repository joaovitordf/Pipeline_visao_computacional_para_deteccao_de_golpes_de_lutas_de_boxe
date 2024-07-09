import math
import numpy as np


def define_lutador(lutador1, lutador2, cor, tolerancia=50):
    lut1 = np.abs(cor - lutador1.cor)
    lut2 = np.abs(cor - lutador2.cor)

    media_lut1 = (lut1[0] + lut1[1] + lut1[2]) / 3
    media_lut2 = (lut2[0] + lut2[1] + lut2[2]) / 3

    if media_lut1 <= tolerancia:
        return 1
    elif media_lut2 <= tolerancia:
        return 2
    else:
        return None

def golpe(keypoints, results, frame, coordenada_corte, lutador1, lutador2):
    info_pessoa = dict()
    boxes = results.boxes

    distancia_mao_ombro = list()
    angulo_mao_ombro = list()
    cabeca = list()

    for pessoa in keypoints:
        keypoints_numpy = pessoa.xyn.cpu().numpy()[0]

        # Cabeca
        x_0 = keypoints_numpy[0][0]
        y_0 = keypoints_numpy[0][1]

        # Ombro direito
        x_6 = keypoints_numpy[6][0]
        y_6 = keypoints_numpy[6][1]

        # Quero o angulo de 8
        # Cotovelo direito
        x_8 = keypoints_numpy[8][0]
        y_8 = keypoints_numpy[8][1]

        # Mao direita
        x_10 = keypoints_numpy[10][0]
        y_10 = keypoints_numpy[10][1]

        pontos = np.array([[x_6, y_6], [x_8, y_8], [x_10, y_10]])
        A = (pontos[0][0], pontos[0][1])
        B = (pontos[1][0], pontos[1][1])
        C = (pontos[2][0], pontos[2][1])

        cabeca.append([(x_0 * frame.shape[1]), (y_0 * frame.shape[0])])

        #print(A,B,C)
        dist_mao_ombro = calcula_distancia_dois_pontos(x_6, y_6, x_10, y_10)
        ang_mao_ombro = printAngle(A, B, C)

    contador = 0
    for box in boxes:
        b = box.xyxy[0]
        box = b.tolist()

        tl = [box[0], box[1]]
        br = [box[2], box[3]]
        for cab in cabeca:
            verificacao = ponto_in_retangulo(tl, br, cab)
            print(verificacao)
        """#print(verificacao)

        coord = coordenada_corte[contador]
        # print(x)
        # print(y)
        recorte = frame[coord[2]: coord[3], coord[0]:coord[1]]
        teste = DominantColors(recorte, 1)
        cor = teste.dominantColors()

        identifica_lutador = define_lutador(lutador1, lutador2, cor[0])"""

        contador += 1

        #distancia_mao_ombro.append(dist_mao_ombro)
        #angulo_mao_ombro.append(ang_mao_ombro)

        """if angulo_entre_mao_ombro > 130:
            print("Jab")
        else:
            print("Cruzado")"""


def ponto_in_retangulo(tl, br, p):
    """print("--------------------------------")
    print(tl)
    print(br)
    print(p)
    print("--------------------------------")"""
    if tl[0] <= p[0] <= br[0] and tl[1] <= p[1] <= br[1]:
        return True
    else:
        return False


def calcula_distancia_dois_pontos(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def printAngle(A, B, C):
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)

    # length of sides be a, b, c  
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)

    # From Cosine law  
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    betta = math.acos((a2 + c2 - b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))

    # Converting to degree  
    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    # printing all the angles  
    #print("alpha : %f" %(alpha))  
    #print("betta : %f" %(betta)) 
    #print("gamma : %f" %(gamma))

    return betta


# returns square of distance b/w two points  
def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff


"""import numpy as np

def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are the points in format [x, y]
    # Calculate the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Example points
p1 = [0, 0]
p2 = [1, 0]
p3 = [1, 1]

angle = calculate_angle(p1, p2, p3)
print(f"The angle is {angle} degrees")"""
