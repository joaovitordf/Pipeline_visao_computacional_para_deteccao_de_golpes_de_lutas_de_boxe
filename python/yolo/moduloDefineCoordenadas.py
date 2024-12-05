

def cabeca_coordenadas(imagem, keypoints):
    x = int(keypoints[0][0] * imagem.shape[1])

    y = int(keypoints[0][1] * imagem.shape[0])

    return x, y