def nose_coordenadas(imagem, keypoints):
    x = int(keypoints[0][0] * imagem.shape[1])
    y = int(keypoints[0][1] * imagem.shape[0])
    return x, y


def left_eye_coordenadas(imagem, keypoints):
    x = int(keypoints[1][0] * imagem.shape[1])
    y = int(keypoints[1][1] * imagem.shape[0])
    return x, y


def right_eye_coordenadas(imagem, keypoints):
    x = int(keypoints[2][0] * imagem.shape[1])
    y = int(keypoints[2][1] * imagem.shape[0])
    return x, y


def left_ear_coordenadas(imagem, keypoints):
    x = int(keypoints[3][0] * imagem.shape[1])
    y = int(keypoints[3][1] * imagem.shape[0])
    return x, y


def right_ear_coordenadas(imagem, keypoints):
    x = int(keypoints[4][0] * imagem.shape[1])
    y = int(keypoints[4][1] * imagem.shape[0])
    return x, y


def left_shoulder_coordenadas(imagem, keypoints):
    x = int(keypoints[5][0] * imagem.shape[1])
    y = int(keypoints[5][1] * imagem.shape[0])
    return x, y


def right_shoulder_coordenadas(imagem, keypoints):
    x = int(keypoints[6][0] * imagem.shape[1])
    y = int(keypoints[6][1] * imagem.shape[0])
    return x, y


def left_elbow_coordenadas(imagem, keypoints):
    x = int(keypoints[7][0] * imagem.shape[1])
    y = int(keypoints[7][1] * imagem.shape[0])
    return x, y


def right_elbow_coordenadas(imagem, keypoints):
    x = int(keypoints[8][0] * imagem.shape[1])
    y = int(keypoints[8][1] * imagem.shape[0])
    return x, y


def left_wrist_coordenadas(imagem, keypoints):
    x = int(keypoints[9][0] * imagem.shape[1])
    y = int(keypoints[9][1] * imagem.shape[0])
    return x, y


def right_wrist_coordenadas(imagem, keypoints):
    x = int(keypoints[10][0] * imagem.shape[1])
    y = int(keypoints[10][1] * imagem.shape[0])
    return x, y


def left_hip_coordenadas(imagem, keypoints):
    x = int(keypoints[11][0] * imagem.shape[1])
    y = int(keypoints[11][1] * imagem.shape[0])
    return x, y


def right_hip_coordenadas(imagem, keypoints):
    x = int(keypoints[12][0] * imagem.shape[1])
    y = int(keypoints[12][1] * imagem.shape[0])
    return x, y


def left_knee_coordenadas(imagem, keypoints):
    x = int(keypoints[13][0] * imagem.shape[1])
    y = int(keypoints[13][1] * imagem.shape[0])
    return x, y


def right_knee_coordenadas(imagem, keypoints):
    x = int(keypoints[14][0] * imagem.shape[1])
    y = int(keypoints[14][1] * imagem.shape[0])
    return x, y


def left_ankle_coordenadas(imagem, keypoints):
    x = int(keypoints[15][0] * imagem.shape[1])
    y = int(keypoints[15][1] * imagem.shape[0])
    return x, y


def right_ankle_coordenadas(imagem, keypoints):
    x = int(keypoints[16][0] * imagem.shape[1])
    y = int(keypoints[16][1] * imagem.shape[0])
    return x, y
