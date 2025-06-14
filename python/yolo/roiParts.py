from python.yolo.moduloDefineCoordenadas import nose_coordenadas

# Fatores dinâmicos para definição dos ROIs
HEAD_ROI_FACTOR = 0.02  # 2% da dimensão para a cabeça
HAND_ROI_WIDTH_FACTOR = 0.03  # 3% da largura para as mãos
HAND_ROI_HEIGHT_FACTOR = 0.02  # 2% da altura para as mãos
WAIST_ROI_WIDTH_FACTOR = 0.01  # 1% da largura para ampliar a região da cintura
MIN_TRUNK_WIDTH_FACTOR = 0.08  # Largura mínima do tronco em relação à largura da imagem

# Multiplicador para imagens em modo retrato (altura > largura)
PORTRAIT_WIDTH_MULTIPLIER = 1.2


def adjust_width_factor(imagem, factor):
    if imagem.shape[0] > imagem.shape[1]:
        return factor * PORTRAIT_WIDTH_MULTIPLIER
    return factor


def roi_cabeca(imagem, keypoints_numpy):
    (x, y) = nose_coordenadas(imagem, keypoints_numpy)
    if x != 0 and y != 0:
        # Apenas o fator de largura é ajustado para o modo retrato
        adjusted_head_factor = adjust_width_factor(imagem, HEAD_ROI_FACTOR)
        offset_x = int(imagem.shape[1] * adjusted_head_factor)
        offset_y = int(imagem.shape[0] * HEAD_ROI_FACTOR)  # fator de altura inalterado
        start_point = (max(x - offset_x, 0), max(y - offset_y, 0))
        end_point = (min(x + offset_x, imagem.shape[1]), min(y + offset_y, imagem.shape[0]))
        return (start_point, end_point)
    return None


def roi_mao_esquerda(imagem, keypoints_numpy):
    # left-wrist: índice 9
    x = int(keypoints_numpy[9][0] * imagem.shape[1])
    y = int(keypoints_numpy[9][1] * imagem.shape[0])
    if x != 0 and y != 0:
        adjusted_hand_width_factor = adjust_width_factor(imagem, HAND_ROI_WIDTH_FACTOR)
        offset_x = int(imagem.shape[1] * adjusted_hand_width_factor)
        offset_y = int(imagem.shape[0] * HAND_ROI_HEIGHT_FACTOR)  # fator de altura inalterado
        start_point = (max(x - offset_x, 0), max(y - offset_y, 0))
        end_point = (min(x + offset_x, imagem.shape[1]), min(y + offset_y, imagem.shape[0]))
        return (start_point, end_point)
    return None


def roi_mao_direita(imagem, keypoints_numpy):
    # right-wrist: índice 10
    x = int(keypoints_numpy[10][0] * imagem.shape[1])
    y = int(keypoints_numpy[10][1] * imagem.shape[0])
    if x != 0 and y != 0:
        adjusted_hand_width_factor = adjust_width_factor(imagem, HAND_ROI_WIDTH_FACTOR)
        offset_x = int(imagem.shape[1] * adjusted_hand_width_factor)
        offset_y = int(imagem.shape[0] * HAND_ROI_HEIGHT_FACTOR)  # fator de altura inalterado
        start_point = (max(x - offset_x, 0), max(y - offset_y, 0))
        end_point = (min(x + offset_x, imagem.shape[1]), min(y + offset_y, imagem.shape[0]))
        return (start_point, end_point)
    return None


def roi_linha_cintura(imagem, keypoints_numpy):
    # vai da cintura esquerda ate o pe direito
    x1 = int(keypoints_numpy[11][0] * imagem.shape[1])
    y1 = int(keypoints_numpy[11][1] * imagem.shape[0])
    x2 = int(keypoints_numpy[16][0] * imagem.shape[1])
    y2 = int(keypoints_numpy[16][1] * imagem.shape[0])
    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
        adjusted_waist_factor = adjust_width_factor(imagem, WAIST_ROI_WIDTH_FACTOR)
        offset = int(imagem.shape[1] * adjusted_waist_factor)
        aux_min_x = max(min(x1, x2) - offset, 0)
        aux_max_x = min(max(x1, x2) + offset, imagem.shape[1])
        start_point = (aux_min_x, min(y1, y2))
        end_point = (aux_max_x, max(y1, y2))
        return (start_point, end_point)
    return None


def roi_tronco(imagem, keypoints_numpy):
    # right-shoulder: índice 6 e left-hip: índice 11
    x1 = int(keypoints_numpy[6][0] * imagem.shape[1])
    y1 = int(keypoints_numpy[6][1] * imagem.shape[0])
    x2 = int(keypoints_numpy[11][0] * imagem.shape[1])
    y2 = int(keypoints_numpy[11][1] * imagem.shape[0])
    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
        aux_min_x = min(x1, x2)
        aux_max_x = max(x1, x2)

        # fator de largura ajustado
        adjusted_trunk_factor = adjust_width_factor(imagem, MIN_TRUNK_WIDTH_FACTOR)
        min_width = int(imagem.shape[1] * adjusted_trunk_factor)

        current_width = aux_max_x - aux_min_x
        if current_width < min_width:
            # centro do tronco
            center_x = (aux_min_x + aux_max_x) // 2
            half_width = min_width // 2

            # expande metade para cada lado
            aux_min_x = center_x - half_width
            aux_max_x = center_x + half_width

            # clamp nos limites da imagem
            if aux_min_x < 0:
                aux_min_x = 0
                aux_max_x = min_width
            if aux_max_x > imagem.shape[1]:
                aux_max_x = imagem.shape[1]
                aux_min_x = imagem.shape[1] - min_width

        start_point = (max(aux_min_x, 0), min(y1, y2))
        end_point = (min(aux_max_x, imagem.shape[1]), max(y1, y2))
        return (start_point, end_point)
    return None
