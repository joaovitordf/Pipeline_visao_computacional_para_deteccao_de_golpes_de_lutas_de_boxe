def colisao(r1, r2):
    # r1 e r2 são tuplas (x1, y1, x2, y2)
    x1_min, y1_min, x1_max, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2

    # Verifica se um retângulo está à esquerda do outro
    if x1_max < x2_min or x2_max < x1_min:
        return False

    # Verifica se um retângulo está acima do outro
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True