class Lutador:
    def __init__(self, identificador, cor, socos=0, coordenadas=None, box=None, roi_cabeca=None, roi_tronco=None,
                 roi_linha_cintura=None, roi_mao_esquerda=None, roi_mao_direita=None, nose=None, left_eye=None,
                 right_eye=None, left_ear=None, right_ear=None, left_shoulder=None, right_shoulder=None,
                 left_elbow=None, right_elbow=None, left_wrist=None, right_wrist=None, left_hip=None,
                 right_hip=None, left_knee=None, right_knee=None, left_ankle=None, right_ankle=None):

        # 'identificador' diferencia os lutadores.
        self.identificador = identificador
        self.cor = cor

        self.primeira_execucao = 0

        # 'socos' guarda o número de socos de cada lutador.
        self.socos = socos

        self.coordenadas = coordenadas

        self.box = box

        self.roi_cabeca = roi_cabeca
        self.roi_tronco = roi_tronco
        self.roi_linha_cintura = roi_linha_cintura
        self.roi_mao_esquerda = roi_mao_esquerda
        self.roi_mao_direita = roi_mao_direita

        self.roi_mao_direitaCabeca = False
        self.roi_mao_esquerdaCabeca = False

        # COORDENADAS DE CADA KEYPOINT DO LUTADOR

        self.nose = nose  # Nariz
        self.left_eye = left_eye  # Olho esquerdo
        self.right_eye = right_eye  # Olho direito
        self.left_ear = left_ear  # Orelha esquerda
        self.right_ear = right_ear  # Orelha direita
        self.left_shoulder = left_shoulder  # Ombro esquerdo
        self.right_shoulder = right_shoulder  # Ombro direito
        self.left_elbow = left_elbow  # Cotovelo esquerdo
        self.right_elbow = right_elbow  # Cotovelo direito
        self.left_wrist = left_wrist  # Pulso esquerdo
        self.right_wrist = right_wrist  # Pulso direito
        self.left_hip = left_hip  # Quadril esquerdo
        self.right_hip = right_hip  # Quadril direito
        self.left_knee = left_knee  # Joelho esquerdo
        self.right_knee = right_knee  # Joelho direito
        self.left_ankle = left_ankle  # Tornozelo esquerdo
        self.right_ankle = right_ankle  # Tornozelo direito

    def soco(self):
        self.socos += 1
        print("Lutador: ", self.identificador, "Socos: ", self.socos)

    def __str__(self):
        return f"Lutador {self.identificador}: {self.socos} socos, {self.coordenadas}, {self.box}"


"""# Exemplo de uso:
lutador1 = Lutador(1)
lutador2 = Lutador(2)

# Lutador 1 dá 3 socos
for _ in range(3):
    lutador1.dar_soco()

# Lutador 2 dá 5 socos
for _ in range(5):
    lutador2.dar_soco()

print(lutador1)  # Output: Lutador 1: 3 socos
print(lutador2)  # Output: Lutador 2: 5 socos"""
