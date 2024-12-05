class Lutador:
    def __init__(self, identificador, cor, socos=0, coordenadas=None, box=None, roi_cabeca=None, roi_tronco=None, roi_linha_cintura=None, roi_mao_esquerda=None, roi_mao_direita=None):
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

        self.maoDireitaCabeca = False
        self.maoEsquerdaCabeca = False

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
