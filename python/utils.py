import os
import platform


def retorna_diretorio(arquivo):
    # Recebe qual sistema operacional esta sendo utilizado
    sistema_operacional = platform.system()
    # Obter o caminho do diretório atual
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Navegar para o diretório pai
    dir_path_pai = os.path.dirname(dir_path)
    if sistema_operacional == "Windows":
        dir_imagem = dir_path_pai + "\\python\\" + arquivo
        # Dependendo da versao do python (aparentemente) é utilizado / ou \\ ou \, a linha abaixo resolve isso
        dir_final = dir_imagem.replace("\\", "/")
    # Se for linux ou macos
    else:
        dir_final = dir_path_pai + "/python/" + arquivo

    return dir_final