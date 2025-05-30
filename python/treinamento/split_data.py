import os
import splitfolders

# pasta atual contém duas subpastas: images/ e labels/
INPUT = os.path.abspath(os.path.dirname(__file__))
# gera ../dataset/train/(images,labels) e ../dataset/val/(images,labels)
OUTPUT = os.path.abspath(os.path.join(INPUT, os.pardir, "dataset"))

splitfolders.ratio(
    INPUT,              # pasta “treinamento/” com images/ e labels/
    output=OUTPUT,      # cria a pasta “dataset/” ao lado de “treinamento/”
    seed=1337,
    ratio=(.8, .2),     # 80% train, 20% val
    group_prefix=None,
    move=False          # se True move em vez de copiar
)