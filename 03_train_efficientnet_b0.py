"""
Treina a efficientnet_b0 nos 4 casos do fluxo:
  1+2 = augmentation + freeze | 1 = só augmentation | 2 = só freeze | nenhum

Rodar a partir da raiz do repositório:  python 03_train_efficientnet_b0.py
Para rodar só alguns casos, comente as linhas em CASOS.
"""
from treino import treinar

BACKBONE = "efficientnet_b0"

CASOS = [
    # (augment, freeze)
    (True, True),     # 1+2
    (True, False),    # 1
    (False, True),    # 2
    (False, False),   # nenhum
]

for augment, freeze in CASOS:
    treinar(BACKBONE, augment=augment, freeze=freeze, pretreinado=True)
