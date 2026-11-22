"""
Linha de base SEM transfer learning: ResNet-18 com pesos aleatórios,
treinada do zero (sem freeze, que não faz sentido com pesos aleatórios).

Compara com resnet18 caso 1 / nenhum para medir o ganho do pré-treino no ImageNet.

Rodar a partir da raiz do repositório:  python 03_train_scratch.py
"""
from treino import treinar

BACKBONE = "resnet18"

CASOS = [
    # (augment, freeze)
    (True, False),    # 1
    (False, False),   # nenhum
]

for augment, freeze in CASOS:
    treinar(BACKBONE, augment=augment, freeze=freeze, pretreinado=False)
