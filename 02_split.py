"""
Passo 2: pega a pasta "grouped" e divide em treino / validação / teste,
no formato que o ImageFolder do PyTorch espera:

  SPLIT_DIR/train/<label>/...   SPLIT_DIR/val/<label>/...   SPLIT_DIR/test/<label>/...

Todos os recortes da mesma foto original vão juntos para o mesmo conjunto,
para não haver vazamento entre treino e teste.
"""
import os
import random
import shutil

GROUPED_DIR = "grouped"
SPLIT_DIR = "split"
PROPORCOES = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

# 1) agrupa os arquivos por classe e, dentro da classe, pela foto de origem
#    nome do arquivo: <label>__<pasta>__<foto>__<n>.jpg
grupos = {}  # {label: {foto_origem: [arquivos]}}
for arquivo in sorted(os.listdir(GROUPED_DIR)):
    if not arquivo.endswith(".jpg"):
        continue
    label, pasta, foto, _ = arquivo[:-4].split("__")
    origem = f"{pasta}/{foto}"
    grupos.setdefault(label, {}).setdefault(origem, []).append(arquivo)

# 2) para cada classe, embaralha as fotos de origem e reparte pelas proporções
random.seed(SEED)
contagem = {}
for label, fotos in sorted(grupos.items()):
    origens = sorted(fotos)
    random.shuffle(origens)

    n = len(origens)
    n_train = int(n * PROPORCOES["train"])
    n_val = int(n * PROPORCOES["val"])
    partes = {
        "train": origens[:n_train],
        "val": origens[n_train:n_train + n_val],
        "test": origens[n_train + n_val:],
    }

    # 3) copia os arquivos para split/<conjunto>/<label>/
    for conjunto, lista in partes.items():
        destino = os.path.join(SPLIT_DIR, conjunto, label)
        os.makedirs(destino, exist_ok=True)
        total = 0
        for origem in lista:
            for arquivo in fotos[origem]:
                shutil.copy(os.path.join(GROUPED_DIR, arquivo), destino)
                total += 1
        contagem[(conjunto, label)] = total

print(f"{'classe':<12}{'train':>8}{'val':>8}{'test':>8}")
for label in sorted(grupos):
    print(f"{label:<12}" + "".join(f"{contagem[(c, label)]:>8}" for c in PROPORCOES))
