"""
Passo 1: lê o dataset raw e junta todas as imagens rotuladas numa pasta só.

Regras:
  - imagem COM json (com caixas)  -> um recorte por caixa, rotulado pelo label do json
  - imagem SEM json numa pasta de classe -> imagem inteira, rotulada pelo nome da pasta
  - imagem SEM json na SALUS_data -> ignorada (não tem como saber a classe)

Saída: GROUPED_DIR/<label>__<pasta>__<foto>__<n>.jpg
"""
import os
import json
from PIL import Image

RAW_DIR = "raw_dataset"
GROUPED_DIR = "grouped"
CLASSES = ["background", "biological", "metal", "paper", "plastic"]
MARGEM = 0.10  # folga em volta da caixa (10% de cada lado)


def recortar(img, pontos):
    (x1, y1), (x2, y2) = pontos
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    mx, my = (x2 - x1) * MARGEM, (y2 - y1) * MARGEM
    caixa = (max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my))
    return img.crop(tuple(int(v) for v in caixa))


os.makedirs(GROUPED_DIR, exist_ok=True)
contagem = {c: 0 for c in CLASSES}
ignoradas = 0

for pasta in sorted(os.listdir(RAW_DIR)):
    caminho_pasta = os.path.join(RAW_DIR, pasta)
    if not os.path.isdir(caminho_pasta):
        continue

    for arquivo in sorted(os.listdir(caminho_pasta)):
        if not arquivo.lower().endswith(".jpg"):
            continue
        nome = os.path.splitext(arquivo)[0]
        caminho_img = os.path.join(caminho_pasta, arquivo)
        caminho_json = os.path.join(caminho_pasta, nome + ".json")

        caixas = []
        if os.path.exists(caminho_json):
            with open(caminho_json, encoding="utf-8") as f:
                caixas = [s for s in json.load(f).get("shapes", [])
                          if s.get("shape_type") == "rectangle"]

        img = Image.open(caminho_img).convert("RGB")

        if caixas:
            # com anotação: um recorte por caixa
            for i, s in enumerate(caixas):
                label = s["label"].strip().lower()
                if label not in CLASSES:
                    continue
                saida = f"{label}__{pasta}__{nome}__{i}.jpg"
                recortar(img, s["points"]).save(os.path.join(GROUPED_DIR, saida), quality=95)
                contagem[label] += 1
        elif pasta.lower() in CLASSES:
            # sem anotação, mas a pasta diz a classe: imagem inteira
            label = pasta.lower()
            saida = f"{label}__{pasta}__{nome}__0.jpg"
            img.save(os.path.join(GROUPED_DIR, saida), quality=95)
            contagem[label] += 1
        else:
            # sem anotação e sem classe na pasta (SALUS_data)
            ignoradas += 1

print("Imagens geradas por classe:")
for c in CLASSES:
    print(f"  {c:<12}{contagem[c]}")
print(f"Ignoradas (sem anotação fora das pastas de classe): {ignoradas}")
