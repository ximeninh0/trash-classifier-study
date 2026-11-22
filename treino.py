"""
Lógica de treino COMPARTILHADA por todos os scripts 03_train_*.py.

Não é executado diretamente: cada script de modelo importa a função treinar()
e a chama para cada caso do fluxo. Assim todos os experimentos usam exatamente
o mesmo split, as mesmas transformações, hiperparâmetros e métricas.

Casos do fluxo:
  (1)   augmentation
  (2)   freeze (congela a rede, treina só a última camada)
  1+2, 1, 2, nenhum -> 4 casos por rede
"""
import csv
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ------------------- configuração comum (NÃO alterar) -------------------
SPLIT_DIR = "split"
RUNS_DIR = "runs"
EPOCAS = 15
BATCH = 32
LR_CABECA = 1e-3         # learning rate da nova camada final
LR_BASE = 1e-4           # learning rate do resto da rede (fine-tuning)
SEED = 42
# ------------------------------------------------------------------------

# Onde fica a última camada (a "cabeça") em cada arquitetura do Torchvision.
# Ex.: na ResNet é model.fc; na EfficientNet é model.classifier[1].
CABECAS = {
    "resnet18": ("fc", None),
    "resnet50": ("fc", None),
    "efficientnet_b0": ("classifier", 1),
    "mobilenet_v3_large": ("classifier", 3),
    "convnext_tiny": ("classifier", 2),
}

MEDIA, DESVIO = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalização do ImageNet


def nome_caso(augment, freeze):
    if augment and freeze:
        return "1+2"
    if augment:
        return "1"
    if freeze:
        return "2"
    return "nenhum"


def criar_transforms(augment):
    """(1) Augmentation: transformações aleatórias aplicadas só no treino."""
    avaliacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])
    if not augment:
        return avaliacao, avaliacao
    treino = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])
    return treino, avaliacao


def criar_modelo(backbone, n_classes, pretreinado):
    """Carrega a arquitetura (com ou sem pesos do ImageNet) e troca a última camada."""
    model = models.get_model(backbone, weights="DEFAULT" if pretreinado else None)
    atributo, indice = CABECAS[backbone]
    if indice is None:                               # ex.: ResNet -> model.fc
        antiga = getattr(model, atributo)
        cabeca = nn.Linear(antiga.in_features, n_classes)
        setattr(model, atributo, cabeca)
    else:                                            # ex.: EfficientNet -> model.classifier[1]
        antiga = getattr(model, atributo)[indice]
        cabeca = nn.Linear(antiga.in_features, n_classes)
        getattr(model, atributo)[indice] = cabeca
    return model, cabeca


def avaliar(model, loader, device):
    """Roda o modelo sem gradiente e devolve (rótulos reais, rótulos preditos)."""
    model.eval()
    reais, preditos = [], []
    with torch.no_grad():
        for x, y in loader:
            preditos += model(x.to(device)).argmax(1).cpu().tolist()
            reais += y.tolist()
    return reais, preditos


def treinar(backbone, augment, freeze, pretreinado=True):
    """Treina e avalia UM caso do fluxo. Resultados em runs/<nome>/ e runs/resultados.csv."""
    if not pretreinado and freeze:
        raise ValueError("Congelar pesos aleatórios não faz sentido (freeze exige pretreinado=True)")

    caso = nome_caso(augment, freeze)
    nome = f"{backbone}_{'aug' if augment else 'noaug'}_{'freeze' if freeze else 'nofreeze'}"
    if not pretreinado:
        nome += "_scratch"
    saida = os.path.join(RUNS_DIR, nome)
    os.makedirs(saida, exist_ok=True)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===== {nome} (caso {caso}) | device: {device} =====")

    # Dados: ImageFolder usa o nome da pasta como rótulo
    tf_treino, tf_avaliacao = criar_transforms(augment)
    ds_treino = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), tf_treino)
    ds_val = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"), tf_avaliacao)
    ds_teste = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), tf_avaliacao)
    classes = ds_treino.classes

    dl_treino = DataLoader(ds_treino, batch_size=BATCH, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH)
    dl_teste = DataLoader(ds_teste, batch_size=BATCH)

    model, cabeca = criar_modelo(backbone, len(classes), pretreinado)
    model = model.to(device)

    # (2) Freeze: sem gradiente na base -> só a cabeça aprende
    params_cabeca = list(cabeca.parameters())
    ids_cabeca = {id(p) for p in params_cabeca}
    params_base = [p for p in model.parameters() if id(p) not in ids_cabeca]

    if freeze:
        for p in params_base:
            p.requires_grad = False
        otimizador = torch.optim.Adam(params_cabeca, lr=LR_CABECA)
    elif pretreinado:
        otimizador = torch.optim.Adam([
            {"params": params_base, "lr": LR_BASE},
            {"params": params_cabeca, "lr": LR_CABECA},
        ])
    else:
        # do zero: não há conhecimento a preservar, mesma lr para tudo
        otimizador = torch.optim.Adam(model.parameters(), lr=LR_CABECA)

    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {treinaveis:,} de {total:,}")

    # Loss com peso por classe: classes raras (ex.: metal) pesam mais
    contagem = torch.bincount(torch.tensor(ds_treino.targets), minlength=len(classes)).float()
    pesos = contagem.sum() / (len(classes) * contagem.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=pesos.to(device))

    # Loop de treino
    melhor_f1 = -1
    inicio = time.time()
    for epoca in range(1, EPOCAS + 1):
        model.train()
        if freeze:  # rede congelada: BatchNorm também não atualiza estatísticas
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        soma_loss = 0
        for x, y in dl_treino:
            x, y = x.to(device), y.to(device)
            otimizador.zero_grad()
            loss = loss_fn(model(x), y)   # forward + loss
            loss.backward()               # calcula gradientes
            otimizador.step()             # atualiza pesos
            soma_loss += loss.item() * len(y)

        reais, preditos = avaliar(model, dl_val, device)
        acc = sum(r == p for r, p in zip(reais, preditos)) / len(reais)
        f1 = f1_score(reais, preditos, average="macro")
        print(f"Época {epoca:02d} | loss {soma_loss / len(ds_treino):.4f} | val acc {acc:.3f} | val F1 {f1:.3f}")

        if f1 > melhor_f1:
            melhor_f1 = f1
            torch.save(model.state_dict(), os.path.join(saida, "best.pt"))
    tempo = time.time() - inicio

    # Teste final com o melhor modelo (escolhido pela validação)
    model.load_state_dict(torch.load(os.path.join(saida, "best.pt"), map_location=device))
    reais, preditos = avaliar(model, dl_teste, device)
    relatorio = classification_report(reais, preditos, labels=range(len(classes)),
                                      target_names=classes, zero_division=0)
    matriz = confusion_matrix(reais, preditos, labels=range(len(classes)))
    acc_teste = sum(r == p for r, p in zip(reais, preditos)) / len(reais)
    f1_teste = f1_score(reais, preditos, average="macro")

    print("\n--- TESTE ---")
    print(relatorio)
    print("Matriz de confusão (linhas = real, colunas = predito):")
    print(classes)
    print(matriz)

    with open(os.path.join(saida, "resultado_teste.txt"), "w", encoding="utf-8") as f:
        f.write(f"{nome} (caso {caso})\nmelhor F1 validação: {melhor_f1:.3f}\n\n"
                f"{relatorio}\n{classes}\n{matriz}\n")

    # Uma linha na tabela geral (runs/resultados.csv)
    tabela = os.path.join(RUNS_DIR, "resultados.csv")
    novo = not os.path.exists(tabela)
    with open(tabela, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["experimento", "backbone", "caso", "pretreinado", "augment", "freeze",
                        "params_treinaveis", "tempo_s", "val_f1", "test_acc", "test_f1"])
        w.writerow([nome, backbone, caso, pretreinado, augment, freeze, treinaveis,
                    round(tempo), f"{melhor_f1:.3f}", f"{acc_teste:.3f}", f"{f1_teste:.3f}"])
