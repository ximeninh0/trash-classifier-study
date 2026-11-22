"""
Passo 3: treina uma ResNet-18 pré-treinada no ImageNet para classificar resíduos.

Lê SPLIT_DIR/train, SPLIT_DIR/val e SPLIT_DIR/test (formato ImageFolder,
gerado pelo 02_split.py) e salva o melhor modelo em SAIDA_DIR.

Experimentos controlados pelas variáveis abaixo:
  MODO    = "cabeca"    -> congela a rede, treina só a última camada
            "finetune"  -> treina a rede inteira (lr menor na base)
  AUGMENT = True/False  -> liga/desliga data augmentation no treino
"""
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ----------------------------- configuração -----------------------------
SPLIT_DIR = "split"
MODO = "cabeca"          # "cabeca" ou "finetune"
AUGMENT = False
EPOCAS = 15
BATCH = 32
LR_CABECA = 1e-3         # learning rate da nova camada final
LR_BASE = 1e-4           # learning rate do resto da rede (só no finetune)
SEED = 42

NOME = f"resnet18_{MODO}_{'aug' if AUGMENT else 'noaug'}"
SAIDA_DIR = os.path.join("runs", NOME)
# ------------------------------------------------------------------------

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAIDA_DIR, exist_ok=True)
print(f"Experimento: {NOME} | device: {device}")

# 1) Transforms: tudo vira 224x224 normalizado como no ImageNet
MEDIA, DESVIO = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

tf_avaliacao = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])

if AUGMENT:
    tf_treino = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])
else:
    tf_treino = tf_avaliacao

# 2) Dataset e DataLoader: o nome da pasta é o rótulo
ds_treino = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), tf_treino)
ds_val = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"), tf_avaliacao)
ds_teste = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), tf_avaliacao)
classes = ds_treino.classes
print("Classes:", classes)

dl_treino = DataLoader(ds_treino, batch_size=BATCH, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=BATCH)
dl_teste = DataLoader(ds_teste, batch_size=BATCH)

# 3) Modelo: ResNet-18 com pesos do ImageNet, trocando a última camada
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))  # 1000 saídas -> 5
model = model.to(device)

# 4) Congelar ou não, e o otimizador correspondente
if MODO == "cabeca":
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    otimizador = torch.optim.Adam(model.fc.parameters(), lr=LR_CABECA)
else:
    base = [p for nome, p in model.named_parameters() if not nome.startswith("fc.")]
    otimizador = torch.optim.Adam([
        {"params": base, "lr": LR_BASE},
        {"params": model.fc.parameters(), "lr": LR_CABECA},
    ])

treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parâmetros treináveis: {treinaveis:,}")

# 5) Loss com peso por classe: classes raras (ex.: metal) pesam mais
contagem = torch.bincount(torch.tensor(ds_treino.targets), minlength=len(classes)).float()
pesos = contagem.sum() / (len(classes) * contagem)
loss_fn = nn.CrossEntropyLoss(weight=pesos.to(device))


def avaliar(loader):
    """Roda o modelo sem gradiente e devolve (rótulos reais, rótulos preditos)."""
    model.eval()
    reais, preditos = [], []
    with torch.no_grad():
        for x, y in loader:
            saida = model(x.to(device))
            preditos += saida.argmax(1).cpu().tolist()
            reais += y.tolist()
    return reais, preditos


# 6) Loop de treino
melhor_f1 = -1
for epoca in range(1, EPOCAS + 1):
    model.train()
    soma_loss = 0
    for x, y in dl_treino:
        x, y = x.to(device), y.to(device)
        otimizador.zero_grad()
        loss = loss_fn(model(x), y)   # forward + loss
        loss.backward()               # calcula gradientes
        otimizador.step()             # atualiza pesos
        soma_loss += loss.item() * len(y)

    reais, preditos = avaliar(dl_val)
    acc = sum(r == p for r, p in zip(reais, preditos)) / len(reais)
    f1 = f1_score(reais, preditos, average="macro")
    print(f"Época {epoca:02d} | loss {soma_loss / len(ds_treino):.4f} | val acc {acc:.3f} | val F1 {f1:.3f}")

    if f1 > melhor_f1:
        melhor_f1 = f1
        torch.save(model.state_dict(), os.path.join(SAIDA_DIR, "best.pt"))

# 7) Teste final com o melhor modelo
model.load_state_dict(torch.load(os.path.join(SAIDA_DIR, "best.pt"), map_location=device))
reais, preditos = avaliar(dl_teste)

relatorio = classification_report(reais, preditos, labels=range(len(classes)),
                                  target_names=classes, zero_division=0)
matriz = confusion_matrix(reais, preditos, labels=range(len(classes)))

print("\n=== TESTE ===")
print(relatorio)
print("Matriz de confusão (linhas = real, colunas = predito):")
print(classes)
print(matriz)

with open(os.path.join(SAIDA_DIR, "resultado_teste.txt"), "w", encoding="utf-8") as f:
    f.write(f"{NOME}\nmelhor F1 validação: {melhor_f1:.3f}\n\n{relatorio}\n{classes}\n{matriz}\n")
