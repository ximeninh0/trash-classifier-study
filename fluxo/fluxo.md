# Fluxo dos experimentos

**Casos:** `1` = data augmentation · `2` = freeze (só a última camada treina) · `nenhum` = fine-tuning completo sem augmentation

```mermaid
flowchart TD
    ORG["Organizar dataset<br/>(01_group.py)"] --> SPLIT["Separar treino / val / teste<br/>(02_split.py)"]

    SPLIT --> CNX["ConvNeXt-Tiny"]
    SPLIT --> EFF["EfficientNet-B0"]
    SPLIT --> R18["ResNet-18"]
    SPLIT --> R50["ResNet-50"]
    SPLIT --> MOB["MobileNetV3-L"]
    SPLIT -.-> SCR["ResNet-18 do zero<br/>(sem transfer learning)"]

    CNX --> CNX12["1+2"] & CNX1["1"] & CNX2["2"] & CNX0["nenhum"]
    EFF --> EFF12["1+2"] & EFF1["1"] & EFF2["2"] & EFF0["nenhum"]
    R18 --> R1812["1+2"] & R181["1"] & R182["2"] & R180["nenhum"]
    R50 --> R5012["1+2"] & R501["1"] & R502["2"] & R500["nenhum"]
    MOB --> MOB12["1+2"] & MOB1["1"] & MOB2["2"] & MOB0["nenhum"]
    SCR -.-> SCR1["1"] & SCR0["nenhum"]
```

## Versão em texto

```
                         ┌──────────────────┐
                         │ Organizar dataset│  01_group.py
                         └────────┬─────────┘
                                  ▼
                       ┌────────────────────┐
                       │ Separar T / V / T  │  02_split.py
                       └─────────┬──────────┘
   ┌──────────────┬──────────────┼──────────────┬──────────────┬ ─ ─ ─ ─ ─ ─ ─ ─┐
   ▼              ▼              ▼              ▼              ▼                ▼
ConvNeXt-Tiny  EfficientNet-B0  ResNet-18     ResNet-50    MobileNetV3-L   ResNet-18 do zero
                                                                         (sem transfer learning)
1+2 1 2 —      1+2 1 2 —      1+2 1 2 —      1+2 1 2 —      1+2 1 2 —          1  —
```

## Experimentos por script (um por pessoa)

| Script | Rede | Casos | Nº de treinos |
|---|---|---|---|
| `03_train_convnext_tiny.py` | ConvNeXt-Tiny (ImageNet) | 1+2, 1, 2, nenhum | 4 |
| `03_train_efficientnet_b0.py` | EfficientNet-B0 (ImageNet) | 1+2, 1, 2, nenhum | 4 |
| `03_train_resnet18.py` | ResNet-18 (ImageNet) | 1+2, 1, 2, nenhum | 4 |
| `03_train_resnet50.py` | ResNet-50 (ImageNet) | 1+2, 1, 2, nenhum | 4 |
| `03_train_mobilenet_v3_large.py` | MobileNetV3-Large (ImageNet) | 1+2, 1, 2, nenhum | 4 |
| `03_train_scratch.py` | ResNet-18 (pesos aleatórios) | 1, nenhum | 2 |
| **Total** | | | **22** |
