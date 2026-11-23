"""Da raiz, execute python 04_consolidar.py após receber runs/resultados_<nome>.csv.

Saídas em artigo/gabriel/gerado/. As figuras requerem matplotlib.
"""
import ast
import csv
import re
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs")
OUT = Path("artigo/gabriel/gerado")
REDES = [
    ("ximenes", "resnet18", "ResNet-18"),
    ("caio", "resnet50", "ResNet-50"),
    ("matheus", "efficientnet_b0", "EfficientNet-B0"),
    ("ito", "mobilenet_v3_large", "MobileNetV3-Large"),
    ("rafa", "convnext_tiny", "ConvNeXt-Tiny"),
    ("gabriel", "resnet18", "ResNet-18 do zero"),
]
CASOS = ("1+2", "1", "2", "nenhum")
COLUNAS = "experimento backbone caso pretreinado augment freeze params_treinaveis tempo_s val_f1 test_acc test_f1".split()
CLASSES = ["background", "biological", "metal", "paper", "plastic"]


def ler():
    dados = {}
    for pessoa, backbone, rede in REDES:
        scratch = pessoa == "gabriel"
        casos = ("1", "nenhum") if scratch else CASOS
        arquivo = RUNS / f"resultados_{pessoa}.csv"
        linhas = {}
        if arquivo.exists():
            with arquivo.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    caso = row["caso"]
                    aug, freeze = caso in ("1+2", "1"), caso in ("1+2", "2")
                    nome = f"{backbone}_{'aug' if aug else 'noaug'}_{'freeze' if freeze else 'nofreeze'}"
                    if scratch:
                        nome += "_scratch"
                    esperado = (nome, backbone, str(not scratch), str(aug), str(freeze))
                    real = tuple(row[k] for k in ("experimento", "backbone", "pretreinado", "augment", "freeze"))
                    if caso not in casos or caso in linhas or real != esperado:
                        raise ValueError(f"{arquivo}: caso duplicado ou configuração divergente: {row}")
                    if not all(0 <= float(row[k]) <= 1 for k in ("test_f1", "test_acc", "val_f1")):
                        raise ValueError(f"{arquivo}: métrica inválida")
                    linhas[caso] = row
        for caso in casos:
            dados[rede, caso] = linhas.get(caso)
    return dados


def formatar(valor):
    return f"{valor:.3f}".replace(".", ",")


def matriz(row):
    rel = RUNS / row["experimento"] / "resultado_teste.txt"
    lines = rel.read_text(encoding="utf-8").splitlines()
    pos = lines.index(str(CLASSES))
    texto = " ".join(lines[pos + 1:])
    valores = ast.literal_eval(re.sub(r"(?<=[0-9\]])\s+(?=[0-9\[])", ",", texto))
    if len(valores) != 5 or any(len(linha) != 5 for linha in valores):
        raise ValueError(f"Matriz inválida em {rel}")
    return valores


def figura_matriz(row, chave, arquivo):
    valores = matriz(row)
    fig, ax = plt.subplots(figsize=(4.4, 3.5), layout="constrained")
    ax.imshow(valores, cmap="Blues")
    for i, linha in enumerate(valores):
        for j, n in enumerate(linha):
            ax.text(j, i, str(n), ha="center", va="center",
                    color="white" if n > max(map(max, valores)) / 2 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Classe predita", ylabel="Classe real", title="%s · caso %s" % chave)
    ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(OUT / arquivo)
    plt.close(fig)


def gerar():
    dados = ler()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tabela_geral.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rede", "caso", "situacao", *[k for k in COLUNAS if k != "caso"]])
        for (rede, caso), row in dados.items():
            w.writerow([rede, caso, "recebido" if row else "pendente",
                        *[(row or {}).get(k, "") for k in COLUNAS if k != "caso"]])

    tabela = [r"\begin{table}[t]\centering\caption{Desempenho por configuração. Traços indicam resultados pendentes.}",
              r"\label{tab:gabriel}\footnotesize\begin{tabular}{llrrr}\hline",
              r"Rede & Caso & F1 val. & Acur. teste & F1 teste \\ \hline"]
    for (rede, caso), row in dados.items():
        metrics = [formatar(float(row[k])) if row else "---" for k in ("val_f1", "test_acc", "test_f1")]
        tabela.append(" & ".join([rede, caso, *metrics]) + r" \\")
    tabela.append(r"\hline\end{tabular}\end{table}")
    (OUT / "tabela.tex").write_text("\n".join(tabela) + "\n", encoding="utf-8")

    f1 = {k: float(v["test_f1"]) for k, v in dados.items() if v}
    parag = [r"\section{Resultados}",
             "O F1 macro de teste atribui o mesmo peso a cada uma das cinco classes. "
             "Em cada treino, a época avaliada no teste foi selecionada pelo maior F1 macro de validação."]
    pendentes = len(dados) - len(f1)
    if pendentes:
        parag.append(f"Das 22 configurações previstas, {pendentes} aguardam entrega de métricas.")
    if f1:
        fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
        redes_recebidas = [r for _, _, r in REDES if any((r, c) in f1 for c in CASOS)]
        for i, caso in enumerate(CASOS):
            pontos = [(j, f1[rede, caso]) for j, rede in enumerate(redes_recebidas) if (rede, caso) in f1]
            if pontos:
                ax.scatter([j + (i - 1.5) * .12 for j, _ in pontos], [v for _, v in pontos],
                           marker=("o", "s", "^", "D")[i], label=f"Caso {caso}")
        ax.set(ylim=(-.03, 1.03), ylabel="F1 macro de teste", xlim=(-0.5, len(redes_recebidas) - 0.5), xticks=range(len(redes_recebidas)),
               xticklabels=[r.replace(" do zero", "\ndo zero") for r in redes_recebidas])
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.legend(ncol=4, fontsize=8)
        fig.savefig(OUT / "f1_por_rede.pdf")
        plt.close(fig)

    scratch = "ResNet-18 do zero"
    if all((scratch, c) in f1 for c in ("1", "nenhum")):
        a, b = (f1[scratch, c] for c in ("1", "nenhum"))
        parag.append(f"A ResNet-18 com inicialização aleatória obteve F1 macro de {formatar(a)} "
                     f"com aumento de dados e de {formatar(b)} sem aumento.")
        melhor = max(((scratch, c) for c in ("1", "nenhum")), key=f1.get)
        figura_matriz(dados[melhor], melhor, "matriz_gabriel.pdf")
        if a == b:
            parag.append("Os dois casos empataram à precisão registrada.")
        else:
            parag.append(f"O maior valor ocorreu no caso {melhor[1]}, com diferença de "
                         f"{formatar(abs(a - b))} entre as configurações.")
    conclusao = []
    for caso in ("1", "nenhum"):
        if ("ResNet-18", caso) in f1 and (scratch, caso) in f1:
            delta = f1["ResNet-18", caso] - f1[scratch, caso]
            conclusao.append(f"No caso {caso}, a diferença de F1 macro entre a ResNet-18 "
                             f"pré-treinada e a de inicialização aleatória foi {formatar(delta)}.")
    if not pendentes:
        for tratamento, pares in [("congelamento", [("1+2", "1"), ("2", "nenhum")]),
                                   ("aumento de dados", [("1+2", "2"), ("1", "nenhum")])]:
            deltas = [f1[r, a] - f1[r, b] for _, _, r in REDES[:-1] for a, b in pares]
            parag.append(f"Nas cinco redes pré-treinadas, a diferença média pareada de F1 macro "
                         f"ao aplicar {tratamento} foi {formatar(mean(deltas))}, "
                         f"com valores de {formatar(min(deltas))} a {formatar(max(deltas))}.")
        melhores = [k for k in f1 if f1[k] == max(f1.values())]
        nomes = ", ".join(f"{r} (caso {c})" for r, c in melhores)
        parag.insert(2, f"O maior F1 macro observado no teste foi {formatar(max(f1.values()))}, "
                     f"atingido por {nomes}.")
        figura_matriz(dados[melhores[0]], melhores[0], "matriz_confusao.pdf")
    parag += [r"\input{artigo/gabriel/gerado/tabela.tex}"]
    for arquivo, legenda, presente in [
            ("f1_por_rede", "F1 macro no teste por rede e configuração.", bool(f1)),
            ("matriz_confusao", "Matriz de confusão da configuração com maior F1 macro de teste.", not pendentes)]:
        if presente:
            parag.append(r"\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{artigo/gabriel/gerado/"
                         + arquivo + r".pdf}\caption{" + legenda + r"}\end{figure}")
    parag += [r"\section{Conclusão}", *conclusao]
    if pendentes:
        parag.append("A comparação global depende da consolidação dos resultados pendentes.")
    parag.append("A comparação de inicializações também incorpora as taxas de aprendizado da base "
                 "previstas em cada protocolo. A avaliação usa uma divisão única e uma semente por configuração. "
                 "Na divisão local, 238 conteúdos de fotos originais, identificados por SHA-256, "
                 "ocorrem tanto no treino quanto no teste. Entre as 1.257 entradas finais de teste, 19 são idênticas, byte a byte, a entradas de treino. Essa sobreposição e a proximidade entre capturas "
                 "limitam a interpretação dos escores como estimativas para imagens inéditas.")
    (OUT / "resultados_conclusao.tex").write_text("\n\n".join(parag) + "\n", encoding="utf-8")
    print(f"{len(f1)}/22 configurações com resultados em {OUT}")


if __name__ == "__main__":
    gerar()
