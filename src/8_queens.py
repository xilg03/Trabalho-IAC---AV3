import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.patheffects as path_effects # Import path_effects

def calcular_conflitos(x):
    conflitos = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                conflitos += 1
    return conflitos

def funcao_objetivo(x):
    return 28 - calcular_conflitos(x)

T0 = 100  # Temperatura inicial definida no trabalho
def decaimento_temperatura(T, alpha=0.95):
    return T * alpha

def perturbar_estado(x):
    novo = x.copy()
    i, j = random.sample(range(8), 2)
    novo[i], novo[j] = novo[j], novo[i]
    return novo

def plotar_tabuleiro_8rainhas(solucao):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Cores madeira
    marrom = "#8B4513"
    bege = "#F5DEB3"

    # Criar matriz do tabuleiro
    tabuleiro = np.add.outer(range(8), range(8)) % 2

    for i in range(8):
        for j in range(8):
            cor = marrom if tabuleiro[i, j] == 0 else bege
            ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=cor))

    # Rainha branca com contorno preto (efeito visual forte)
    for col, linha in enumerate(solucao):
        texto = ax.text(
            col + 0.5, 7 - (linha - 1) + 0.5,
            '♕',
            ha='center',
            va='center',
            fontsize=36,
            color='#000000',  # branco forte
            weight='bold'
        )
        texto.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white'),
            path_effects.Normal()
        ])

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title("Tabuleiro das 8 Rainhas", fontsize=16)
    plt.tight_layout()
    plt.show()

def simulated_annealing(T0=1000, max_iter=10000, alpha=0.95):
    x = random.sample(range(1, 9), 8)  # Solução inicial sem rainhas na mesma linha
    f_best = funcao_objetivo(x)
    x_best = x.copy()
    T = T0
    historico_T = []

    for _ in range(max_iter):
        historico_T.append(T)
        x_new = perturbar_estado(x)
        f_new = funcao_objetivo(x_new)
        delta = f_new - funcao_objetivo(x)

        if delta > 0 or random.random() < math.exp(delta / T):
            x = x_new

        if funcao_objetivo(x) > f_best:
            f_best = funcao_objetivo(x)
            x_best = x.copy()

        T = decaimento_temperatura(T, alpha)

        if f_best == 28:
            break

    return x_best, f_best, historico_T

def encontrar_92_solucoes():
    solucoes_unicas = set()
    total_execucoes = 0
    inicio = time.time()

    while len(solucoes_unicas) < 92:
        solucao, aptidao, historico_T = simulated_annealing()
        total_execucoes += 1

        if aptidao == 28:
            tupla = tuple(solucao)
            if tupla not in solucoes_unicas:
                solucoes_unicas.add(tupla)
                print(f"✅ Solução {len(solucoes_unicas)} encontrada na execução {total_execucoes}: {solucao}")

                # Visualização opcional
                plotar_tabuleiro_8rainhas(solucao)
                plt.figure(figsize=(8, 4))
                plt.plot(historico_T)
                plt.title(f'Decaimento da Temperatura - Solução {len(solucoes_unicas)}')
                plt.xlabel('Iteração')
                plt.ylabel('Temperatura')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    fim = time.time()
    print(f"\n✔️ Total de 92 soluções únicas encontradas.")
    print(f"🔁 Total de execuções necessárias: {total_execucoes}")
    print(f"⏱️ Tempo total: {fim - inicio:.2f} segundos")

    return solucoes_unicas, total_execucoes

# Executar a função para encontrar todas as 92 soluções únicas
solucoes_finais, total_execucoes = encontrar_92_solucoes()

# Exibir as soluções encontradas
print(f"\n✅ Total de 92 soluções encontradas (detalhadas):\n")
for i, sol in enumerate(sorted(solucoes_finais), 1):
    print(f"{i:02d}: {sol}")

# Exibir o total de execuções realizadas
print(f"\n🔁 Total de execuções necessárias para encontrar todas as soluções: {total_execucoes}")

