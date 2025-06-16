import numpy as np
import matplotlib.pyplot as plt

# Função auxiliar para plotar gráfico 3D de uma função contínua f(x1, x2)
def plot_function(f, domain, best_point=None, title="Função f(x1, x2)"):
    x1 = np.linspace(domain[0][0], domain[0][1], 300)
    x2 = np.linspace(domain[1][0], domain[1][1], 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.5)

    if best_point is not None:
        x1b, x2b = best_point
        ax.scatter(x1b, x2b, f(x1b, x2b), color='red', s=80, label='Melhor ponto')
        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Segue fielmente o pseudocódigo do professor Paulo Cirillo, com suporte a MINIMIZAÇÃO ou MAXIMIZAÇÃO.

class HillClimbing:
    def __init__(self, f, tipo_otimizacao, dominio, epsilon=0.1, max_it=1000, max_vizinhos=30):
        self.f = f
        self.tipo = tipo_otimizacao
        self.dom = np.array(dominio)
        self.epsilon = epsilon
        self.max_it = max_it
        self.max_vizinhos = max_vizinhos

    def run(self, retornar_caminho=False):
        x_best = np.copy(self.dom[:, 0])
        f_best = self.f(*x_best)
        caminho = [np.copy(x_best)]

        for i in range(self.max_it):
            melhorou = False
            for _ in range(self.max_vizinhos):
                x_cand = x_best + np.random.uniform(-self.epsilon, self.epsilon, size=2)
                x_cand = np.clip(x_cand, self.dom[:, 0], self.dom[:, 1])
                f_cand = self.f(*x_cand)

                if (self.tipo == "min" and f_cand < f_best) or (self.tipo == "max" and f_cand > f_best):
                    x_best, f_best = x_cand, f_cand
                    caminho.append(np.copy(x_best))
                    melhorou = True
                    break
            if not melhorou:
                break

        return caminho if retornar_caminho else x_best
    
# Baseado no pseudocódigo do professor, adaptado para problemas contínuos (f(x1, x2))

class LocalRandomSearch:
    def __init__(self, f, tipo_otimizacao, dominio, sigma=0.1, max_it=1000):
        self.f = f
        self.tipo = tipo_otimizacao
        self.dom = np.array(dominio)
        self.sigma = sigma
        self.max_it = max_it

    def run(self, retornar_caminho=False):
        x_best = np.random.uniform(self.dom[:, 0], self.dom[:, 1])
        f_best = self.f(*x_best)
        caminho = [np.copy(x_best)]

        for i in range(self.max_it):
            ruido = np.random.normal(0, self.sigma, size=2)
            x_cand = x_best + ruido
            x_cand = np.clip(x_cand, self.dom[:, 0], self.dom[:, 1])
            f_cand = self.f(*x_cand)

            if (self.tipo == "min" and f_cand < f_best) or (self.tipo == "max" and f_cand > f_best):
                x_best, f_best = x_cand, f_cand
                caminho.append(np.copy(x_best))

        return caminho if retornar_caminho else x_best

# Baseado no pseudocódigo do professor, utilizando amostragem totalmente aleatória no domínio

class GlobalRandomSearch:
    def __init__(self, f, tipo_otimizacao, dominio, max_it=1000):
        self.f = f
        self.tipo = tipo_otimizacao
        self.dom = np.array(dominio)
        self.max_it = max_it

    def run(self, retornar_caminho=False):
        x_best = np.random.uniform(self.dom[:, 0], self.dom[:, 1])
        f_best = self.f(*x_best)
        caminho = [np.copy(x_best)]

        for i in range(self.max_it):
            x_cand = np.random.uniform(self.dom[:, 0], self.dom[:, 1])
            f_cand = self.f(*x_cand)

            if (self.tipo == "min" and f_cand < f_best) or (self.tipo == "max" and f_cand > f_best):
                x_best, f_best = x_cand, f_cand
                caminho.append(np.copy(x_best))

        return caminho if retornar_caminho else x_best

# Cada função retorna o valor da função objetivo para x1, x2
# Também definimos uma lista com os domínios e se é um problema de minimização ou maximização

def f1(x1, x2):
    return x1**2 + x2**2

def f2(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

def f3(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)

def f4(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + \
           (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def f5(x1, x2):
    return (x1 * np.cos(x1)) / 20 + 2 * np.exp(-(x1**2) - (x2 - 1)**2) + 0.01 * x1 * x2

def f6(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

def f7(x1, x2):
    return -np.sin(x1) * np.sin(x1**2 / np.pi)**20 - np.sin(x2) * np.sin(2 * x2**2 / np.pi)**20

def f8(x1, x2):
    return - (x2 + 47) * np.sin(np.sqrt(abs(x1/2 + x2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - x2 - 47)))

# Lista com funções, domínios e tipo de otimização
funcoes_info = [
    (f1, [(-100, 100), (-100, 100)], 'min'),
    (f2, [(-2, 4), (-2, 5)], 'max'),
    (f3, [(-8, 8), (-8, 8)], 'min'),
    (f4, [(-5.12, 5.12), (-5.12, 5.12)], 'min'),
    (f5, [(-10, 10), (-10, 10)], 'max'),
    (f6, [(-1, 3), (-1, 3)], 'max'),
    (f7, [(0, np.pi), (0, np.pi)], 'min'),
    (f8, [(-200, 20), (-200, 20)], 'min')
]

import pandas as pd
from collections import Counter

# NOVA CÉLULA 6 - Consolidada: Execuções, parâmetros, moda, melhores soluções, caminhos e gráficos

# 1. Configurações e hiperparâmetros
f_obj, dominio, tipo = funcoes_info[0]
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# 2. Execuções com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# 3. Cálculo da melhor solução
def melhor_solucao(f, tipo, lista_solucoes):
    return min(lista_solucoes, key=lambda x: f(*x)) if tipo == 'min' else max(lista_solucoes, key=lambda x: f(*x))

melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# 4. Cálculo da moda
def calcular_moda(solucoes, casas_decimais=3):
    arredondados = [tuple(np.round(sol, casas_decimais)) for sol in solucoes]
    contagem = Counter(arredondados)
    moda, freq = contagem.most_common(1)[0]
    return np.array(moda), freq

moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# 5. Tabela de resultados
tabela_moda_f1 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# 6. Impressão de resultados
print("🔁 Total de soluções por algoritmo:")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F1):")
print(tabela_moda_f1.to_string(index=False))

# 7. Função gráfica final
def plot_multiplos_caminhos(f, dominio, lista_caminhos, titulo, tipo="linha", melhor_ponto=None):
    X = np.linspace(dominio[0][0], dominio[0][1], 300)
    Y = np.linspace(dominio[1][0], dominio[1][1], 300)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.3, edgecolor="none")

    for caminho in lista_caminhos:
        caminho = np.array(caminho)
        Z_caminho = np.array([f(*p) for p in caminho])
        if tipo == "linha":
            ax.plot(caminho[:, 0], caminho[:, 1], Z_caminho, alpha=0.3, linewidth=2)
            ax.scatter(caminho[-1, 0], caminho[-1, 1], Z_caminho[-1], color='red', marker='x', s=20)
        elif tipo == "pontos":
            ax.scatter(caminho[:, 0], caminho[:, 1], Z_caminho, alpha=0.3, s=20, c='red', marker='x')

    # Adiciona o X verde do melhor ponto
    if melhor_ponto is not None:
        ax.scatter(melhor_ponto[0], melhor_ponto[1], f(*melhor_ponto), c='green', marker='X', s=300, label='Melhor Solução')
        ax.legend()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title(titulo)
    plt.show()

# Replotando para F1 com o ponto verde incluso (sem mudar nenhum outro comportamento)
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (100 execuções)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (100 execuções)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (X)", tipo="pontos", melhor_ponto=melhor_grs)


# Selecionando a função F2
f_obj, dominio, tipo = funcoes_info[1]

# Hiperparâmetros fixos
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com registro de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
def melhor_solucao(f, tipo, lista_solucoes):
    return min(lista_solucoes, key=lambda x: f(*x)) if tipo == 'min' else max(lista_solucoes, key=lambda x: f(*x))

melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
def calcular_moda(solucoes, casas_decimais=3):
    arredondados = [tuple(np.round(sol, casas_decimais)) for sol in solucoes]
    contagem = Counter(arredondados)
    moda, freq = contagem.most_common(1)[0]
    return np.array(moda), freq

moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela
tabela_moda_f2 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo:")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F2):")
print(tabela_moda_f2.to_string(index=False))

# Plotando os rastros
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F2)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F2)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F2)", tipo="pontos", melhor_ponto=melhor_grs)

# Selecionando a função F3
f_obj, dominio, tipo = funcoes_info[2]

# Hiperparâmetros fixos
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com registro de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela
tabela_moda_f3 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo:")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F3):")
print(tabela_moda_f3.to_string(index=False))

# Plotando os rastros
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F3)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F3)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F3)", tipo="pontos", melhor_ponto=melhor_grs)

# Selecionando a função F4
f_obj, dominio, tipo = funcoes_info[3]

# Hiperparâmetros
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela de resultados
tabela_moda_f4 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo (F4):")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F4):")
print(tabela_moda_f4.to_string(index=False))

# Gráficos com destaque no melhor ponto
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F4)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F4)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F4)", tipo="pontos", melhor_ponto=melhor_grs)

# Selecionando a função F5
f_obj, dominio, tipo = funcoes_info[4]

# Hiperparâmetros
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela de resultados
tabela_moda_f5 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo (F5):")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F5):")
print(tabela_moda_f5.to_string(index=False))

# Gráficos com destaque no melhor ponto
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F5)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F5)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F5)", tipo="pontos", melhor_ponto=melhor_grs)

# CÉLULA DE EXECUÇÃO DA FUNÇÃO F6

# Selecionando a função F6
f_obj, dominio, tipo = funcoes_info[5]

# Hiperparâmetros
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.1
max_vizinhos = 100

# Execução dos algoritmos com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela de resultados
tabela_moda_f6 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo (F6):")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F6):")
print(tabela_moda_f6.to_string(index=False))

# Gráficos com destaque no melhor ponto
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F6)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F6)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F6)", tipo="pontos", melhor_ponto=melhor_grs)

# CÉLULA DE EXECUÇÃO DA FUNÇÃO F7

# Selecionando a função F7
f_obj, dominio, tipo = funcoes_info[6]

# Hiperparâmetros
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela de resultados
tabela_moda_f7 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo (F7):")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F7):")
print(tabela_moda_f7.to_string(index=False))

# Gráficos com destaque no melhor ponto
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F7)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F7)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F7)", tipo="pontos", melhor_ponto=melhor_grs)

# CÉLULA DE EXECUÇÃO DA FUNÇÃO F8

# Selecionando a função F8
f_obj, dominio, tipo = funcoes_info[7]

# Hiperparâmetros
num_rodadas = 100
max_it = 1000
epsilon = 0.1
sigma = 0.5
max_vizinhos = 100

# Execução dos algoritmos com coleta de caminhos
caminhos_hc, caminhos_lrs, caminhos_grs = [], [], []
sol_hc, sol_lrs, sol_grs = [], [], []

for _ in range(num_rodadas):
    hc = HillClimbing(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                      epsilon=epsilon, max_it=max_it, max_vizinhos=max_vizinhos)
    caminho = hc.run(retornar_caminho=True)
    caminhos_hc.append(caminho)
    sol_hc.append(caminho[-1])

for _ in range(num_rodadas):
    lrs = LocalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                            sigma=sigma, max_it=max_it)
    caminho = lrs.run(retornar_caminho=True)
    caminhos_lrs.append(caminho)
    sol_lrs.append(caminho[-1])

for _ in range(num_rodadas):
    grs = GlobalRandomSearch(f=f_obj, tipo_otimizacao=tipo, dominio=dominio,
                             max_it=max_it)
    caminho = grs.run(retornar_caminho=True)
    caminhos_grs.append(caminho)
    sol_grs.append(caminho[-1])

# Melhor solução
melhor_hc = melhor_solucao(f_obj, tipo, sol_hc)
melhor_lrs = melhor_solucao(f_obj, tipo, sol_lrs)
melhor_grs = melhor_solucao(f_obj, tipo, sol_grs)

# Moda
moda_hc, freq_hc = calcular_moda(sol_hc)
moda_lrs, freq_lrs = calcular_moda(sol_lrs)
moda_grs, freq_grs = calcular_moda(sol_grs)

# Tabela de resultados
tabela_moda_f8 = pd.DataFrame({
    "Algoritmo": ["Hill Climbing", "LRS", "GRS"],
    "Moda (x1, x2)": [moda_hc, moda_lrs, moda_grs],
    "f(moda)": [f_obj(*moda_hc), f_obj(*moda_lrs), f_obj(*moda_grs)],
    "Frequência (3 casas)": [f"{freq_hc}/100", f"{freq_lrs}/100", f"{freq_grs}/100"]
})

# Impressão dos resultados
print("🔁 Total de soluções por algoritmo (F8):")
print("HC :", len(sol_hc))
print("LRS:", len(sol_lrs))
print("GRS:", len(sol_grs))

print("\n⭐ Melhor solução Hill Climbing:", melhor_hc, "f =", f_obj(*melhor_hc))
print("⭐ Melhor solução LRS:", melhor_lrs, "f =", f_obj(*melhor_lrs))
print("⭐ Melhor solução GRS:", melhor_grs, "f =", f_obj(*melhor_grs))

print("\n📊 Moda das Soluções (F8):")
print(tabela_moda_f8.to_string(index=False))

# Gráficos com destaque no melhor ponto
plot_multiplos_caminhos(f_obj, dominio, caminhos_hc, "Hill Climbing - Caminhos (F8)", tipo="linha", melhor_ponto=melhor_hc)
plot_multiplos_caminhos(f_obj, dominio, caminhos_lrs, "Local Random Search - Caminhos (F8)", tipo="linha", melhor_ponto=melhor_lrs)
plot_multiplos_caminhos(f_obj, dominio, caminhos_grs, "Global Random Search - Pontos Visitados (F8)", tipo="pontos", melhor_ponto=melhor_grs)
