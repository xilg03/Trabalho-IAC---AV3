import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # Para ajudar a encontrar a moda

# --- Definição da Função Objetivo e Domínio ---
# Problema 1: f(x1,x2) = x1^2 + x2^2
# Domínio: x1, x2 em [-100, 100]
# Objetivo: Minimização
def funcao_objetivo_1(x1, x2):
    return x1**2 + x2**2

dominio_problema_1 = {
    'x1': (-100, 100),
    'x2': (-100, 100)
}
EH_MAXIMIZACAO_PROBLEMA_1 = False

# --- Verificador de Limites ---
def verificar_limites(candidato_x1, candidato_x2, dominio_x1_lim, dominio_x2_lim):
    if not (dominio_x1_lim[0] <= candidato_x1 <= dominio_x1_lim[1]):
        return False
    if not (dominio_x2_lim[0] <= candidato_x2 <= dominio_x2_lim[1]):
        return False
    return True

# --- Algoritmo: Busca Aleatória Global (GRS) ---
def busca_aleatoria_global(func_obj, dominio_x1_lim, dominio_x2_lim, max_iter, t_parada_antecipada, eh_maximizacao):
    """
    Implementação da Busca Aleatória Global (GRS).
    Gera novo candidato aleatoriamente no espaço de busca.
    """
    # Gera um candidato inicial aleatório para começar
    x1_melhor = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
    x2_melhor = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
    custo_melhor = func_obj(x1_melhor, x2_melhor)
    iter_sem_melhora = 0

    for i in range(max_iter): # Loop de iterações do algoritmo [cite: 18]
        cand_x1 = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
        cand_x2 = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
        
        # Não é estritamente necessário verificar limites aqui para GRS se a geração já os respeita,
        # mas é uma boa prática se a geração pudesse, por algum motivo, sair dos limites.
        # if not verificar_limites(cand_x1, cand_x2, dominio_x1_lim, dominio_x2_lim):
        # continue # Pula este candidato se estiver fora dos limites (improvável com random.uniform)

        custo_candidato = func_obj(cand_x1, cand_x2)

        if eh_maximizacao:
            if custo_candidato > custo_melhor:
                x1_melhor, x2_melhor, custo_melhor = cand_x1, cand_x2, custo_candidato
                iter_sem_melhora = 0
            else:
                iter_sem_melhora += 1
        else: # Minimização
            if custo_candidato < custo_melhor:
                x1_melhor, x2_melhor, custo_melhor = cand_x1, cand_x2, custo_candidato
                iter_sem_melhora = 0
            else:
                iter_sem_melhora += 1
        
        if iter_sem_melhora >= t_parada_antecipada: # Critério de parada antecipada [cite: 26]
            # print(f"GRS: Parada antecipada na iteração {i + 1} de {max_iter}")
            break
            
    return x1_melhor, x2_melhor, custo_melhor

# --- Configuração e Execução do Experimento ---
NOME_PROBLEMA = "Função x1^2 + x2^2"
NOME_ALGORITMO = "Busca Aleatória Global (GRS)"

MAX_RODADAS = 100  # Número de rodadas do experimento [cite: 17]
MAX_ITER_ALGORITMO = 1000 # Máximo de iterações por rodada do algoritmo [cite: 26]
T_PARADA_ANTECIPADA = 100 # Iterações sem melhoria para parada antecipada [cite: 26]

print(f"Iniciando experimento para: {NOME_PROBLEMA}")
print(f"Algoritmo: {NOME_ALGORITMO}")
print(f"Número de rodadas: {MAX_RODADAS}")
print(f"Máximo de iterações por rodada: {MAX_ITER_ALGORITMO}")
print(f"Parada antecipada após {T_PARADA_ANTECIPADA} iterações sem melhora.")

resultados_rodadas = []
for rodada_idx in range(MAX_RODADAS):
    # print(f"  Executando rodada {rodada_idx + 1}/{MAX_RODADAS}...")
    melhor_x1_rodada, melhor_x2_rodada, melhor_custo_rodada = busca_aleatoria_global(
        funcao_objetivo_1,
        dominio_problema_1['x1'],
        dominio_problema_1['x2'],
        MAX_ITER_ALGORITMO,
        T_PARADA_ANTECIPADA,
        EH_MAXIMIZACAO_PROBLEMA_1
    )
    resultados_rodadas.append({
        'rodada': rodada_idx + 1,
        'x1': melhor_x1_rodada,
        'x2': melhor_x2_rodada,
        'custo': melhor_custo_rodada
    })

print("\n--- Resultados do Experimento ---")

# Encontrar o melhor resultado geral entre todas as rodadas
if EH_MAXIMIZACAO_PROBLEMA_1:
    melhor_resultado_geral = max(resultados_rodadas, key=lambda r: r['custo'])
else:
    melhor_resultado_geral = min(resultados_rodadas, key=lambda r: r['custo'])

print(f"Melhor solução encontrada em {MAX_RODADAS} rodadas:")
print(f"  Rodada: {melhor_resultado_geral['rodada']}")
print(f"  x1 = {melhor_resultado_geral['x1']:.6f}")
print(f"  x2 = {melhor_resultado_geral['x2']:.6f}")
print(f"  Custo (f(x1,x2)) = {melhor_resultado_geral['custo']:.6f}")

# Calcular a moda dos custos (arredondados para agrupar) [cite: 19, 28]
# O arredondamento ajuda a agrupar valores muito próximos em problemas contínuos.
# A precisão do arredondamento (número de casas decimais) pode depender da sensibilidade da função.
CASAS_DECIMAIS_MODA = 4 
custos_para_moda = [round(r['custo'], CASAS_DECIMAIS_MODA) for r in resultados_rodadas]
contagem_custos = Counter(custos_para_moda)
moda_custo_info = contagem_custos.most_common(1)[0] # (valor_modal, frequencia)

print(f"\nModa dos custos (arredondados para {CASAS_DECIMAIS_MODA} casas decimais):")
print(f"  Custo modal: {moda_custo_info[0]}")
print(f"  Frequência: {moda_custo_info[1]} de {MAX_RODADAS} rodadas")

# --- Visualização com Matplotlib ---
print("\nGerando visualização gráfica...")

# 1. Preparar dados para a superfície da função
x1_vals = np.linspace(dominio_problema_1['x1'][0], dominio_problema_1['x1'][1], 100)
x2_vals = np.linspace(dominio_problema_1['x2'][0], dominio_problema_1['x2'][1], 100)
X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
Y_grid = funcao_objetivo_1(X1_grid, X2_grid)

# 2. Ponto da melhor solução encontrada para destacar no gráfico
x1_solucao = melhor_resultado_geral['x1']
x2_solucao = melhor_resultado_geral['x2']
custo_solucao = melhor_resultado_geral['custo']

# 3. Plotagem
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superfície da função
surf = ax.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', edgecolor='none', alpha=0.7)

# Ponto da solução
ax.scatter(x1_solucao, x2_solucao, custo_solucao, color='red', marker='o', s=100, depthshade=True, 
           label=f'Melhor Solução GRS ({x1_solucao:.2f}, {x2_solucao:.2f}) = {custo_solucao:.2f}')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title(f'Superfície de {NOME_PROBLEMA} e Melhor Solução Encontrada pelo {NOME_ALGORITMO}')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5, label='Custo f(x1,x2)')

plt.show()

print("Visualização concluída.")