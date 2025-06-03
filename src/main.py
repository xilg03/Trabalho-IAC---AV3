import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time # Para medir o tempo de execução, se desejar

# Importa as funções e algoritmos do arquivo problemas_continuos.py
import problemas_continuos as pc

# --- Configuração do Problema Específico ---
# Vamos usar o Problema 1 como exemplo
# f(x1,x2) = x1^2 + x2^2, Domínio: x1, x2 em [-100, 100], Minimização
PROBLEMA_ESCOLHIDO = {
    'nome': "Função Objetivo 2 (Maximização)",
    'func_obj': pc.funcao_objetivo_2, # Referencia a função importada
    'dominio_x1': (-2, 4),
    'dominio_x2': (-2, 5),
    'eh_maximizacao': True
}

# --- Parâmetros Globais do Experimento ---
MAX_RODADAS = 100  # Número de rodadas do experimento [cite: 17]
MAX_ITER_ALGORITMO = 1000 # Máximo de iterações por rodada do algoritmo [cite: 26]
T_PARADA_ANTECIPADA = 100 # Iterações sem melhoria para parada antecipada [cite: 26]

# --- Hiperparâmetros Específicos dos Algoritmos ---
# Estes valores são exemplos, você precisará ajustá-los e encontrar os ótimos [cite: 24, 25]
EPSILON_HC = 0.01 # Ajuste este valor para o Hill Climbing
SIGMA_LRS = 0.1   # Ajuste este valor para o LRS ($0 < \sigma < 1$)

# --- Lista de Algoritmos para Testar ---
# Cada item é um dicionário com o nome do algoritmo, a função a ser chamada, e seus hiperparâmetros
algoritmos_para_testar = [
    {
        'nome': "Busca Aleatória Global (GRS)",
        'funcao': pc.grs, # Referencia a função importada
        'args': {} # GRS não tem hiperparâmetros especiais além dos globais
    },
    {
        'nome': "Hill Climbing (Subida de Encosta)",
        'funcao': pc.hill_climbing, # Referencia a função importada
        'args': {'epsilon': EPSILON_HC}
    },
    {
        'nome': "Local Random Search (LRS)",
        'funcao': pc.lrs, # Referencia a função importada
        'args': {'sigma': SIGMA_LRS}
    }
]

# --- Execução dos Experimentos ---
todos_os_resultados_finais = [] # Para armazenar o melhor de cada algoritmo para plotagem comparativa

print(f"Iniciando experimentos para o problema: {PROBLEMA_ESCOLHIDO['nome']}")
print(f"Número de rodadas por algoritmo: {MAX_RODADAS}")
print(f"Máximo de iterações por rodada: {MAX_ITER_ALGORITMO}")
print(f"Parada antecipada após {T_PARADA_ANTECIPADA} iterações sem melhora.")
print("-" * 40)

for algo_info in algoritmos_para_testar:
    print(f"\nExecutando algoritmo: {algo_info['nome']}")
    
    resultados_rodadas_algo = []
    tempo_inicio_algo = time.time()

    for rodada_idx in range(MAX_RODADAS):
        # Monta os argumentos específicos para a função do algoritmo
        kwargs_algo = {
            'func_obj': PROBLEMA_ESCOLHIDO['func_obj'],
            'dominio_x1_lim': PROBLEMA_ESCOLHIDO['dominio_x1'],
            'dominio_x2_lim': PROBLEMA_ESCOLHIDO['dominio_x2'],
            'max_iter': MAX_ITER_ALGORITMO,
            't_parada_antecipada': T_PARADA_ANTECIPADA,
            'eh_maximizacao': PROBLEMA_ESCOLHIDO['eh_maximizacao'],
            **algo_info['args'] # Adiciona hiperparâmetros específicos (epsilon, sigma)
        }
        
        melhor_x1_rodada, melhor_x2_rodada, melhor_custo_rodada = algo_info['funcao'](**kwargs_algo)
        
        resultados_rodadas_algo.append({
            'rodada': rodada_idx + 1,
            'x1': melhor_x1_rodada,
            'x2': melhor_x2_rodada,
            'custo': melhor_custo_rodada
        })
    
    tempo_fim_algo = time.time()
    print(f"Tempo de execução para {algo_info['nome']}: {tempo_fim_algo - tempo_inicio_algo:.2f} segundos")

    # Análise dos resultados para este algoritmo
    if PROBLEMA_ESCOLHIDO['eh_maximizacao']:
        melhor_resultado_algo = max(resultados_rodadas_algo, key=lambda r: r['custo'])
    else:
        melhor_resultado_algo = min(resultados_rodadas_algo, key=lambda r: r['custo'])

    print(f"  Melhor solução encontrada pelo {algo_info['nome']}:")
    print(f"    x1 = {melhor_resultado_algo['x1']:.6f}, x2 = {melhor_resultado_algo['x2']:.6f}, Custo = {melhor_resultado_algo['custo']:.6f}")
    todos_os_resultados_finais.append({**melhor_resultado_algo, 'algoritmo': algo_info['nome']})

    CASAS_DECIMAIS_MODA = 4 
    custos_para_moda_algo = [round(r['custo'], CASAS_DECIMAIS_MODA) for r in resultados_rodadas_algo]
    contagem_custos_algo = Counter(custos_para_moda_algo)
    if contagem_custos_algo:
        moda_custo_info_algo = contagem_custos_algo.most_common(1)[0]
        print(f"  Moda dos custos (arredondados): {moda_custo_info_algo[0]} (Frequência: {moda_custo_info_algo[1]}/{MAX_RODADAS})")
    else:
        print("  Não foi possível calcular a moda dos custos (sem resultados).")
    print("-" * 40)

# --- Visualização Comparativa com Matplotlib (para o problema escolhido) ---
print("\nGerando visualização gráfica comparativa...")

# 1. Preparar dados para a superfície da função
x1_vals = np.linspace(PROBLEMA_ESCOLHIDO['dominio_x1'][0], PROBLEMA_ESCOLHIDO['dominio_x1'][1], 100)
x2_vals = np.linspace(PROBLEMA_ESCOLHIDO['dominio_x2'][0], PROBLEMA_ESCOLHIDO['dominio_x2'][1], 100)
X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
Y_grid = PROBLEMA_ESCOLHIDO['func_obj'](X1_grid, X2_grid)

# 2. Plotagem
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', edgecolor='none', alpha=0.6, rstride=5, cstride=5)

# 3. Plotar as melhores soluções de cada algoritmo
markers = ['o', '^', 's', 'D', 'P', '*'] # Diferentes marcadores para cada algoritmo
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown'] # Diferentes cores

for i, res_final in enumerate(todos_os_resultados_finais):
    ax.scatter(res_final['x1'], res_final['x2'], res_final['custo'], 
               color=colors[i % len(colors)], 
               marker=markers[i % len(markers)], 
               s=150, # Tamanho do marcador
               depthshade=True, 
               label=f"Melhor {res_final['algoritmo']} ({res_final['custo']:.4f})")

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title(f"Comparação de Soluções para {PROBLEMA_ESCOLHIDO['nome']}")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1)) # Legenda fora do plot
fig.colorbar(surf, shrink=0.5, aspect=10, label='Custo f(x1,x2)', pad=0.1)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar layout para caber a legenda
plt.show()

print("Execução e visualização concluídas.")