# main.py

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time # Para medir o tempo de execução, se desejar

# Importa as funções e algoritmos do arquivo problemas_continuos.py
import problemas_continuos as pc # 'pc' é o alias para o módulo

# --- Definição de Todos os Problemas Contínuos ---
# Cada dicionário descreve um problema: nome, função objetivo, domínios e tipo (min/max)
LISTA_DE_PROBLEMAS = [
    {
        'id': 1, # Para referência
        'nome': "Função Objetivo 1 (x1^2 + x2^2)",
        'func_obj': pc.funcao_objetivo_1,
        'dominio_x1': (-100, 100),
        'dominio_x2': (-100, 100),
        'eh_maximizacao': False
    },
    {
        'id': 2,
        'nome': "Função Objetivo 2 (Exponenciais)",
        'func_obj': pc.funcao_objetivo_2,
        'dominio_x1': (-2, 4),
        'dominio_x2': (-2, 5),
        'eh_maximizacao': True
    },
    {
        'id': 3,
        'nome': "Função Objetivo 3 (Ackley)",
        'func_obj': pc.funcao_objetivo_3,
        'dominio_x1': (-8, 8),
        'dominio_x2': (-8, 8),
        'eh_maximizacao': False
    },
    {
        'id': 4,
        'nome': "Função Objetivo 4 (Rastrigin)",
        'func_obj': pc.funcao_objetivo_4,
        'dominio_x1': (-5.12, 5.12),
        'dominio_x2': (-5.12, 5.12),
        'eh_maximizacao': False
    },
    {
        'id': 5,
        'nome': "Função Objetivo 5 (Complexa com Cosseno e Exponencial)",
        'func_obj': pc.funcao_objetivo_5,
        'dominio_x1': (-10, 10),
        'dominio_x2': (-10, 10),
        'eh_maximizacao': True
    },
    {
        'id': 6,
        'nome': "Função Objetivo 6 (Seno e Cosseno)",
        'func_obj': pc.funcao_objetivo_6,
        'dominio_x1': (-1, 3),
        'dominio_x2': (-1, 3),
        'eh_maximizacao': True
    },
    {
        'id': 7,
        'nome': "Função Objetivo 7 (Seno com Potência)",
        'func_obj': pc.funcao_objetivo_7,
        'dominio_x1': (0, np.pi), # Usando np.pi para precisão
        'dominio_x2': (0, np.pi),
        'eh_maximizacao': False
    },
    {
        'id': 8,
        'nome': "Função Objetivo 8 (Eggholder Modificada)",
        'func_obj': pc.funcao_objetivo_8,
        'dominio_x1': (-200, 20),
        'dominio_x2': (-200, 20),
        'eh_maximizacao': False
    }
]

# --- Parâmetros Globais do Experimento ---
MAX_RODADAS = 100  # Número de rodadas do experimento [cite: 17]
MAX_ITER_ALGORITMO = 1000 # Máximo de iterações por rodada do algoritmo [cite: 26]
T_PARADA_ANTECIPADA = 100 # Iterações sem melhoria para parada antecipada [cite: 26]

# --- Hiperparâmetros Específicos dos Algoritmos ---
# Estes valores são exemplos, você precisará ajustá-los e encontrar os ótimos [cite: 24, 25]
# O ideal é que você otimize EPSILON_HC e SIGMA_LRS para cada problema,
# ou encontre valores robustos. Para este exemplo, usaremos valores fixos.
EPSILON_HC = 0.01
SIGMA_LRS = 0.1

# --- Lista de Algoritmos para Testar ---
algoritmos_para_testar = [
    {
        'nome': "Global Random Search (GRS)",
        'funcao': pc.grs,
        'args': {}
    },
    {
        'nome': "Hill Climbing",
        'funcao': pc.hill_climbing,
        'args': {'epsilon': EPSILON_HC}
    },
    {
        'nome': "Local Random Search (LRS)",
        'funcao': pc.lrs,
        'args': {'sigma': SIGMA_LRS}
    }
]

# --- Execução dos Experimentos ---
# Loop externo para iterar sobre cada problema definido em LISTA_DE_PROBLEMAS
for problema_atual in LISTA_DE_PROBLEMAS:
    print(f"\n\n======================================================================")
    print(f"Iniciando experimentos para o problema: {problema_atual['nome']} (ID: {problema_atual['id']})")
    print(f"======================================================================")
    print(f"Número de rodadas por algoritmo: {MAX_RODADAS}")
    print(f"Máximo de iterações por rodada: {MAX_ITER_ALGORITMO}")
    print(f"Parada antecipada após {T_PARADA_ANTECIPADA} iterações sem melhora.")
    print("-" * 70)

    # Para armazenar o melhor resultado de cada algoritmo PARA ESTE PROBLEMA ESPECÍFICO (para plotagem)
    resultados_finais_para_plot_problema_atual = []

    # Loop interno para iterar sobre cada algoritmo
    for algo_info in algoritmos_para_testar:
        print(f"\n  Executando algoritmo: {algo_info['nome']} no Problema ID: {problema_atual['id']}")
        
        resultados_rodadas_algo = []
        tempo_inicio_algo = time.time()

        for rodada_idx in range(MAX_RODADAS):
            # Monta os argumentos específicos para a função do algoritmo
            kwargs_algo = {
                'func_obj': problema_atual['func_obj'],
                'dominio_x1_lim': problema_atual['dominio_x1'],
                'dominio_x2_lim': problema_atual['dominio_x2'],
                'max_iter': MAX_ITER_ALGORITMO,
                't_parada_antecipada': T_PARADA_ANTECIPADA,
                'eh_maximizacao': problema_atual['eh_maximizacao'],
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
        print(f"    Tempo de execução para {algo_info['nome']}: {tempo_fim_algo - tempo_inicio_algo:.2f} segundos")

        # Análise dos resultados para este algoritmo e este problema
        if not resultados_rodadas_algo: # Verifica se houve resultados
            print(f"    Nenhum resultado gerado para {algo_info['nome']} neste problema.")
            continue

        if problema_atual['eh_maximizacao']:
            melhor_resultado_algo_problema = max(resultados_rodadas_algo, key=lambda r: r['custo'])
        else:
            melhor_resultado_algo_problema = min(resultados_rodadas_algo, key=lambda r: r['custo'])

        print(f"    Melhor solução encontrada pelo {algo_info['nome']}:")
        print(f"      x1 = {melhor_resultado_algo_problema['x1']:.6f}, x2 = {melhor_resultado_algo_problema['x2']:.6f}, Custo = {melhor_resultado_algo_problema['custo']:.6f}")
        
        resultados_finais_para_plot_problema_atual.append({
            **melhor_resultado_algo_problema, 
            'algoritmo': algo_info['nome']
        })

        CASAS_DECIMAIS_MODA = 4 
        custos_para_moda_algo = [round(r['custo'], CASAS_DECIMAIS_MODA) for r in resultados_rodadas_algo]
        contagem_custos_algo = Counter(custos_para_moda_algo)
        if contagem_custos_algo:
            moda_custo_info_algo = contagem_custos_algo.most_common(1)[0]
            print(f"    Moda dos custos (arredondados): {moda_custo_info_algo[0]} (Frequência: {moda_custo_info_algo[1]}/{MAX_RODADAS})")
        else:
            print("    Não foi possível calcular a moda dos custos (sem resultados).")
        print("-" * 70)

    # --- Visualização Comparativa com Matplotlib (para o problema ATUAL) ---
    if not resultados_finais_para_plot_problema_atual:
        print(f"\nNenhuma solução final para plotar para o Problema ID: {problema_atual['id']}.")
        continue

    print(f"\n  Gerando visualização gráfica comparativa para o Problema ID: {problema_atual['id']}...")

    x1_vals_plot = np.linspace(problema_atual['dominio_x1'][0], problema_atual['dominio_x1'][1], 100)
    x2_vals_plot = np.linspace(problema_atual['dominio_x2'][0], problema_atual['dominio_x2'][1], 100)
    X1_grid_plot, X2_grid_plot = np.meshgrid(x1_vals_plot, x2_vals_plot)
    
    # Tentar calcular Y_grid, mas tratar exceções se a função não for bem comportada em toda a grade
    try:
        Y_grid_plot = problema_atual['func_obj'](X1_grid_plot, X2_grid_plot)
    except Exception as e:
        print(f"    AVISO: Não foi possível gerar Y_grid para plotagem da superfície da função {problema_atual['nome']}. Erro: {e}")
        Y_grid_plot = None


    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if Y_grid_plot is not None:
        surf = ax.plot_surface(X1_grid_plot, X2_grid_plot, Y_grid_plot, cmap='viridis', edgecolor='none', alpha=0.6, rstride=5, cstride=5)
        fig.colorbar(surf, shrink=0.5, aspect=10, label=f"Custo f(x1,x2) - Problema {problema_atual['id']}", pad=0.1)
    else: # Se não puder plotar a superfície, plote apenas os pontos
        ax.text2D(0.5, 0.5, "Superfície não pôde ser plotada", transform=ax.transAxes, ha="center")


    markers = ['o', '^', 's', 'D', 'P', '*'] 
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown'] 

    for i, res_final_plot in enumerate(resultados_finais_para_plot_problema_atual):
        ax.scatter(res_final_plot['x1'], res_final_plot['x2'], res_final_plot['custo'], 
                   color=colors[i % len(colors)], 
                   marker=markers[i % len(markers)], 
                   s=150, 
                   depthshade=True, 
                   label=f"Melhor {res_final_plot['algoritmo']} ({res_final_plot['custo']:.4f})")

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(f"Comparação de Soluções para {problema_atual['nome']}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1)) 
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    # Opcional: Salvar a figura em vez de mostrar, ou mostrar uma de cada vez
    # plt.savefig(f"grafico_problema_{problema_atual['id']}.png")
    # plt.close(fig) # Fecha a figura para não acumular muitas abertas
    plt.show() # Mostra o gráfico para o problema atual

print("\n\nTodos os experimentos e visualizações (se houver) foram concluídos.")