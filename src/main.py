# main.py (com seção de otimização de hiperparâmetros)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
import problemas_continuos as pc

# --- Definição de Todos os Problemas Contínuos (LISTA_DE_PROBLEMAS) ---
# (Mantenha a LISTA_DE_PROBLEMAS como definida anteriormente)
LISTA_DE_PROBLEMAS = [
    {
        'id': 1, 'nome': "Função Objetivo 1 (x1^2 + x2^2)", 'func_obj': pc.funcao_objetivo_1,
        'dominio_x1': (-100, 100), 'dominio_x2': (-100, 100), 'eh_maximizacao': False,
        'otimo_conhecido': 0.0 # Adicionar se o ótimo for conhecido
    },
    {
        'id': 2, 'nome': "Função Objetivo 2 (Exponenciais)", 'func_obj': pc.funcao_objetivo_2,
        'dominio_x1': (-2, 4), 'dominio_x2': (-2, 5), 'eh_maximizacao': True
        # Para a função 2, o ótimo é próximo de 2.0 (em x1=1.7, x2=1.7 e/ou x1=0, x2=0)
    },
    # Adicione os outros problemas aqui, incluindo 'otimo_conhecido' se aplicável
    # ... (restante da LISTA_DE_PROBLEMAS)
     {
        'id': 3, 'nome': "Função Objetivo 3 (Ackley)", 'func_obj': pc.funcao_objetivo_3,
        'dominio_x1': (-8, 8), 'dominio_x2': (-8, 8), 'eh_maximizacao': False,
        'otimo_conhecido': 0.0
    },
    {
        'id': 4, 'nome': "Função Objetivo 4 (Rastrigin)", 'func_obj': pc.funcao_objetivo_4,
        'dominio_x1': (-5.12, 5.12), 'dominio_x2': (-5.12, 5.12), 'eh_maximizacao': False,
        'otimo_conhecido': 0.0
    },
    {
        'id': 5, 'nome': "Função Objetivo 5 (Complexa com Cosseno e Exponencial)", 'func_obj': pc.funcao_objetivo_5,
        'dominio_x1': (-10, 10), 'dominio_x2': (-10, 10), 'eh_maximizacao': True
    },
    {
        'id': 6, 'nome': "Função Objetivo 6 (Seno e Cosseno)", 'func_obj': pc.funcao_objetivo_6,
        'dominio_x1': (-1, 3), 'dominio_x2': (-1, 3), 'eh_maximizacao': True
    },
    {
        'id': 7, 'nome': "Função Objetivo 7 (Seno com Potência)", 'func_obj': pc.funcao_objetivo_7,
        'dominio_x1': (0, np.pi), 'dominio_x2': (0, np.pi), 'eh_maximizacao': False,
        'otimo_conhecido': -2.0 # Aproximadamente, para x1 e x2 perto de 2.328
    },
    {
        'id': 8, 'nome': "Função Objetivo 8 (Eggholder Modificada)", 'func_obj': pc.funcao_objetivo_8,
        'dominio_x1': (-200, 20), 'dominio_x2': (-200, 20), 'eh_maximizacao': False
        # Ótimo complexo, geralmente valores negativos grandes.
    }
]


# --- Parâmetros para Otimização de Hiperparâmetros ---
EPSILON_VALORES_TESTE = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2] # Valores de Epsilon para testar no HC
SIGMA_VALORES_TESTE = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]   # Valores de Sigma para testar no LRS
N_TUNING_ROUNDS = 30      # Número de rodadas para cada valor de hiperparâmetro no tuning
MAX_ITER_TUNING = 500     # Máx de iterações por rodada durante o tuning (pode ser menor que o principal)
T_PARADA_TUNING = 50      # Parada antecipada para tuning

# Dicionário para armazenar os melhores hiperparâmetros encontrados para cada problema/algoritmo
melhores_hiperparametros_otimizados = {}

print("\n\n======================================================================")
print("INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("======================================================================")

for problema_atual_tuning in LISTA_DE_PROBLEMAS:
    id_problema = problema_atual_tuning['id']
    nome_problema = problema_atual_tuning['nome']
    print(f"\n--- Otimizando para Problema ID: {id_problema} ({nome_problema}) ---")
    
    melhores_hiperparametros_otimizados[id_problema] = {}

    # --- Otimização de Epsilon para Hill Climbing ---
    print(f"  Otimizando Epsilon para Hill Climbing...")
    resultados_tuning_hc_eps = []
    for epsilon_teste in EPSILON_VALORES_TESTE:
        custos_rodadas_tuning_hc = []
        for _ in range(N_TUNING_ROUNDS):
            _, _, custo_rodada_hc = pc.hill_climbing(
                func_obj=problema_atual_tuning['func_obj'],
                dominio_x1_lim=problema_atual_tuning['dominio_x1'],
                dominio_x2_lim=problema_atual_tuning['dominio_x2'],
                epsilon=epsilon_teste,
                max_iter=MAX_ITER_TUNING,
                t_parada_antecipada=T_PARADA_TUNING,
                eh_maximizacao=problema_atual_tuning['eh_maximizacao']
            )
            custos_rodadas_tuning_hc.append(custo_rodada_hc)
        
        # Usar o melhor custo encontrado nas N_TUNING_ROUNDS como métrica de performance
        performance_epsilon = min(custos_rodadas_tuning_hc) if not problema_atual_tuning['eh_maximizacao'] else max(custos_rodadas_tuning_hc)
        resultados_tuning_hc_eps.append({'epsilon': epsilon_teste, 'performance': performance_epsilon})
        print(f"    Epsilon: {epsilon_teste:.4f}, Melhor Custo Médio/Melhor nas Rodadas: {performance_epsilon:.6f}")

    # Selecionar o melhor epsilon: menor epsilon que alcança a melhor performance
    if resultados_tuning_hc_eps:
        if problema_atual_tuning['eh_maximizacao']:
            melhor_performance_geral_hc = max(r['performance'] for r in resultados_tuning_hc_eps)
        else:
            melhor_performance_geral_hc = min(r['performance'] for r in resultados_tuning_hc_eps)
        
        # Filtra todos os epsilons que alcançaram essa melhor performance
        candidatos_melhor_epsilon = [r for r in resultados_tuning_hc_eps if abs(r['performance'] - melhor_performance_geral_hc) < 1e-5] # Tolerância para floats
        melhor_epsilon_final = min(c['epsilon'] for c in candidatos_melhor_epsilon) # Pega o menor epsilon
        melhores_hiperparametros_otimizados[id_problema]['epsilon_hc'] = melhor_epsilon_final
        print(f"    Melhor Epsilon para HC no Problema {id_problema}: {melhor_epsilon_final:.4f} (Performance: {melhor_performance_geral_hc:.6f})")
    else:
        melhores_hiperparametros_otimizados[id_problema]['epsilon_hc'] = 0.01 # Valor default


    # --- Otimização de Sigma para LRS ---
    print(f"\n  Otimizando Sigma para Local Random Search (LRS)...")
    resultados_tuning_lrs_sig = []
    for sigma_teste in SIGMA_VALORES_TESTE:
        custos_rodadas_tuning_lrs = []
        for _ in range(N_TUNING_ROUNDS):
            _, _, custo_rodada_lrs = pc.lrs(
                func_obj=problema_atual_tuning['func_obj'],
                dominio_x1_lim=problema_atual_tuning['dominio_x1'],
                dominio_x2_lim=problema_atual_tuning['dominio_x2'],
                sigma=sigma_teste,
                max_iter=MAX_ITER_TUNING,
                t_parada_antecipada=T_PARADA_TUNING,
                eh_maximizacao=problema_atual_tuning['eh_maximizacao']
            )
            custos_rodadas_tuning_lrs.append(custo_rodada_lrs)

        performance_sigma = min(custos_rodadas_tuning_lrs) if not problema_atual_tuning['eh_maximizacao'] else max(custos_rodadas_tuning_lrs)
        resultados_tuning_lrs_sig.append({'sigma': sigma_teste, 'performance': performance_sigma})
        print(f"    Sigma: {sigma_teste:.4f}, Melhor Custo Médio/Melhor nas Rodadas: {performance_sigma:.6f}")

    if resultados_tuning_lrs_sig:
        if problema_atual_tuning['eh_maximizacao']:
            melhor_performance_geral_lrs = max(r['performance'] for r in resultados_tuning_lrs_sig)
        else:
            melhor_performance_geral_lrs = min(r['performance'] for r in resultados_tuning_lrs_sig)
            
        candidatos_melhor_sigma = [r for r in resultados_tuning_lrs_sig if abs(r['performance'] - melhor_performance_geral_lrs) < 1e-5]
        melhor_sigma_final = min(c['sigma'] for c in candidatos_melhor_sigma)
        melhores_hiperparametros_otimizados[id_problema]['sigma_lrs'] = melhor_sigma_final
        print(f"    Melhor Sigma para LRS no Problema {id_problema}: {melhor_sigma_final:.4f} (Performance: {melhor_performance_geral_lrs:.6f})")
    else:
        melhores_hiperparametros_otimizados[id_problema]['sigma_lrs'] = 0.1 # Valor default

print("\n======================================================================")
print("OTIMIZAÇÃO DE HIPERPARÂMETROS CONCLUÍDA")
print("Melhores hiperparâmetros encontrados:")
for prob_id, params in melhores_hiperparametros_otimizados.items():
    print(f"  Problema ID {prob_id}: Epsilon HC = {params.get('epsilon_hc', 'N/A')}, Sigma LRS = {params.get('sigma_lrs', 'N/A')}")
print("======================================================================")


# --- Parâmetros Globais do Experimento PRINCIPAL ---
MAX_RODADAS = 100
MAX_ITER_ALGORITMO = 1000
T_PARADA_ANTECIPADA = 100

# --- Lista de Algoritmos para Testar (agora usando hiperparâmetros otimizados ou defaults) ---
# Acessaremos os hiperparâmetros otimizados dentro do loop de problemas.

# --- Execução dos Experimentos PRINCIPAIS ---
# (O loop externo sobre LISTA_DE_PROBLEMAS e o interno sobre algoritmos_para_testar permanecem como antes)
# A diferença chave será como os hiperparâmetros são passados para HC e LRS.

for problema_atual in LISTA_DE_PROBLEMAS: # Este é o loop principal do experimento
    id_problema_atual = problema_atual['id']
    print(f"\n\n======================================================================")
    print(f"Iniciando EXPERIMENTO PRINCIPAL para: {problema_atual['nome']} (ID: {id_problema_atual})")
    print(f"Usando hiperparâmetros otimizados (ou defaults):")
    epsilon_usar = melhores_hiperparametros_otimizados.get(id_problema_atual, {}).get('epsilon_hc', 0.01) # Default se não otimizado
    sigma_usar = melhores_hiperparametros_otimizados.get(id_problema_atual, {}).get('sigma_lrs', 0.1) # Default se não otimizado
    print(f"  Epsilon para HC: {epsilon_usar}, Sigma para LRS: {sigma_usar}")
    print(f"======================================================================")

    resultados_finais_para_plot_problema_atual = []
    
    # Atualiza a lista de algoritmos com os hiperparâmetros otimizados para o problema atual
    algoritmos_configurados = [
        {
            'nome': "Global Random Search (GRS)", 'funcao': pc.grs, 'args': {}
        },
        {
            'nome': "Hill Climbing", 'funcao': pc.hill_climbing, 
            'args': {'epsilon': epsilon_usar} # Usa o epsilon otimizado
        },
        {
            'nome': "Local Random Search (LRS)", 'funcao': pc.lrs, 
            'args': {'sigma': sigma_usar} # Usa o sigma otimizado
        }
    ]

    for algo_info in algoritmos_configurados: # Usa a lista configurada
        print(f"\n  Executando algoritmo: {algo_info['nome']} no Problema ID: {id_problema_atual}")
        # ... (resto do loop interno de algoritmos, coleta de resultados e plotagem como antes) ...
        # Certifique-se de que kwargs_algo é construído corretamente com os hiperparâmetros de algo_info['args']
        resultados_rodadas_algo = []
        tempo_inicio_algo = time.time()

        for rodada_idx in range(MAX_RODADAS):
            kwargs_algo = {
                'func_obj': problema_atual['func_obj'],
                'dominio_x1_lim': problema_atual['dominio_x1'],
                'dominio_x2_lim': problema_atual['dominio_x2'],
                'max_iter': MAX_ITER_ALGORITMO,
                't_parada_antecipada': T_PARADA_ANTECIPADA,
                'eh_maximizacao': problema_atual['eh_maximizacao'],
                **algo_info['args']
            }
            
            melhor_x1_rodada, melhor_x2_rodada, melhor_custo_rodada = algo_info['funcao'](**kwargs_algo)
            
            resultados_rodadas_algo.append({
                'rodada': rodada_idx + 1, 'x1': melhor_x1_rodada,
                'x2': melhor_x2_rodada, 'custo': melhor_custo_rodada
            })
        
        tempo_fim_algo = time.time()
        print(f"    Tempo de execução para {algo_info['nome']}: {tempo_fim_algo - tempo_inicio_algo:.2f} segundos")

        if not resultados_rodadas_algo:
            print(f"    Nenhum resultado gerado para {algo_info['nome']} neste problema.")
            continue

        if problema_atual['eh_maximizacao']:
            melhor_resultado_algo_problema = max(resultados_rodadas_algo, key=lambda r: r['custo'])
        else:
            melhor_resultado_algo_problema = min(resultados_rodadas_algo, key=lambda r: r['custo'])

        print(f"    Melhor solução encontrada pelo {algo_info['nome']}:")
        print(f"      x1 = {melhor_resultado_algo_problema['x1']:.6f}, x2 = {melhor_resultado_algo_problema['x2']:.6f}, Custo = {melhor_resultado_algo_problema['custo']:.6f}")
        
        resultados_finais_para_plot_problema_atual.append({
            **melhor_resultado_algo_problema, 'algoritmo': algo_info['nome']
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

    # Plotagem para o problema atual
    if not resultados_finais_para_plot_problema_atual:
        print(f"\nNenhuma solução final para plotar para o Problema ID: {problema_atual['id']}.")
        continue
    print(f"\n  Gerando visualização gráfica comparativa para o Problema ID: {problema_atual['id']}...")
    # ... (código de plotagem como antes, usando problema_atual e resultados_finais_para_plot_problema_atual) ...
    x1_vals_plot = np.linspace(problema_atual['dominio_x1'][0], problema_atual['dominio_x1'][1], 100)
    x2_vals_plot = np.linspace(problema_atual['dominio_x2'][0], problema_atual['dominio_x2'][1], 100)
    X1_grid_plot, X2_grid_plot = np.meshgrid(x1_vals_plot, x2_vals_plot)
    
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
    else:
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
    plt.show()


print("\n\nTodos os experimentos e visualizações (se houver) foram concluídos.")