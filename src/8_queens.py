import math
import random

# --- Problema das 8 Rainhas ---
# Representação: um vetor de 8 posições, onde o índice é a coluna (0-7)
# e o valor é a linha (0-7) onde a rainha está. [cite: 49, 50]

def calcular_h_custo(tabuleiro):
    """
    Calcula h(x), o número de pares de rainhas se atacando.
    O tabuleiro é um vetor onde tabuleiro[coluna] = linha.
    """
    n = len(tabuleiro)
    ataques = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Checar ataque na mesma linha
            if tabuleiro[i] == tabuleiro[j]:
                ataques += 1
            # Checar ataque na diagonal
            # |linha1 - linha2| == |coluna1 - coluna2|
            if abs(tabuleiro[i] - tabuleiro[j]) == abs(i - j):
                ataques += 1
    return ataques

def funcao_objetivo_8rainhas(tabuleiro):
    """
    Função objetivo f(x) = 28 - h(x)[cite: 55].
    Queremos maximizar f(x), o que significa minimizar h(x).
    Um valor de 28 significa 0 ataques.
    """
    h = calcular_h_custo(tabuleiro)
    return 28 - h

def gerar_vizinho_8rainhas(tabuleiro_atual):
    """
    Gera um vizinho trocando as posições de duas rainhas em colunas aleatórias.
    Esta é uma forma de perturbação que explora uma porção controlada do espaço de estados[cite: 60].
    """
    n = len(tabuleiro_atual)
    vizinho = list(tabuleiro_atual) # Cria uma cópia
    
    # Escolhe duas colunas distintas aleatoriamente
    col1, col2 = random.sample(range(n), 2)
    
    # Troca as linhas das rainhas nessas colunas
    vizinho[col1], vizinho[col2] = vizinho[col2], vizinho[col1]
    return vizinho

def tempera_simulada_8rainhas(temp_inicial, taxa_resfriamento, max_iter_sem_melhora_global, max_iter_total):
    """
    Implementação da Têmpera Simulada para o problema das 8 Rainhas.
    """
    n_rainhas = 8
    # Gera um estado inicial aleatório (uma rainha por coluna, em linha aleatória)
    estado_atual = random.sample(range(n_rainhas), n_rainhas)
    custo_atual = funcao_objetivo_8rainhas(estado_atual)

    melhor_estado = list(estado_atual)
    melhor_custo = custo_atual
    
    temp_corrente = temp_inicial
    iter_sem_melhora_global = 0
    
    # Armazenar soluções encontradas para a tarefa de encontrar as 92
    solucoes_encontradas = set() # Usar tuplas para poder adicionar a um set

    for i in range(max_iter_total):
        vizinho = gerar_vizinho_8rainhas(estado_atual)
        custo_vizinho = funcao_objetivo_8rainhas(vizinho)

        delta_energia = custo_vizinho - custo_atual # Maximização

        if delta_energia > 0: # Vizinho é melhor, aceita
            estado_atual = vizinho
            custo_atual = custo_vizinho
        else:
            # Aceita com probabilidade e^(delta_E / T)
            if random.random() < math.exp(delta_energia / temp_corrente):
                estado_atual = vizinho
                custo_atual = custo_vizinho
        
        if custo_atual > melhor_custo:
            melhor_estado = list(estado_atual)
            melhor_custo = custo_atual
            iter_sem_melhora_global = 0
        else:
            iter_sem_melhora_global +=1

        # Adiciona à lista de soluções se for uma solução ótima (custo 28)
        if melhor_custo == 28:
            solucao_tupla = tuple(melhor_estado) # Converte para tupla para ser hashable
            if solucao_tupla not in solucoes_encontradas:
                #print(f"Solução encontrada: {melhor_estado}, Custo: {melhor_custo}, Temp: {temp_corrente:.2f}, Iter: {i+1}")
                solucoes_encontradas.add(solucao_tupla)
            # Critério de parada: encontrou uma solução e o objetivo é apenas uma [cite: 61]
            # Se o objetivo for encontrar as 92, continue procurando ou reinicie.
            # Para a tarefa de encontrar todas as 92, você não pararia aqui necessariamente.
            # A parada pode ser "atingir o máximo de iterações OU quando a função objetivo atingir seu valor ótimo" [cite: 61]
            # Se o objetivo é encontrar *uma* solução, pode parar:
            # return melhor_estado, melhor_custo, solucoes_encontradas

        if iter_sem_melhora_global >= max_iter_sem_melhora_global and melhor_custo < 28 : # Evita parar se já achou uma solução e quer mais
             #print(f"Parada por iterações sem melhora global com T={temp_corrente:.2f} na iter {i+1}")
             #break # Poderia reiniciar (re-aquecer) ou parar
             pass


        temp_corrente *= taxa_resfriamento # Decaimento da temperatura [cite: 59]
        if temp_corrente < 0.001 and melhor_custo < 28 : # Limite inferior para temperatura
            #print(f"Temperatura muito baixa e solução não encontrada. Parando. Iter: {i+1}")
            break # Ou re-aquecer

        # Critério de parada se o objetivo é encontrar todas as 92 soluções [cite: 62]
        if len(solucoes_encontradas) >= 92:
             print(f"Todas as 92 soluções encontradas! Iter: {i+1}")
             break
             
    #print(f"Finalizado. Melhor estado: {melhor_estado}, Melhor custo: {melhor_custo}")
    #print(f"Total de soluções distintas encontradas: {len(solucoes_encontradas)}")
    return melhor_estado, melhor_custo, solucoes_encontradas


# Exemplo de execução para 8 Rainhas
# temp_inicial_8q = 1000
# taxa_resfriamento_8q = 0.995 # Escolha uma maneira de decaimento [cite: 59]
# max_iter_sem_melhora_8q = 500
# max_iter_total_8q = 200000 # Ajustar para encontrar mais soluções

# print("\nExecutando Têmpera Simulada para 8 Rainhas...")
# _, _, solucoes_8q = tempera_simulada_8rainhas(
# temp_inicial_8q,
# taxa_resfriamento_8q,
# max_iter_sem_melhora_8q,
# max_iter_total_8q
# )
# print(f"Número de soluções distintas encontradas: {len(solucoes_8q)}")
# if len(solucoes_8q) > 0:
# print("Uma das soluções:")
# for sol_tuple in list(solucoes_8q)[:1]: # Mostra a primeira encontrada
#       board_vis = [['.' for _ in range(8)] for _ in range(8)]
#       for col, row in enumerate(sol_tuple):
#           board_vis[row][col] = 'Q'
#       for r in board_vis:
#           print(' '.join(r))