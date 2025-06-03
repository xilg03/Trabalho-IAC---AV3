import math
import random

# --- Funções Objetivo ---
# Exemplo para a primeira função: f(x1,x2) = x1^2 + x2^2
def funcao_objetivo_1(x1, x2):
    return x1**2 + x2**2

# --- Verificador de Limites ---
def verificar_limites(candidato_x1, candidato_x2, dominio_x1_lim, dominio_x2_lim):
    """Verifica se um candidato está dentro dos limites do domínio."""
    if not (dominio_x1_lim[0] <= candidato_x1 <= dominio_x1_lim[1]):
        return False
    if not (dominio_x2_lim[0] <= candidato_x2 <= dominio_x2_lim[1]):
        return False
    return True

# --- Algoritmo: Hill Climbing (Subida de Encosta) ---
def subida_de_encosta(func_obj, dominio_x1_lim, dominio_x2_lim, epsilon, 
                        max_iter, t_parada_antecipada, eh_maximizacao):
    # Ponto inicial: limite inferior do domínio [cite: 22]
    x1_atual = dominio_x1_lim[0]
    x2_atual = dominio_x2_lim[0]
    
    if not verificar_limites(x1_atual, x2_atual, dominio_x1_lim, dominio_x2_lim):
        x1_atual = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
        x2_atual = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
        
    custo_atual = func_obj(x1_atual, x2_atual)

    x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
    iter_sem_melhora_global = 0
    
    for i in range(max_iter):
        N_VIZINHOS_POR_ITERACAO = 10 
        x1_candidato_iter, x2_candidato_iter, custo_candidato_iter = x1_atual, x2_atual, custo_atual
        achou_melhor_nesta_iteracao = False

        for _ in range(N_VIZINHOS_POR_ITERACAO):
            # Gera perturbação. O critério é |x_best - y| <= epsilon [cite: 23]
            # Esta geração tenta passos aleatórios limitados por epsilon.
            pert_x1 = random.uniform(-epsilon, epsilon) 
            pert_x2 = random.uniform(-epsilon, epsilon)

            viz_x1 = x1_atual + pert_x1
            viz_x2 = x2_atual + pert_x2

            if not verificar_limites(viz_x1, viz_x2, dominio_x1_lim, dominio_x2_lim):
                continue

            custo_vizinho = func_obj(viz_x1, viz_x2)

            if eh_maximizacao:
                if custo_vizinho > custo_candidato_iter:
                    x1_candidato_iter, x2_candidato_iter, custo_candidato_iter = viz_x1, viz_x2, custo_vizinho
                    achou_melhor_nesta_iteracao = True
            else: 
                if custo_vizinho < custo_candidato_iter:
                    x1_candidato_iter, x2_candidato_iter, custo_candidato_iter = viz_x1, viz_x2, custo_vizinho
                    achou_melhor_nesta_iteracao = True
        
        if achou_melhor_nesta_iteracao:
            x1_atual, x2_atual, custo_atual = x1_candidato_iter, x2_candidato_iter, custo_candidato_iter
            
            if eh_maximizacao:
                if custo_atual > custo_melhor_geral:
                    x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
                    iter_sem_melhora_global = 0
                else:
                    iter_sem_melhora_global +=1 
            else: 
                if custo_atual < custo_melhor_geral:
                    x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
                    iter_sem_melhora_global = 0
                else:
                    iter_sem_melhora_global +=1
        else:
            iter_sem_melhora_global += N_VIZINHOS_POR_ITERACAO 

        if iter_sem_melhora_global >= t_parada_antecipada:
            break
            
    return x1_melhor_geral, x2_melhor_geral, custo_melhor_geral

def busca_local_aleatoria(func_obj, dominio_x1_lim, dominio_x2_lim, sigma, 
                           max_iter, t_parada_antecipada, eh_maximizacao):
    # Candidato inicial x_best gerado por distribuição uniforme [cite: 25]
    x1_atual = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
    x2_atual = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
    custo_atual = func_obj(x1_atual, x2_atual)

    x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
    iter_sem_melhora_global = 0

    for i in range(max_iter):
        # Gera candidato 'y' a partir de 'x_atual' com desvio padrão sigma [cite: 25]
        pert_x1 = random.gauss(0, sigma)
        pert_x2 = random.gauss(0, sigma)

        cand_x1 = x1_atual + pert_x1
        cand_x2 = x2_atual + pert_x2

        cand_x1 = max(dominio_x1_lim[0], min(cand_x1, dominio_x1_lim[1]))
        cand_x2 = max(dominio_x2_lim[0], min(cand_x2, dominio_x2_lim[1]))
        
        custo_candidato = func_obj(cand_x1, cand_x2)
        
        if eh_maximizacao:
            if custo_candidato > custo_atual:
                x1_atual, x2_atual, custo_atual = cand_x1, cand_x2, custo_candidato
        else: 
            if custo_candidato < custo_atual:
                x1_atual, x2_atual, custo_atual = cand_x1, cand_x2, custo_candidato
        
        if eh_maximizacao:
            if custo_atual > custo_melhor_geral:
                x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
                iter_sem_melhora_global = 0
            else:
                iter_sem_melhora_global += 1
        else: 
            if custo_atual < custo_melhor_geral:
                x1_melhor_geral, x2_melhor_geral, custo_melhor_geral = x1_atual, x2_atual, custo_atual
                iter_sem_melhora_global = 0
            else:
                iter_sem_melhora_global += 1
        
        if iter_sem_melhora_global >= t_parada_antecipada:
            break
            
    return x1_melhor_geral, x2_melhor_geral, custo_melhor_geral

# --- Algoritmo: Busca Aleatória Global (GRS) ---
def busca_aleatoria_global(func_obj, dominio_x1_lim, dominio_x2_lim, 
                            max_iter, t_parada_antecipada, eh_maximizacao):
    # Gera um candidato inicial aleatório para começar
    x1_melhor = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
    x2_melhor = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
    custo_melhor = func_obj(x1_melhor, x2_melhor)
    iter_sem_melhora = 0

    for i in range(max_iter):
        # Gera novo candidato através de um número aleatório com distribuição uniforme [cite: 26]
        cand_x1 = random.uniform(dominio_x1_lim[0], dominio_x1_lim[1])
        cand_x2 = random.uniform(dominio_x2_lim[0], dominio_x2_lim[1])
        
        custo_candidato = func_obj(cand_x1, cand_x2)

        if eh_maximizacao:
            if custo_candidato > custo_melhor:
                x1_melhor, x2_melhor, custo_melhor = cand_x1, cand_x2, custo_candidato
                iter_sem_melhora = 0
            else:
                iter_sem_melhora += 1
        else: 
            if custo_candidato < custo_melhor:
                x1_melhor, x2_melhor, custo_melhor = cand_x1, cand_x2, custo_candidato
                iter_sem_melhora = 0
            else:
                iter_sem_melhora += 1
        
        if iter_sem_melhora >= t_parada_antecipada:
            break
            
    return x1_melhor, x2_melhor, custo_melhor