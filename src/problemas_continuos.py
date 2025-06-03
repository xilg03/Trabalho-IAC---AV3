import math
import random
import numpy as np

# --- Funções Objetivo ---

# Função 1 [cite: 29]
# f(x1,x2) = x1^2 + x2^2
# Domínio: x1, x2 em [-100, 100]
# Objetivo: Minimização
def funcao_objetivo_1(x1, x2):
    return x1**2 + x2**2

# Função 2 [cite: 31]
# f(x1,x2) = exp(-(x1^2+x2^2)) + 2*exp(-((x1-1.7)^2+(x2-1.7)^2))
# Domínio: x1 em [-2,4], x2 em [-2,5]
# Objetivo: Maximização
def funcao_objetivo_2(x1, x2):
    term1 = np.exp(-(x1**2 + x2**2))
    term2 = 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))
    return term1 + term2

# Função 3 (Ackley) [cite: 31]
# f(x1,x2) = -20*exp(-0.2*sqrt(0.5*(x1^2+x2^2))) - exp(0.5*(cos(2*pi*x1)+cos(2*pi*x2))) + 20 + e
# Domínio: x1, x2 em [-8,8]
# Objetivo: Minimização
def funcao_objetivo_3(x1, x2):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return term1 + term2 + 20 + np.e

# Função 4 (Rastrigin) [cite: 33]
# f(x1,x2) = (x1^2 - 10*cos(2*pi*x1) + 10) + (x2^2 - 10*cos(2*pi*x2) + 10)
# Domínio: x1, x2 em [-5.12, 5.12]
# Objetivo: Minimização
def funcao_objetivo_4(x1, x2):
    term_x1 = x1**2 - 10 * np.cos(2 * np.pi * x1) + 10
    term_x2 = x2**2 - 10 * np.cos(2 * np.pi * x2) + 10
    return term_x1 + term_x2

# Função 5 [cite: 33]
# f(x1,x2) = (x1*cos(x1))/20 + 2*exp(-(x1^2 + (x2-1)^2)) + 0.01*x1*x2
# Domínio: x1, x2 em [-10,10]
# Objetivo: Maximização
def funcao_objetivo_5(x1, x2):
    term1 = (x1 * np.cos(x1)) / 20.0
    term2 = 2 * np.exp(-(x1**2 + (x2 - 1)**2))
    term3 = 0.01 * x1 * x2
    return term1 + term2 + term3

# Função 6 [cite: 35]
# f(x1,x2) = x1*sin(4*pi*x1) - x2*sin(4*pi*x2 + pi) + 1
# Domínio: x1 em [-1,3], x2 em [-1,3]
# Objetivo: Maximização
def funcao_objetivo_6(x1, x2):
    term_x1 = x1 * np.sin(4 * np.pi * x1)
    term_x2 = x2 * np.sin(4 * np.pi * x2 + np.pi)
    return term_x1 - term_x2 + 1

# Função 7 [cite: 35]
# f(x1,x2) = -sin(x1)*(sin(x1^2/pi))^(20) - sin(x2)*(sin(2*x2^2/pi))^(20)
# (Nota: 2*10 no expoente é 20)
# Domínio: x1, x2 em [0, pi]
# Objetivo: Minimização
def funcao_objetivo_7(x1, x2):
    # Tratar casos onde sin(.) pode ser negativo antes da potência par,
    # ou onde o argumento de sin para a base da potência pode ser >1 ou <-1 (não deveria para seno).
    # Para (sin(algo))^20, se sin(algo) for negativo, o resultado será positivo.
    # Se |sin(algo)| > 1, isso seria um problema, mas sin está entre -1 e 1.
    
    # Evitar math domain error para base negativa em potência fracionária,
    # mas aqui a potência é inteira (20), então não é um problema.
    # Contudo, a precisão de ponto flutuante pode ser um fator.
    
    val_sin_term1_base = np.sin(x1**2 / np.pi)
    term1_base_pow_20 = np.pow(val_sin_term1_base, 20) # (sin(x1^2/pi))^20
    term1 = -np.sin(x1) * term1_base_pow_20

    val_sin_term2_base = np.sin(2 * x2**2 / np.pi)
    term2_base_pow_20 = np.pow(val_sin_term2_base, 20) # (sin(2*x2^2/pi))^20
    term2 = -np.sin(x2) * term2_base_pow_20
    
    return term1 + term2

# Função 8 [cite: 38]
# f(x1,x2) = -(x2+47)*sin(sqrt(|x1/2 + (x2+47)|)) - x1*sin(sqrt(|x1 - (x2+47)|))
# Domínio: x1 em [-200,20], x2 em [-200,20]
# Objetivo: Minimização
def funcao_objetivo_8(x1, x2):
    # Usar math.fabs para o valor absoluto |.|
    term1_factor = -(x2 + 47)
    term1_arg_sqrt = np.fabs(x1 / 2.0 + (x2 + 47))
    term1 = term1_factor * np.sin(np.sqrt(term1_arg_sqrt))

    term2_factor = -x1
    term2_arg_sqrt = np.fabs(x1 - (x2 + 47))
    term2 = term2_factor * np.sin(np.sqrt(term2_arg_sqrt))
    
    return term1 + term2

# --- Verificador de Limites ---
def verificar_limites(candidato_x1, candidato_x2, dominio_x1_lim, dominio_x2_lim):
    """Verifica se um candidato está dentro dos limites do domínio."""
    if not (dominio_x1_lim[0] <= candidato_x1 <= dominio_x1_lim[1]):
        return False
    if not (dominio_x2_lim[0] <= candidato_x2 <= dominio_x2_lim[1]):
        return False
    return True

# --- Algoritmo: Hill Climbing (Subida de Encosta) ---
def hill_climbing(func_obj, dominio_x1_lim, dominio_x2_lim, epsilon, 
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

# --- Algoritmo: Busca Aleatória Local (LRS) ---
def lrs(func_obj, dominio_x1_lim, dominio_x2_lim, sigma, 
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
def grs(func_obj, dominio_x1_lim, dominio_x2_lim, 
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