import numpy as np

# Tabela 1: Parâmetros físicos e geométricos do problema
ALPHA = 1.0e-5       # Difusividade térmica (m2/s)
LX = 1.0             # Comprimento (m)
LY = 1.0             # Altura (m)
T0 = 100.0           # Temperatura inicial característica (K)
T_FINAL = 5000.0     # Tempo final de simulação (s)
EPSILON = 1.0e-10    # Tolerância de convergência (K)

def solucao_analitica(x, y, t, alpha=ALPHA, lx=LX, ly=LY, t0=T0):
    """Calcula a solução analítica exata (Eq. 7)."""
    termo_x = np.sin(np.pi * x / lx)
    termo_y = np.sin(np.pi * y / ly)
    termo_t = np.exp(-alpha * np.pi**2 * (1/lx**2 + 1/ly**2) * t)
    return t0 * termo_x * termo_y * termo_t

def tdma(a, b, c, d):
    """Algoritmo de Thomas (TDMA) para sistemas tridiagonais."""
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)
    phi = np.zeros(n)
    
    # Condições de contorno de Dirichlet (T=0 nas bordas)
    P[0] = 0.0
    Q[0] = 0.0
    
    for i in range(1, n-1):
        den = a[i] - c[i] * P[i-1]
        P[i] = b[i] / den
        Q[i] = (d[i] + c[i] * Q[i-1]) / den
        
    Q[n-1] = 0.0
    phi[n-1] = 0.0
    
    for i in range(n-2, 0, -1):
        phi[i] = P[i] * phi[i+1] + Q[i]
        
    return phi

def resolver_conducao(dx, dy, dt, beta, t_final=T_FINAL, alpha=ALPHA, lx=LX, ly=LY, t0=T0, eps=EPSILON):
    """Resolve a EDP pelo método de diferenças finitas usando o esquema geral em beta."""
    nx = int(np.round(lx / dx)) + 1
    ny = int(np.round(ly / dy)) + 1
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    sx = alpha * dt / dx**2
    sy = alpha * dt / dy**2
    
    # Condição inicial
    T = t0 * np.sin(np.pi * X / lx) * np.sin(np.pi * Y / ly)
    T[:, 0] = 0; T[:, -1] = 0; T[0, :] = 0; T[-1, :] = 0
    
    nt = int(np.round(t_final / dt))
    iteracoes_totais = 0
    
    for n in range(nt):
        T_new = np.copy(T)
        
        if beta == 0:
            # Formulação explícita
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    T_new[i, j] = T[i,j] + sx * (T[i+1,j] - 2*T[i,j] + T[i-1,j]) + \
                                  sy * (T[i,j+1] - 2*T[i,j] + T[i,j-1])
        else:
            # Formulações Implícitas (Crank-Nicolson ou Totalmente Implícita)
            T_k = np.copy(T)
            erro = eps + 1
            iter_k = 0
            
            while erro >= eps:
                T_k_next = np.copy(T_k)
                
                # Varredura na direção X
                for j in range(1, ny-1):
                    a = np.zeros(nx); b = np.zeros(nx); c = np.zeros(nx); d = np.zeros(nx)
                    for i in range(1, nx-1):
                        a[i] = 1 + 2*beta*(sx + sy)
                        b[i] = beta * sx
                        c[i] = beta * sx
                        termo_n = T[i,j] + (1-beta)*sx*(T[i+1,j] - 2*T[i,j] + T[i-1,j]) + \
                                  (1-beta)*sy*(T[i,j+1] - 2*T[i,j] + T[i,j-1])
                        termo_k = beta * sy * (T_k[i, j+1] + T_k[i, j-1])
                        d[i] = termo_n + termo_k
                    T_k_next[:, j] = tdma(a, b, c, d)
                    
                # Varredura na direção Y
                T_k_next_y = np.copy(T_k_next)
                for i in range(1, nx-1):
                    a = np.zeros(ny); b = np.zeros(ny); c = np.zeros(ny); d = np.zeros(ny)
                    for j in range(1, ny-1):
                        a[j] = 1 + 2*beta*(sx + sy)
                        b[j] = beta * sy
                        c[j] = beta * sy
                        termo_n = T[i,j] + (1-beta)*sx*(T[i+1,j] - 2*T[i,j] + T[i-1,j]) + \
                                  (1-beta)*sy*(T[i,j+1] - 2*T[i,j] + T[i,j-1])
                        termo_k = beta * sx * (T_k_next[i+1, j] + T_k_next[i-1, j])
                        d[j] = termo_n + termo_k
                    T_k_next_y[i, :] = tdma(a, b, c, d)
                    
                # Checagem de erro
                erro = np.max(np.abs(T_k_next_y - T_k))
                T_k = np.copy(T_k_next_y)
                iter_k += 1
                
                if iter_k > 5000:
                    print("Aviso: Limite máximo de iterações atingido no passo", n)
                    break
                    
            T_new = np.copy(T_k)
            iteracoes_totais += iter_k
            
        T = np.copy(T_new)
        
    return X, Y, T, iteracoes_totais

def calcular_rms(T_num, T_ana):
    """Calcula o Erro Quadrático Médio (RMS)."""
    return np.sqrt(np.mean((T_num - T_ana)**2))