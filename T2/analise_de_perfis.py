from parametros_e_metodos import resolver_conducao, solucao_analitica, LX, LY
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

print("--- PERFIS DE TEMPERATURA NO CENTRO ---")
# Tempo fixo para coleta dos perfis (O PDF sugere 1250, 2500, 3750 ou 5000 s)
t_alvo = 5000.0  
h = 0.05
dt = 25.0
beta = 0.5 # Usando Crank-Nicolson, pois é estável e muito preciso

print(f"Calculando distribuição térmica para t = {t_alvo} s...")
X, Y, T_num, _ = resolver_conducao(h, h, dt, beta, t_final=t_alvo)
T_ana = solucao_analitica(X, Y, t_alvo)

# Índices para o centro do domínio (x = L_x/2 e y = L_y/2)
idx_x_meio = int(np.round((LX/2) / h))
idx_y_meio = int(np.round((LY/2) / h))

# ----- Dados Originais Discretos -----
x_coords = X[:, idx_y_meio]
T_num_x = T_num[:, idx_y_meio]
T_ana_x = T_ana[:, idx_y_meio]

y_coords = Y[idx_x_meio, :]
T_num_y = T_num[idx_x_meio, :]
T_ana_y = T_ana[idx_x_meio, :]

# ----- 1. IMPRESSÃO DAS TABELAS NO TERMINAL -----
print(f"\nTABELA 1: Perfil ao longo de x (com y = {LY/2}m) no tempo t = {t_alvo}s:")
print(f"{'x [m]':>10} | {'T_Numérica [K]':>15} | {'T_Analítica [K]':>15}")
for i in range(len(x_coords)):
    print(f"{x_coords[i]:10.4f} | {T_num_x[i]:15.4f} | {T_ana_x[i]:15.4f}")

print(f"\nTABELA 2: Perfil ao longo de y (com x = {LX/2}m) no tempo t = {t_alvo}s:")
print(f"{'y [m]':>10} | {'T_Numérica [K]':>15} | {'T_Analítica [K]':>15}")
for j in range(len(y_coords)):
    print(f"{y_coords[j]:10.4f} | {T_num_y[j]:15.4f} | {T_ana_y[j]:15.4f}")

# ----- 2. GERAÇÃO DAS IMAGENS GRÁFICAS -----
# Interpolação para suavizar as curvas numéricas
x_denso = np.linspace(x_coords.min(), x_coords.max(), 300)
y_denso = np.linspace(y_coords.min(), y_coords.max(), 300)

spline_x = make_interp_spline(x_coords, T_num_x, k=3)
T_num_x_suave = spline_x(x_denso)

spline_y = make_interp_spline(y_coords, T_num_y, k=3)
T_num_y_suave = spline_y(y_denso)

# Gráfico 1: Perfil ao longo de X
plt.figure(figsize=(9, 5))
plt.plot(x_denso, T_num_x_suave, 'r--', linewidth=2.5, label='Numérica (TDMA)')
plt.plot(x_coords, T_ana_x, 'b-', linewidth=1.5, label='Analítica')
plt.title(f'Perfil de Temperatura ao longo de X (y = {LY/2} m) em t = {t_alvo} s', pad=15)
plt.xlabel('Coordenada x (m)')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('perfil_x.png', dpi=300) # Salva em alta resolução
plt.close()

# Gráfico 2: Perfil ao longo de Y
plt.figure(figsize=(9, 5))
plt.plot(y_denso, T_num_y_suave, 'r--', linewidth=2.5, label='Numérica (TDMA)')
plt.plot(y_coords, T_ana_y, 'b-', linewidth=1.5, label='Analítica')
plt.title(f'Perfil de Temperatura ao longo de Y (x = {LX/2} m) em t = {t_alvo} s', pad=15)
plt.xlabel('Coordenada y (m)')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('perfil_y.png', dpi=300) # Salva em alta resolução
plt.close()

print("\nSucesso! As imagens 'perfil_x.png' e 'perfil_y.png' foram salvas na sua pasta.")