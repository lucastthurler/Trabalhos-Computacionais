from parametros_e_metodos import resolver_conducao, solucao_analitica, LX, LY
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

h = 0.05
dt = 25.0
beta = 0.5

tempos_alvo = [1250.0, 2500.0, 3750.0, 5000.0]
cores = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']

plt.figure(figsize=(10, 6))

for i, t_alvo in enumerate(tempos_alvo):
    print(f"Processando t = {t_alvo} s (Eixo X)...")
    X, Y, T_num, iteracoes = resolver_conducao(h, h, dt, beta, t_final=t_alvo)
    
    idx_y_meio = int(np.round((LY/2) / h))
    x_coords = X[:, idx_y_meio]
    T_num_x = T_num[:, idx_y_meio]
    
    x_denso = np.linspace(x_coords.min(), x_coords.max(), 300)
    spline_x = make_interp_spline(x_coords, T_num_x, k=3)
    T_num_x_suave = spline_x(x_denso)
    
    plt.plot(x_denso, T_num_x_suave, linewidth=2, color=cores[i], label=f't = {int(t_alvo)} s')

plt.title(f'Evolução Temporal do Perfil de Temperatura ao longo de X (y = {LY/2}m)', fontsize=12, pad=15)
plt.xlabel('Coordenada x (m)', fontsize=11)
plt.ylabel('Temperatura (K)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('comparativo_perfis_x.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))

for i, t_alvo in enumerate(tempos_alvo):
    print(f"Processando t = {t_alvo} s (Eixo Y)...")
    X, Y, T_num, iteracoes = resolver_conducao(h, h, dt, beta, t_final=t_alvo)
    
    idx_x_meio = int(np.round((LX/2) / h))
    y_coords = Y[idx_x_meio, :]
    T_num_y = T_num[idx_x_meio, :]
    
    y_denso = np.linspace(y_coords.min(), y_coords.max(), 300)
    spline_y = make_interp_spline(y_coords, T_num_y, k=3)
    T_num_y_suave = spline_y(y_denso)
    
    plt.plot(y_denso, T_num_y_suave, linewidth=2, color=cores[i], label=f't = {int(t_alvo)} s')

plt.title(f'Evolução Temporal do Perfil de Temperatura ao longo de Y (x = {LX/2}m)', fontsize=12, pad=15)
plt.xlabel('Coordenada y (m)', fontsize=11)
plt.ylabel('Temperatura (K)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('comparativo_perfis_y.png', dpi=300)
plt.show()