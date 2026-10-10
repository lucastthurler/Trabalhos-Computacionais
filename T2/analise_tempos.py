from parametros_e_metodos import resolver_conducao, solucao_analitica, LX, LY
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

print("--- GERANDO GRÁFICO COMPARATIVO DE TEMPOS ---")
h = 0.05
dt = 25.0
beta = 0.5  # Crank-Nicolson

tempos_alvo = [1250.0, 2500.0, 3750.0, 5000.0]
cores = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']

plt.figure(figsize=(10, 6))

for i, t_alvo in enumerate(tempos_alvo):
    print(f"Processando t = {t_alvo} s...")
    X, Y, T_num, _ = resolver_conducao(h, h, dt, beta, t_final=t_alvo)
    
    idx_x_meio = int(np.round((LX/2) / h))
    idx_y_meio = int(np.round((LY/2) / h))
    
    x_coords = X[:, idx_y_meio]
    T_num_x = T_num[:, idx_y_meio]
    
    # Suavização por Spline
    x_denso = np.linspace(x_coords.min(), x_coords.max(), 300)
    spline_x = make_interp_spline(x_coords, T_num_x, k=3)
    T_num_x_suave = spline_x(x_denso)
    
    # Plota cada curva com uma cor diferente
    plt.plot(x_denso, T_num_x_suave, linewidth=2, color=cores[i], label=f't = {int(t_alvo)} s')

plt.title('Evolução Temporal do Perfil de Temperatura ao longo de X (y = L_y/2)', fontsize=12, pad=15)
plt.xlabel('Coordenada x (m)', fontsize=11)
plt.ylabel('Temperatura (K)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Salva a imagem para o relatório
plt.savefig('comparativo_perfis_x.png', dpi=300)
plt.show()
print("\nGráfico comparativo salvo com sucesso como 'comparativo_perfis_x.png'!")