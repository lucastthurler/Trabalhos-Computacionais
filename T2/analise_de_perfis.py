from parametros_e_metodos import resolver_conducao, solucao_analitica, LX, LY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

t_alvo = 5000.0
h = 0.05
dt = 25.0
beta = 0.5

print(f"Calculando distribuição térmica para t = {t_alvo} s...")
X, Y, T_num, iteracoes = resolver_conducao(h, h, dt, beta, t_final=t_alvo)
T_ana = solucao_analitica(X, Y, t_alvo)

idx_x_meio = int(np.round((LX/2) / h))
idx_y_meio = int(np.round((LY/2) / h))

x_coords = X[:, idx_y_meio]
T_num_x = T_num[:, idx_y_meio]
T_ana_x = T_ana[:, idx_y_meio]

y_coords = Y[idx_x_meio, :]
T_num_y = T_num[idx_x_meio, :]
T_ana_y = T_ana[idx_x_meio, :]

erro_abs_x = np.abs(T_num_x - T_ana_x)
df_x = pd.DataFrame({
    'Coordenada x (m)': x_coords,
    'T_Numérica [K]': T_num_x,
    'T_Analítica [K]': T_ana_x,
    'Erro Abs. [K]': erro_abs_x
})

erro_abs_y = np.abs(T_num_y - T_ana_y)
df_y = pd.DataFrame({
    'Coordenada y (m)': y_coords,
    'T_Numérica [K]': T_num_y,
    'T_Analítica [K]': T_ana_y,
    'Erro Abs. [K]': erro_abs_y
})

def exibir_tabela_grafica(df, titulo, nome_ficheiro):
    df_fmt = df.copy()
    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].apply(lambda v: f"{v:.4f}".replace('.', ','))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    ax.axis('tight')
    
    tabela = ax.table(cellText=df_fmt.values, colLabels=df_fmt.columns, cellLoc='center', loc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.3)
    
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0')
            
    plt.title(titulo, pad=15, fontsize=12, weight='bold')
    plt.tight_layout()
    
    plt.savefig(nome_ficheiro, dpi=300)
    plt.show()

exibir_tabela_grafica(df_x, f"Tabela 4 - Perfil em X (y = {LY/2}m) em t = {t_alvo}s", "tabela_perfil_x.png")
exibir_tabela_grafica(df_y, f"Tabela 5 - Perfil em Y (x = {LX/2}m) em t = {t_alvo}s", "tabela_perfil_y.png")

x_denso = np.linspace(x_coords.min(), x_coords.max(), 300)
y_denso = np.linspace(y_coords.min(), y_coords.max(), 300)

spline_x = make_interp_spline(x_coords, T_num_x, k=3)
T_num_x_suave = spline_x(x_denso)

spline_y = make_interp_spline(y_coords, T_num_y, k=3)
T_num_y_suave = spline_y(y_denso)

plt.figure(figsize=(9, 5))
plt.plot(x_denso, T_num_x_suave, 'r--', linewidth=2.5, label='Numérica (TDMA)')
plt.plot(x_coords, T_ana_x, 'b', linewidth=1.5, label='Analítica')
plt.title(f'Perfil de Temperatura ao longo de X (y = {LY/2} m) em t = {t_alvo} s', pad=15)
plt.xlabel('Coordenada x (m)')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('perfil_x.png', dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(y_denso, T_num_y_suave, 'r--', linewidth=2.5, label='Numérica (TDMA)')
plt.plot(y_coords, T_ana_y, 'b', linewidth=1.5, label='Analítica')
plt.title(f'Perfil de Temperatura ao longo de Y (x = {LX/2} m) em t = {t_alvo} s', pad=15)
plt.xlabel('Coordenada y (m)')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('perfil_y.png', dpi=300)
plt.close()
