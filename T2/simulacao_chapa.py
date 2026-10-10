from parametros_e_metodos import resolver_conducao
import matplotlib.pyplot as plt

print("--- SIMULAÇÃO 2D DA CHAPA DE CALOR ---")
t_alvo = 5000 # Mude este tempo para ver a evolução (ex: 1250, 2500, 5000)
h = 0.05
dt = 25.0
beta = 0.5

print(f"Calculando distribuição térmica para t = {t_alvo}s...")
X, Y, T_num, _ = resolver_conducao(h, h, dt, beta, t_final=t_alvo)

# Cria a figura para o mapa de calor
plt.figure(figsize=(8, 6))

# plt.contourf gera os contornos preenchidos simulando calor
mapa_calor = plt.contourf(X, Y, T_num, levels=100, cmap='hot')

# Adiciona a barra lateral de referência de temperatura
barra = plt.colorbar(mapa_calor)
barra.set_label('Temperatura (K)')

# Estilização
plt.title(f'Distribuição de Temperatura 2D na Chapa (t = {t_alvo}s)(β = {beta})')
plt.xlabel('Posição x (m)')
plt.ylabel('Posição y (m)')

# Salva e exibe
plt.tight_layout()
plt.savefig('chapa_2d_heatmap.png')
plt.show()