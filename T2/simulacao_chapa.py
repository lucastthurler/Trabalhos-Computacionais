from parametros_e_metodos import resolver_conducao
import matplotlib.pyplot as plt

t_alvo = 5000
h = 0.05
dt = 25.0
beta = 0.5

X, Y, T_num, _ = resolver_conducao(h, h, dt, beta, t_final=t_alvo)

plt.figure(figsize=(8, 6))

mapa_calor = plt.contourf(X, Y, T_num, levels=100, cmap='hot')

barra = plt.colorbar(mapa_calor)
barra.set_label('Temperatura (K)')

plt.title(f'Distribuição de Temperatura 2D na Chapa (t = {t_alvo}s)(β = {beta})')
plt.xlabel('Posição x (m)')
plt.ylabel('Posição y (m)')

plt.tight_layout()
plt.savefig('chapa_2d_heatmap.png')
plt.show()