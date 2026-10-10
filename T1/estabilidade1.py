import numpy as np
import matplotlib.pyplot as plt

def sistema(t, Y, mu):
    x, y, u, v = Y
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    return np.array([
        u,
        v,
        2*v + x - ((1 - mu)*(x + mu))/(r1**3) - (mu*(x - 1 + mu))/(r2**3),
        -2*u + y - ((1 - mu)*y)/(r1**3) - (mu*y)/(r2**3)
    ])

def rk4(f, t, y, h, mu):
    k1 = f(t, y, mu)
    k2 = f(t + h/2, y + h*k1/2, mu)
    k3 = f(t + h/2, y + h*k2/2, mu)
    k4 = f(t + h, y + h*k3, mu)

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def simular_trajetoria(Y0, h, tf, mu):
    Y = Y0.copy()
    traj_x, traj_y = [], []
    t = 0
    while t < tf:
        traj_x.append(Y[0])
        traj_y.append(Y[1])
        Y = rk4(sistema, t, Y, h, mu)
        t += h
    return traj_x, traj_y

def poincare(Y0, h, tf, mu):
    Y = Y0.copy()
    t = 0
    px, pu = [], []
    while t < tf:
        Y_prev = Y.copy()
        Y = rk4(sistema, t, Y, h, mu)
        t += h

        if Y_prev[1] < 0 and Y[1] >= 0:
            alpha = Y_prev[1] / (Y_prev[1] - Y[1])
            x = Y_prev[0] + alpha*(Y[0] - Y_prev[0])
            u = Y_prev[2] + alpha*(Y[2] - Y_prev[2])
            v = Y_prev[3] + alpha*(Y[3] - Y_prev[3])

            if v > 0:
                px.append(x)
                pu.append(u)

    return px, pu

plt.style.use('dark_background')

mu = 0.01215
h = 0.0001
tf_orbita = 5.432 * np.pi
tf_poincare = 44 * np.pi

Y_base = np.array([0.994, 0.0, 0.0, -2.011752])
epsilons = [1e-6, 1e-5, 1e-4, 1e-3]
cores = ['#00ffcc', '#ffcc00', '#ff6600', '#ff0055']

tx_nom, ty_nom = simular_trajetoria(Y_base, h, tf_orbita, mu)

dados = {}
for eps in epsilons:
    Y_pert = Y_base.copy()
    Y_pert[0] += eps
    
    tx, ty = simular_trajetoria(Y_pert, h, tf_orbita, mu)
    px, pu = poincare(Y_pert, h, tf_poincare, mu)
    
    dados[eps] = {'tx': tx, 'ty': ty, 'px': px, 'pu': pu}

fig1, ax1 = plt.subplots(figsize=(9, 7))
ax1.plot(tx_nom, ty_nom, color='white', linestyle='--', alpha=0.5, label="Nominal (ε=0)")

for eps, cor in zip(epsilons, cores):
    ax1.plot(dados[eps]['tx'], dados[eps]['ty'], color=cor, alpha=0.8, label=f"ε = 10$^{{{int(np.log10(eps))}}}$")

ax1.scatter(-mu, 0, color='yellow', s=120, zorder=5)
ax1.text(-mu, 0.05, "Terra", color='yellow', ha='center')
ax1.scatter(1 - mu, 0, color='white', s=60, zorder=5)
ax1.text(1 - mu, 0.05, "Lua", color='white', ha='center')

ax1.set_title(f"Órbitas Sobrepostas - Plano x-y\nPRTC | μ={mu} | h={h} | t ∈ [0, 5.432π]")
ax1.set_xlabel("x (adim.)")
ax1.set_ylabel("y (adim.)")
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', facecolor='#111111')
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

for i, (eps, cor) in enumerate(zip(epsilons, cores)):
    ax = axes2[i]
    ax.plot(tx_nom, ty_nom, color='white', linestyle='--', alpha=0.4, label="Nominal")
    ax.plot(dados[eps]['tx'], dados[eps]['ty'], color=cor, label=f"ε = 10$^{{{int(np.log10(eps))}}}$")
    
    ax.scatter(-mu, 0, color='yellow', s=60)
    ax.scatter(1 - mu, 0, color='white', s=30)
    
    ax.set_title(f"Perturbação ε = 10$^{{{int(np.log10(eps))}}}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

fig2.suptitle(f"Órbitas Separadas - Plano x-y\nt ∈ [0, 5.432π]", fontsize=14)
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(Y_base[0], Y_base[2], color='white', s=90, marker='X', zorder=6, label="Nominal Teórico")

for eps, cor in zip(epsilons, cores):
    ax3.scatter(dados[eps]['px'], dados[eps]['pu'], s=25, color=cor, alpha=0.8, label=f"ε = 10$^{{{int(np.log10(eps))}}}$")

ax3.set_title(f"Seção de Poincaré Sobreposta (y=0, v>0)\nPRTC | μ={mu} | h={h} | t ∈ [0, 44π]")
ax3.set_xlabel("x (posição)")
ax3.set_ylabel("u (velocidade em x)")
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', facecolor='#111111')
plt.tight_layout()
plt.show()

fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
axes4 = axes4.flatten()

for i, (eps, cor) in enumerate(zip(epsilons, cores)):
    ax = axes4[i]
    ax.scatter(Y_base[0], Y_base[2], color='white', s=70, marker='X', label="Nominal")
    ax.scatter(dados[eps]['px'], dados[eps]['pu'], s=20, color=cor, label=f"ε = 10$^{{{int(np.log10(eps))}}}$")
    
    ax.set_title(f"Poincaré para ε = 10$^{{{int(np.log10(eps))}}}$")
    ax.set_xlabel("x (posição)")
    ax.set_ylabel("u (velocidade em x)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')

fig4.suptitle(f"Seção de Poincaré Separada (y=0, v>0)\nt ∈ [0, 44π]", fontsize=14)
plt.tight_layout()
plt.show()