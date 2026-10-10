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

def simular_trajetoria(mu, delta, h, tf):
    x0 = 0.5 - mu + delta
    y0 = np.sqrt(3)/2
    Y = np.array([x0, y0, 0.0, 0.0])

    t = 0
    traj_x, traj_y = [], []

    while t < tf:
        traj_x.append(Y[0])
        traj_y.append(Y[1])
        Y = rk4(sistema, t, Y, h, mu)
        t += h

    return np.array(traj_x), np.array(traj_y)

def obter_pontos_lagrange(mu):
    # Função do equilíbrio colinear (L1, L2, L3)
    def f(x):
        r1 = np.abs(x + mu)
        r2 = np.abs(x - 1 + mu)
        return x - (1 - mu) * np.sign(x + mu) / (r1**2) - mu * np.sign(x - 1 + mu) / (r2**2)

    def bissect(a, b):
        for _ in range(60):
            c = (a + b) / 2
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2

    x_L1 = bissect(-mu + 1e-5, 1 - mu - 1e-5)
    x_L2 = bissect(1 - mu + 1e-5, 2.5)
    x_L3 = bissect(-2.5, -mu - 1e-5)

    return {
        'L1': (x_L1, 0.0),
        'L2': (x_L2, 0.0),
        'L3': (x_L3, 0.0),
        'L4': (0.5 - mu, np.sqrt(3)/2),
        'L5': (0.5 - mu, -np.sqrt(3)/2)
    }

plt.style.use('dark_background')

h = 0.001
tf = 70 * np.pi
delta = 0.0
mus = [0.01, 0.0385, 0.04]

for mu in mus:
    traj_x, traj_y = simular_trajetoria(mu, delta, h, tf)
    pontos_L = obter_pontos_lagrange(mu)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Trajetória
    ax.plot(traj_x, traj_y, color='cyan', lw=0.8, label=f"Trajetória (δ={delta})")

    # Ponto Inicial e Ponto Final
    ax.plot(traj_x[0], traj_y[0], 's', color='lime', markersize=6, zorder=6, label="Ponto Inicial")
    ax.plot(traj_x[-1], traj_y[-1], 'X', color='magenta', markersize=7, zorder=6, label="Ponto Final")

    # Primários (Terra e Lua)
    ax.scatter(-mu, 0, color='yellow', s=120, zorder=5, label="Terra")
    ax.scatter(1 - mu, 0, color='white', s=60, zorder=5, label="Lua")
    ax.text(-mu, -0.08, "Terra", color='yellow', ha='center', fontsize=9)
    ax.text(1 - mu, -0.08, "Lua", color='white', ha='center', fontsize=9)

    # Todos os Pontos de Lagrange (L1 a L5)
    for nome, (lx, ly) in pontos_L.items():
        cor = 'red' if nome == 'L4' else 'orange'
        ax.scatter(lx, ly, color=cor, s=50, zorder=5)
        ax.text(lx, ly + 0.05, nome, color=cor, ha='center', fontsize=9, fontweight='bold')

    # Ajuste dinâmico de foco mantendo proporção 1:1 sem cortar nenhum elemento
    lx_vals = [p[0] for p in pontos_L.values()]
    ly_vals = [p[1] for p in pontos_L.values()]

    todos_x = np.concatenate([traj_x, [-mu, 1 - mu], lx_vals])
    todos_y = np.concatenate([traj_y, [0, 0], ly_vals])

    xmin, xmax = todos_x.min(), todos_x.max()
    ymin, ymax = todos_y.min(), todos_y.max()

    centro_x = (xmin + xmax) / 2
    centro_y = (ymin + ymax) / 2
    max_range = max(xmax - xmin, ymax - ymin)
    margem = 0.12

    ax.set_xlim(centro_x - max_range * (0.5 + margem), centro_x + max_range * (0.5 + margem))
    ax.set_ylim(centro_y - max_range * (0.5 + margem), centro_y + max_range * (0.5 + margem))

    ax.set_title(
        f"Estabilidade Linear em L4 | μ = {mu}\n"
        f"PRTC | RK4 | h = {h} | τ ∈ [0, 70π] | δ = {delta}",
        fontsize=11
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()