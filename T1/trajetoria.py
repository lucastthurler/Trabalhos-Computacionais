import numpy as np
import matplotlib.pyplot as plt

def sistema(t, Y, mu):
    x, y, u, v = Y

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    dxdt = u
    dydt = v

    dudt = 2*v + x - ((1 - mu)*(x + mu))/(r1**3) - (mu*(x - 1 + mu))/(r2**3)
    dvdt = -2*u + y - ((1 - mu)*y)/(r1**3) - (mu*y)/(r2**3)

    return np.array([dxdt, dydt, dudt, dvdt])

def rk4(f, t, y, h, mu):
    k1 = f(t, y, mu)
    k2 = f(t + h/2, y + h*k1/2, mu)
    k3 = f(t + h/2, y + h*k2/2, mu)
    k4 = f(t + h, y + h*k3, mu)

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

mu = 0.012150
Y0 = np.array([0.994, 0.0, 0.0, -2.011752])

h = 0.001
t0 = 0.0
tf = 5.432 * np.pi

def simular():
    Y = Y0.copy()
    t = t0

    traj_x, traj_y = [], []

    while t < tf:
        traj_x.append(Y[0])
        traj_y.append(Y[1])

        Y = rk4(sistema, t, Y, h, mu)
        t += h

    return np.array(traj_x), np.array(traj_y)

traj_x, traj_y = simular()

L1 = (0.8369, 0)
L2 = (1.1557, 0)
L3 = (-1.005, 0)
L4 = (0.5 - mu, np.sqrt(3)/2)
L5 = (0.5 - mu, -np.sqrt(3)/2)

lagrange_points = [L1, L2, L3, L4, L5]
labels = ["L1", "L2", "L3", "L4", "L5"]

def ajustar_limites_globais(ax, x_orb, y_orb, pontos_lagrange, mu, margem=0.1):
    todos_x = list(x_orb) + [p[0] for p in pontos_lagrange] + [-mu, 1 - mu]
    todos_y = list(y_orb) + [p[1] for p in pontos_lagrange] + [0, 0]

    xmin, xmax = np.min(todos_x), np.max(todos_x)
    ymin, ymax = np.min(todos_y), np.max(todos_y)

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(xmin - margem*dx, xmax + margem*dx)
    ax.set_ylim(ymin - margem*dy, ymax + margem*dy)

def formatar_pi(valor):
    k = valor / np.pi

    if np.isclose(k, round(k)):
        return f"{int(round(k))}π"
    else:
        return f"{k:.3f}π"

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(9, 9))

ax.plot(traj_x, traj_y, color='cyan', lw=1.5, label="Órbita")
ax.plot(traj_x[-1], traj_y[-1], 'o', color='green', label="Satélite")

ax.scatter(-mu, 0, color='yellow', s=120, label="Terra (P1)")
ax.scatter(1 - mu, 0, color='white', s=60, label="Lua (P2)")

for (x, y), lab in zip(lagrange_points, labels):
    ax.scatter(x, y, color='red', s=40)
    ax.text(x, y + 0.05, lab, color='red', ha='center')

ajustar_limites_globais(ax, traj_x, traj_y, lagrange_points, mu)

titulo = (
    "Problema Restrito dos 3 Corpos (PRTC)\n"
    f"μ = {mu} | Método: RK4 | h = {h}\n"
    f"t ∈ [{t0}, {formatar_pi(tf)}] | "
    f"CI: (x0={Y0[0]}, y0={Y0[1]}, u0={Y0[2]}, v0={Y0[3]})"
)

ax.set_title(titulo, fontsize=11)
ax.set_xlabel("x(adimensional)")
ax.set_ylabel("y(adimensional)")
ax.grid(alpha=0.3)
ax.set_aspect('equal')

ax.legend(loc="upper right")

plt.show()