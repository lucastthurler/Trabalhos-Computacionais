import numpy as np
import matplotlib.pyplot as plt

MU = 0.012150
ESTADO_INICIAL = np.array([0.994, 0.0, 0.0, -2.011752], dtype=float)
T_FINAL = 44 * np.pi
DT = 0.0001  # Passo de integração refinado

def sistema_prtc(estado, mu=MU):
    x, y, u, v = estado
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    dx = u
    dy = v
    du = 2 * v + x - (1 - mu) * (x + mu) / (r1 ** 3) - mu * (x - 1 + mu) / (r2 ** 3)
    dv = -2 * u + y - (1 - mu) * y / (r1 ** 3) - mu * y / (r2 ** 3)
    
    return np.array([dx, dy, du, dv], dtype=float)

def rk4_step(estado, dt, mu=MU):
    k1 = sistema_prtc(estado, mu)
    k2 = sistema_prtc(estado + 0.5 * dt * k1, mu)
    k3 = sistema_prtc(estado + 0.5 * dt * k2, mu)
    k4 = sistema_prtc(estado + dt * k3, mu)
    return estado + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def gerar_secao_poincare(estado_inicial, t_final, dt):
    tempo = 0.0
    estado = estado_inicial.copy()
    
    pontos_x = []
    pontos_u = []
    
    while tempo < t_final:
        dt_atual = min(dt, t_final - tempo)
        estado_prox = rk4_step(estado, dt_atual)
        if estado[1] <= 0 and estado_prox[1] > 0 and estado_prox[3] > 0:
            alpha = -estado[1] / (estado_prox[1] - estado[1])
            x_cruzamento = estado[0] + alpha * (estado_prox[0] - estado[0])
            u_cruzamento = estado[2] + alpha * (estado_prox[2] - estado[2])
            
            pontos_x.append(x_cruzamento)
            pontos_u.append(u_cruzamento)
            
        estado = estado_prox
        tempo += dt_atual
        
    return np.array(pontos_x), np.array(pontos_u)

px, pu = gerar_secao_poincare(ESTADO_INICIAL, T_FINAL, DT)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    px, pu, 
    color="cyan", 
    s=25, 
    edgecolor="white", 
    linewidth=0.5, 
    label="Interseções ($y = 0, v > 0$)"
)

ax.scatter(
    ESTADO_INICIAL[0], ESTADO_INICIAL[2], 
    color="orange", 
    s=70, 
    zorder=5, 
    label="Ponto Teórico Inicial"
)

ax.set_title(
    "Seção de Poincaré ($y = 0, v > 0$) no Plano $x - u$\n"
    f"$\\tau \\in [0, 44\\pi]$ | $h = {DT}$", 
    fontsize=12, 
    pad=12
)
ax.set_xlabel("x (posição)", fontsize=11)
ax.set_ylabel("u (velocidade em x)", fontsize=11)
ax.grid(True, linestyle="--", color="#333333", alpha=0.6)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    facecolor="#111111",
    edgecolor="#555555",
    fontsize=10
)

plt.tight_layout()
plt.show()