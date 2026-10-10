import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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

def jacobi(Y, mu):
    x, y, u, v = Y
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    return x**2 + y**2 + 2*(1 - mu)/r1 + 2*mu/r2 - (u**2 + v**2)

mu = 0.012150
Y0 = np.array([0.994, 0.0, 0.0, -2.011752])

tf = 5.432 * np.pi 
hs = [0.01, 0.001, 0.0001]

COR_FUNDO = '#0c1016'
COR_TEXTO = '#babbbd'
COR_TITULO = '#ffffff'
COR_GRADE = '#444c56'
COR_LINHA_ERRO = '#c11b65'

plt.rcParams.update({
    'figure.facecolor': COR_FUNDO,
    'axes.facecolor': COR_FUNDO,
    'axes.edgecolor': COR_GRADE,
    'axes.labelcolor': COR_TEXTO,
    'xtick.color': COR_TEXTO,
    'ytick.color': COR_TEXTO,
    'text.color': COR_TITULO,
    'grid.color': COR_GRADE,
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'font.family': 'sans-serif',
    'legend.facecolor': COR_FUNDO,
    'legend.edgecolor': COR_GRADE,
    'legend.fontsize': 10
})

C0 = jacobi(Y0, mu)

for h in hs:
    Y = Y0.copy()
    t = 0.0

    erros = []
    tempos = []

    while t < tf:
        Y = rk4(sistema, t, Y, h, mu)
        t += h

        C = jacobi(Y, mu)
        erro = abs((C - C0) / C0)

        erros.append(max(erro, 1e-16))
        tempos.append(t / np.pi)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tempos, erros, color=COR_LINHA_ERRO, lw=1.5)

    ax.set_yscale("log")
    
    ax.set_ylim(1e-8, 1e-2)
    
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    
    ax.margins(x=0.05)

    ax.grid(True, which='major', axis='both', alpha=0.5)

    titulo = f"Conservação da Integral de Jacobi — RK4 (h = {h})\nC_J,0 = {C0:.6f}"
    ax.set_title(titulo, fontsize=12, pad=12)
    ax.set_xlabel("Tempo t/π", fontsize=11)
    ax.set_ylabel("Erro relativo |ΔC_J / C_J,0|", fontsize=11)

    plt.tight_layout()

plt.show()