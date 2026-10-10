import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

MU = 0.012150
ESTADO_INICIAL = np.array([0.994, 0.0, 0.0, -2.011752], dtype=float)

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

def calcular_pontos_lagrange(mu=MU):
    return {
        "L1": (0.8369, 0.0),
        "L2": (1.1557, 0.0),
        "L3": (-1.005, 0.0),
        "L4": (0.5 - mu, np.sqrt(3) / 2),
        "L5": (0.5 - mu, -np.sqrt(3) / 2)
    }

def calcular_erro_periodicidade(estado_inicial, estado_final):
    diferenca = estado_final - estado_inicial
    erro_absoluto = np.abs(diferenca)
    erro_total = np.linalg.norm(diferenca)
    return diferenca, erro_absoluto, erro_total

def integrar(estado_inicial, tempo_final, dt, mu=MU):
    tempo = 0.0
    estado = estado_inicial.copy()
    tempos = [tempo]
    estados = [estado.copy()]
    while tempo < tempo_final:
        dt_atual = min(dt, tempo_final - tempo)
        estado = rk4(sistema, tempo, estado, dt_atual, mu)
        tempo += dt_atual
        tempos.append(tempo)
        estados.append(estado.copy())
    return np.array(tempos), np.array(estados)

def gerar_secao_poincare(estado_inicial, t_final, dt):
    tempo = 0.0
    estado = estado_inicial.copy()
    pontos_x = []
    pontos_u = []
    while tempo < t_final:
        dt_atual = min(dt, t_final - tempo)
        estado_prox = rk4(sistema, tempo, estado, dt_atual, MU)
        if estado[1] <= 0 and estado_prox[1] > 0 and estado_prox[3] > 0:
            alpha = -estado[1] / (estado_prox[1] - estado[1])
            x_cruzamento = estado[0] + alpha * (estado_prox[0] - estado[0])
            u_cruzamento = estado[2] + alpha * (estado_prox[2] - estado[2])
            pontos_x.append(x_cruzamento)
            pontos_u.append(u_cruzamento)
        estado = estado_prox
        tempo += dt_atual
    return np.array(pontos_x), np.array(pontos_u)

def obter_pontos_lagrange(mu):
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

def formatar_pi(valor):
    k = valor / np.pi
    if np.isclose(k, round(k)):
        return f"{int(round(k))}π"
    else:
        return f"{k:.3f}π"

def ajustar_limites_globais(ax, x_orb, y_orb, pontos_lagrange, mu, margem=0.1):
    todos_x = list(x_orb) + [p[0] for p in pontos_lagrange] + [-mu, 1 - mu]
    todos_y = list(y_orb) + [p[1] for p in pontos_lagrange] + [0, 0]
    xmin, xmax = np.min(todos_x), np.max(todos_x)
    ymin, ymax = np.min(todos_y), np.max(todos_y)
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - margem*dx, xmax + margem*dx)
    ax.set_ylim(ymin - margem*dy, ymax + margem*dy)

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

def simular_trajetoria_delta(mu, delta, h, tf):
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

plt.style.use('dark_background')

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

def analise_jacobi_erro():
    print("\n[1/6] Executando análise de Erro na Integral de Jacobi...")
    tf = 5.432 * np.pi 
    hs = [0.01, 0.001, 0.0001]
    C0 = jacobi(ESTADO_INICIAL, MU)
    
    for h in hs:
        Y = ESTADO_INICIAL.copy()
        t = 0.0
        erros = []
        tempos = []
        while t < tf:
            Y = rk4(sistema, t, Y, h, MU)
            t += h
            C = jacobi(Y, MU)
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

def analise_trajetoria():
    print("\n[2/6] Executando análise de Trajetória...")
    h = 0.001
    t0 = 0.0
    tf = 5.432 * np.pi
    
    Y = ESTADO_INICIAL.copy()
    t = t0
    traj_x, traj_y = [], []
    while t < tf:
        traj_x.append(Y[0])
        traj_y.append(Y[1])
        Y = rk4(sistema, t, Y, h, MU)
        t += h
    
    traj_x = np.array(traj_x)
    traj_y = np.array(traj_y)
    
    L1 = (0.8369, 0)
    L2 = (1.1557, 0)
    L3 = (-1.005, 0)
    L4 = (0.5 - MU, np.sqrt(3)/2)
    L5 = (0.5 - MU, -np.sqrt(3)/2)
    lagrange_points = [L1, L2, L3, L4, L5]
    labels = ["L1", "L2", "L3", "L4", "L5"]
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(traj_x, traj_y, color='cyan', lw=1.5, label="Órbita")
    ax.plot(traj_x[-1], traj_y[-1], 'o', color='green', label="Satélite")
    ax.scatter(-MU, 0, color='yellow', s=120, label="Terra (P1)")
    ax.scatter(1 - MU, 0, color='white', s=60, label="Lua (P2)")
    
    for (x, y), lab in zip(lagrange_points, labels):
        ax.scatter(x, y, color='red', s=40)
        ax.text(x, y + 0.05, lab, color='red', ha='center')
    
    ajustar_limites_globais(ax, traj_x, traj_y, lagrange_points, MU)
    
    titulo = (
        "Problema Restrito dos 3 Corpos (PRTC)\n"
        f"μ = {MU} | Método: RK4 | h = {h}\n"
        f"t ∈ [{t0}, {formatar_pi(tf)}] | "
        f"CI: (x0={ESTADO_INICIAL[0]}, y0={ESTADO_INICIAL[1]}, u0={ESTADO_INICIAL[2]}, v0={ESTADO_INICIAL[3]})"
    )
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("x(adimensional)")
    ax.set_ylabel("y(adimensional)")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def analise_poincare():
    print("\n[3/6] Executando análise de Seção de Poincaré...")
    T_FINAL = 44 * np.pi
    DT = 0.0001
    px, pu = gerar_secao_poincare(ESTADO_INICIAL, T_FINAL, DT)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(px, pu, color="cyan", s=25, edgecolor="white", linewidth=0.5, label="Interseções ($y = 0, v > 0$)")
    ax.scatter(ESTADO_INICIAL[0], ESTADO_INICIAL[2], color="orange", s=70, zorder=5, label="Ponto Teórico Inicial")
    ax.set_title(
        "Seção de Poincaré ($y = 0, v > 0$) no Plano $x - u$\n"
        f"$\\tau \\in [0, 44\\pi]$ | $h = {DT}$", 
        fontsize=12, pad=12
    )
    ax.set_xlabel("x (posição)", fontsize=11)
    ax.set_ylabel("u (velocidade em x)", fontsize=11)
    ax.grid(True, linestyle="--", color="#333333", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, facecolor="#111111", edgecolor="#555555", fontsize=10)
    plt.tight_layout()
    plt.show()

def analise_periodicidade():
    print("\n[4/6] Executando análise de Periodicidade...")
    T = 5.432 * np.pi
    PASSOS = [0.01, 0.001, 0.0001]
    NUM_PERIODOS = [1, 2, 4]
    lagrange = calcular_pontos_lagrange()
    dados_tabela = []
    
    for dt in PASSOS:
        for numero_periodos in NUM_PERIODOS:
            tempo_final = numero_periodos * T
            tempos, estados = integrar(ESTADO_INICIAL, tempo_final, dt, MU)
            estado_final = estados[-1]
            diferenca, erro_absoluto, erro_total = calcular_erro_periodicidade(ESTADO_INICIAL, estado_final)
            
            x = estados[:, 0]
            y = estados[:, 1]
            x_inicial = estados[0, 0]
            y_inicial = estados[0, 1]
            x_final = estados[-1, 0]
            y_final = estados[-1, 1]
            terra_x = -MU
            terra_y = 0.0
            lua_x = 1 - MU
            lua_y = 0.0
            
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.plot(x, y, color="cyan", linewidth=1.2, label="Órbita")
            ax.scatter(x_inicial, y_inicial, color="orange", s=65, zorder=6, label="Estado inicial")
            ax.scatter(x_final, y_final, color="lime", s=65, zorder=6, label="Estado final")
            ax.scatter(terra_x, terra_y, color="yellow", s=180, zorder=5, label="Terra (P1)")
            ax.scatter(lua_x, lua_y, color="white", s=70, zorder=5, label="Lua (P2)")
            
            for nome, (lx, ly) in lagrange.items():
                ax.scatter(lx, ly, color="red", s=40, zorder=5)
                ax.text(lx, ly + 0.05, nome, color="red", fontsize=10, fontweight="bold", ha="center", va="bottom")
            
            limite_x = max(1.5, np.max(np.abs(x)), abs(terra_x), abs(lua_x), max(abs(lx) for lx, ly in lagrange.values()))
            limite_y = max(1.5, np.max(np.abs(y)), max(abs(ly) for lx, ly in lagrange.values()))
            margem = 0.10
            limite_x = limite_x * (1 + margem)
            limite_y = limite_y * (1 + margem)
            limite = max(limite_x, limite_y)
            ax.set_xlim(-limite, limite)
            ax.set_ylim(-limite, limite)
            
            t_final_pi = numero_periodos * 5.432
            titulo = (
                "Análise de Periodicidade - Órbita de Arenstorf\n"
                f"μ = {MU:.6f} | RK4 Clássico | h = {dt}\n"
                f"t ∈ [0, {t_final_pi:.3f}π] | {numero_periodos} período(s)"
            )
            ax.set_title(titulo, fontsize=12, pad=12)
            ax.set_xlabel("x (adimensional)", fontsize=11)
            ax.set_ylabel("y (adimensional)", fontsize=11)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="-", color="#333333", alpha=0.6)
            ax.legend(loc="upper right", facecolor="#111111", edgecolor="#555555", fontsize=9)
            plt.tight_layout()
            plt.show()
            
            dados_tabela.append({
                "Passo": dt,
                "Períodos": numero_periodos,
                "Tempo final": tempo_final,
                "Δx": diferenca[0],
                "Δy": diferenca[1],
                "Δu": diferenca[2],
                "Δv": diferenca[3],
                "|Δx|": erro_absoluto[0],
                "|Δy|": erro_absoluto[1],
                "|Δu|": erro_absoluto[2],
                "|Δv|": erro_absoluto[3],
                "Erro total": erro_total
            })
    
    tabela_periodicidade = pd.DataFrame(dados_tabela)
    print("\nTabela de análise da periodicidade:\n")
    print(tabela_periodicidade.to_string(index=False, float_format=lambda x: f"{x:.10e}"))
    
    tabela_visual = tabela_periodicidade[["Passo", "Períodos", "Tempo final", "|Δx|", "|Δy|", "|Δu|", "|Δv|", "Erro total"]].copy()
    tabela_formatada = pd.DataFrame()
    tabela_formatada["Passo"] = tabela_visual["Passo"].map(lambda valor: f"{valor:.4f}")
    tabela_formatada["Períodos"] = tabela_visual["Períodos"].astype(int)
    tabela_formatada["Tempo final"] = tabela_visual["Tempo final"].map(lambda valor: f"{valor:.6f}")
    tabela_formatada["|Δx|"] = tabela_visual["|Δx|"].map(lambda valor: f"{valor:.6e}")
    tabela_formatada["|Δy|"] = tabela_visual["|Δy|"].map(lambda valor: f"{valor:.6e}")
    tabela_formatada["|Δu|"] = tabela_visual["|Δu|"].map(lambda valor: f"{valor:.6e}")
    tabela_formatada["|Δv|"] = tabela_visual["|Δv|"].map(lambda valor: f"{valor:.6e}")
    tabela_formatada["Erro total"] = tabela_visual["Erro total"].map(lambda valor: f"{valor:.6e}")
    
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis("off")
    tabela = ax.table(
        cellText=tabela_formatada.values,
        colLabels=tabela_formatada.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.25, 2.0)
    
    for (linha, coluna), celula in tabela.get_celld().items():
        celula.set_edgecolor("#444444")
        celula.get_text().set_color("black")
        if linha == 0:
            celula.set_facecolor("#d0d0d0")
            celula.get_text().set_color("black")
            celula.get_text().set_fontweight("bold")
        else:
            celula.set_facecolor("#f2f2f2")
            celula.get_text().set_color("black")
    
    ax.set_title("Tabela de Análise da Periodicidade", fontsize=16, pad=20, color="white")
    plt.tight_layout()
    plt.show()

def analise_estabilidade_linear_L4():
    print("\n[5/6] Executando análise de Estabilidade Linear em L4...")
    h = 0.001
    tf = 70 * np.pi
    delta = 0.0
    mus = [0.01, 0.0385, 0.04]
    
    for mu in mus:
        traj_x, traj_y = simular_trajetoria_delta(mu, delta, h, tf)
        pontos_L = obter_pontos_lagrange(mu)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(traj_x, traj_y, color='cyan', lw=0.8, label=f"Trajetória (δ={delta})")
        ax.plot(traj_x[0], traj_y[0], 's', color='lime', markersize=6, zorder=6, label="Ponto Inicial")
        ax.plot(traj_x[-1], traj_y[-1], 'X', color='magenta', markersize=7, zorder=6, label="Ponto Final")
        ax.scatter(-mu, 0, color='yellow', s=120, zorder=5, label="Terra")
        ax.scatter(1 - mu, 0, color='white', s=60, zorder=5, label="Lua")
        ax.text(-mu, -0.08, "Terra", color='yellow', ha='center', fontsize=9)
        ax.text(1 - mu, -0.08, "Lua", color='white', ha='center', fontsize=9)
        
        for nome, (lx, ly) in pontos_L.items():
            cor = 'red' if nome == 'L4' else 'orange'
            ax.scatter(lx, ly, color=cor, s=50, zorder=5)
            ax.text(lx, ly + 0.05, nome, color=cor, ha='center', fontsize=9, fontweight='bold')
        
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

def analise_estabilidade_perturbacoes():
    print("\n[6/6] Executando análise de Estabilidade com Perturbações...")
    mu = 0.01215
    h = 0.0001
    tf_orbita = 5.432 * np.pi
    tf_poincare = 44 * np.pi
    epsilons = [1e-6, 1e-5, 1e-4, 1e-3]
    cores = ['#00ffcc', '#ffcc00', '#ff6600', '#ff0055']
    
    tx_nom, ty_nom = simular_trajetoria(ESTADO_INICIAL, h, tf_orbita, mu)
    
    dados = {}
    for eps in epsilons:
        Y_pert = ESTADO_INICIAL.copy()
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
    ax3.scatter(ESTADO_INICIAL[0], ESTADO_INICIAL[2], color='white', s=90, marker='X', zorder=6, label="Nominal Teórico")
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
        ax.scatter(ESTADO_INICIAL[0], ESTADO_INICIAL[2], color='white', s=70, marker='X', label="Nominal")
        ax.scatter(dados[eps]['px'], dados[eps]['pu'], s=20, color=cor, label=f"ε = 10$^{{{int(np.log10(eps))}}}$")
        ax.set_title(f"Poincaré para ε = 10$^{{{int(np.log10(eps))}}}$")
        ax.set_xlabel("x (posição)")
        ax.set_ylabel("u (velocidade em x)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper left')
    fig4.suptitle(f"Seção de Poincaré Separada (y=0, v>0)\nt ∈ [0, 44π]", fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("ANÁLISES COMPLETAS - PRTC (Problema Restrito dos 3 Corpos)")
    print("="*60)
    
    analise_jacobi_erro()
    analise_trajetoria()
    analise_poincare()
    analise_periodicidade()
    analise_estabilidade_linear_L4()
    analise_estabilidade_perturbacoes()
    
    print("\n" + "="*60)
    print("Todas as análises foram concluídas!")
    print("="*60)

if __name__ == "__main__":
    main()
