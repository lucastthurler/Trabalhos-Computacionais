import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

MU = 0.012150

X0 = 0.994
Y0 = 0.0
U0 = 0.0
V0 = -2.011752

ESTADO_INICIAL = np.array([X0, Y0, U0, V0], dtype=float)

T = 5.432 * np.pi

PASSOS = [0.01, 0.001, 0.0001]
NUM_PERIODOS = [1, 2, 4]


def sistema_prtc(estado, mu=MU):
    x, y, u, v = estado

    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    dx = u
    dy = v

    du = (
        2 * v
        + x
        - (1 - mu) * (x + mu) / r1 ** 3
        - mu * (x - 1 + mu) / r2 ** 3
    )

    dv = (
        -2 * u
        + y
        - (1 - mu) * y / r1 ** 3
        - mu * y / r2 ** 3
    )

    return np.array([dx, dy, du, dv], dtype=float)


def rk4_step(estado, dt, mu=MU):
    k1 = sistema_prtc(estado, mu)
    k2 = sistema_prtc(estado + 0.5 * dt * k1, mu)
    k3 = sistema_prtc(estado + 0.5 * dt * k2, mu)
    k4 = sistema_prtc(estado + dt * k3, mu)

    return estado + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrar(estado_inicial, tempo_final, dt, mu=MU):
    tempo = 0.0
    estado = estado_inicial.copy()

    tempos = [tempo]
    estados = [estado.copy()]

    while tempo < tempo_final:
        dt_atual = min(dt, tempo_final - tempo)

        estado = rk4_step(estado, dt_atual, mu)
        tempo += dt_atual

        tempos.append(tempo)
        estados.append(estado.copy())

    return np.array(tempos), np.array(estados)


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


def plotar_orbita(estados, dt, numero_periodos, lagrange, mu=MU):
    x = estados[:, 0]
    y = estados[:, 1]

    x_inicial = estados[0, 0]
    y_inicial = estados[0, 1]

    x_final = estados[-1, 0]
    y_final = estados[-1, 1]

    terra_x = -mu
    terra_y = 0.0

    lua_x = 1 - mu
    lua_y = 0.0

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(
        x,
        y,
        color="cyan",
        linewidth=1.2,
        label="Órbita"
    )

    ax.scatter(
        x_inicial,
        y_inicial,
        color="orange",
        s=65,
        zorder=6,
        label="Estado inicial"
    )

    ax.scatter(
        x_final,
        y_final,
        color="lime",
        s=65,
        zorder=6,
        label="Estado final"
    )

    ax.scatter(
        terra_x,
        terra_y,
        color="yellow",
        s=180,
        zorder=5,
        label="Terra (P1)"
    )

    ax.scatter(
        lua_x,
        lua_y,
        color="white",
        s=70,
        zorder=5,
        label="Lua (P2)"
    )

    for nome, (lx, ly) in lagrange.items():
        ax.scatter(
            lx,
            ly,
            color="red",
            s=40,
            zorder=5
        )

        ax.text(
            lx,
            ly + 0.05,
            nome,
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom"
        )

    limite_x = max(
        1.5,
        np.max(np.abs(x)),
        abs(terra_x),
        abs(lua_x),
        max(abs(lx) for lx, ly in lagrange.values())
    )

    limite_y = max(
        1.5,
        np.max(np.abs(y)),
        max(abs(ly) for lx, ly in lagrange.values())
    )

    margem = 0.10

    limite_x = limite_x * (1 + margem)
    limite_y = limite_y * (1 + margem)

    limite = max(limite_x, limite_y)

    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)

    t_final_pi = numero_periodos * 5.432

    titulo = (
        "Análise de Periodicidade - Órbita de Arenstorf\n"
        f"μ = {mu:.6f} | RK4 Clássico | h = {dt}\n"
        f"t ∈ [0, {t_final_pi:.3f}π] | {numero_periodos} período(s)"
    )

    ax.set_title(titulo, fontsize=12, pad=12)

    ax.set_xlabel("x (adimensional)", fontsize=11)
    ax.set_ylabel("y (adimensional)", fontsize=11)

    ax.set_aspect("equal", adjustable="box")

    ax.grid(
        True,
        linestyle="-",
        color="#333333",
        alpha=0.6
    )

    ax.legend(
        loc="upper right",
        facecolor="#111111",
        edgecolor="#555555",
        fontsize=9
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plotar_tabela_periodicidade(tabela_periodicidade):
    tabela_visual = tabela_periodicidade[
        [
            "Passo",
            "Períodos",
            "Tempo final",
            "|Δx|",
            "|Δy|",
            "|Δu|",
            "|Δv|",
            "Erro total"
        ]
    ].copy()

    tabela_formatada = pd.DataFrame()

    tabela_formatada["Passo"] = tabela_visual["Passo"].map(
        lambda valor: f"{valor:.4f}"
    )

    tabela_formatada["Períodos"] = tabela_visual["Períodos"].astype(int)

    tabela_formatada["Tempo final"] = tabela_visual["Tempo final"].map(
        lambda valor: f"{valor:.6f}"
    )

    tabela_formatada["|Δx|"] = tabela_visual["|Δx|"].map(
        lambda valor: f"{valor:.6e}"
    )

    tabela_formatada["|Δy|"] = tabela_visual["|Δy|"].map(
        lambda valor: f"{valor:.6e}"
    )

    tabela_formatada["|Δu|"] = tabela_visual["|Δu|"].map(
        lambda valor: f"{valor:.6e}"
    )

    tabela_formatada["|Δv|"] = tabela_visual["|Δv|"].map(
        lambda valor: f"{valor:.6e}"
    )

    tabela_formatada["Erro total"] = tabela_visual["Erro total"].map(
        lambda valor: f"{valor:.6e}"
    )

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

    ax.set_title(
        "Tabela de Análise da Periodicidade",
        fontsize=16,
        pad=20,
        color="white"
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


lagrange = calcular_pontos_lagrange()

dados_tabela = []

for dt in PASSOS:
    for numero_periodos in NUM_PERIODOS:

        tempo_final = numero_periodos * T

        tempos, estados = integrar(
            ESTADO_INICIAL,
            tempo_final,
            dt,
            MU
        )

        estado_final = estados[-1]

        diferenca, erro_absoluto, erro_total = calcular_erro_periodicidade(
            ESTADO_INICIAL,
            estado_final
        )

        plotar_orbita(
            estados,
            dt,
            numero_periodos,
            lagrange,
            MU
        )

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

print(
    tabela_periodicidade.to_string(
        index=False,
        float_format=lambda x: f"{x:.10e}"
    )
)

plotar_tabela_periodicidade(tabela_periodicidade)