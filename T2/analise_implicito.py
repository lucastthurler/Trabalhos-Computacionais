from parametros_e_metodos import resolver_conducao, solucao_analitica, calcular_rms, T_FINAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 0.05
dx = dy = h
passos_tempo = [100.0, 50.0, 25.0, 12.5]
betas = [0.5, 1.0]
nomes_beta = {0.5: "Crank-Nicolson", 1.0: "Totalmente Implícito"}

dados_tabela = []

for beta in betas:
    nome = nomes_beta[beta]
    print(f"\nCalculando: {nome} (beta = {beta})")
    
    for dt in passos_tempo:
        print(f"  -> dt = {dt} s...")
        X, Y, T_num, iteracoes = resolver_conducao(dx, dy, dt, beta)
        T_ana = solucao_analitica(X, Y, T_FINAL)
        erro_rms = calcular_rms(T_num, T_ana)
        
        dt_str = str(dt).replace('.', ',')
        rms_str = f"{erro_rms:.5f}".replace('.', ',')
        
        dados_tabela.append({
            "Formulação": nome,
            "dt [s]": dt_str,
            "Iterações Totais": iteracoes,
            "Erro RMS [K]": rms_str
        })

df = pd.DataFrame(dados_tabela)

print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.axis('tight')

tabela = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1.2, 1.5)

for (row, col), cell in tabela.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4C72B0')

plt.title("Influência do Passo de Tempo nas Formulações Implícitas", pad=20)
plt.tight_layout()

plt.show()