from parametros_e_metodos import resolver_conducao, solucao_analitica, calcular_rms, ALPHA, T_FINAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 0.05
dx = dy = h
beta = 0.0

s_valores = [0.125, 0.25, 0.5]
dados_tabela = []

for s in s_valores:
    dt = round(s * (h**2) / ALPHA, 5)
    print(f"Calculando para s = {s} (dt = {dt} s)...")
    
    try:
        X, Y, T_num, iteracoes = resolver_conducao(dx, dy, dt, beta)
        T_ana = solucao_analitica(X, Y, T_FINAL)
        
        if np.any(np.isnan(T_num)) or np.max(T_num) > 105.0 or np.min(T_num) < -5.0:
             erro_rms = calcular_rms(T_num, T_ana)
             status = "INSTÁVEL"
        else:
             erro_rms = calcular_rms(T_num, T_ana)
             status = "ESTÁVEL"
             
        s_str = str(s).replace('.', ',')
        dt_str = f"{dt:.5f}".replace('.', ',')
        rms_str = f"{erro_rms:.5f}".replace('.', ',')
        
        dados_tabela.append({
            "Valor de s": s_str,
            "dt [s]": dt_str,
            "Status": status,
            "Erro RMS [K]": rms_str
        })
        
    except OverflowError:
        s_str = str(s).replace('.', ',')
        dt_str = f"{dt:.5f}".replace('.', ',')
        dados_tabela.append({
            "Valor de s": s_str,
            "dt [s]": dt_str,
            "Status": "INSTÁVEL",
            "Erro RMS [K]": "Falha (Overflow)"
        })

df = pd.DataFrame(dados_tabela)

print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3))
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

plt.title("Resultados da Formulação Explícita (Beta = 0)", pad=20)
plt.tight_layout()

plt.show()