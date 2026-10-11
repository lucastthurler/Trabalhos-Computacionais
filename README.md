# 🚀 Trabalhos Computacionais — UERJ

Repositório dedicado aos códigos-fonte, simulações e implementações desenvolvidas para os trabalhos computacionais da graduação em Engenharia de Computação na Universidade do Estado do Rio de Janeiro (UERJ).

---

## 🛠️ Tecnologias Utilizadas

- Linguagem: Python 3.x
- Bibliotecas Científicas: NumPy, Matplotlib

---

## 📁 Estrutura do Repositório

├── T1/
│   ├── analises_completas.py
│   ├── estabilidade1.py
│   ├── estabilidade2.py
│   ├── jacobi_erro.py
│   ├── periodicidade.py
│   ├── poncaire.py
│   └── trajetória.py
├── T2/
│   ├── analise_de_perfis.py
│   ├── analise_explicito.py
│   ├── analise_implicito.py
│   ├── analise_tempos.py
│   ├── parametros_e_metodos.py
│   └── simulacao_chapa.py
└── README.md

---

## 🛰️ Trabalho 1 (T1): Mecânica Orbital e Dinâmica de Sistemas

Focado na modelação computacional e análise de sistemas não lineares aplicados à mecânica celeste.

- Problema Restrito de Três Corpos (PRTC): Simulação de trajetórias orbitais e comportamento dinâmico nas proximidades dos pontos de Lagrange ($L_1$ a $L_5$).
- Integração Numérica: Implementação de algoritmos de alta ordem, como o método Runge-Kutta de 4ª ordem (RK4).
- Espaço de Fase: Construção de secções de Poincaré, análise de periodicidade e validação da conservação da constante de Jacobi.

---

## 🔬 Trabalho 2 (T2): Condução de Calor Transiente 2D

Focado na resolução numérica de Equações Diferenciais Parciais (EDPs) parabólicas aplicadas à transferência de calor bidimensional numa chapa plana.

- Método das Diferenças Finitas (MDF): Discretização espacial de malha cartesiana uniforme.
- Esquemas de Avanço Temporal: Implementação comparativa da formulação geral em $\beta$:
  - Esquema Explícito ($\beta = 0$): Condicionalmente estável, governado pelo número de Fourier ($s \le 0,25$).
  - Esquema Totalmente Implícito ($\beta = 1$).
  - Esquema de Crank-Nicolson ($\beta = 1/2$).
- Análise de Desempenho: Avaliação de perfis térmicos, refinamento de malha e tempos de processamento computacional.

---

## ⚙️ Como Executar os Projetos

Certifique-se de ter o Python instalado na sua máquina junto com as bibliotecas necessárias (`numpy` e `matplotlib`).

1. Clone este repositório:
   git clone https://github.com/lucastthurler/Trabalhos-Computacionais.git

2. Navegue até a pasta do trabalho desejado (ex: T2):
   cd Trabalhos-Computacionais/T2

3. Execute o script correspondente:
   python simulacao_chapa.py

---

## 👨‍💻 Autor

- Lucas Thurler Pecles — Engenharia de Computação | UERJ  
- GitHub: https://github.com/lucastthurler
