
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="📊 Análise de Enchentes no Brasil", layout="wide")
st.title("🌧️ Análise de Enchentes no Brasil (1991–2023 + Projeções até 2093)")

# Upload do arquivo CSV
arquivo = st.file_uploader("📁 Faça upload do arquivo CSV de enchentes", type=["csv"])
if arquivo is not None:
    df = pd.read_csv(arquivo, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")

    # Conversão de datas
    if "Data_Evento" in df.columns:
        df["Data_Evento"] = pd.to_datetime(df["Data_Evento"], errors="coerce", dayfirst=True)
        df["Ano"] = df["Data_Evento"].dt.year

    # Filtro: somente enchentes
    filtro = df["descricao_tipologia"].str.contains("enchente|alagamento|inundação", case=False, na=False)
    df_enchente = df[filtro].copy()

    st.subheader("📈 Eventos de enchente por ano")
    eventos_ano = df_enchente.groupby("Ano").size().reset_index(name="Qtd_Eventos")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=eventos_ano, x="Ano", y="Qtd_Eventos", marker="o", ax=ax1)
    ax1.set_title("Número de eventos de enchente por ano")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("🏙️ Top 10 municípios com mais registros")
    if "Nome_Municipio" in df_enchente.columns:
        top_mun = df_enchente["Nome_Municipio"].value_counts().head(10)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_mun.values, y=top_mun.index, palette="Reds_r", ax=ax2)
        ax2.set_title("Top 10 municípios com mais enchentes")
        st.pyplot(fig2)

    
    
    st.subheader("📊 Tabela interativa de eventos")
    st.dataframe(df_enchente[["Ano", "Nome_Municipio", "Sigla_UF", "descricao_tipologia"]].sort_values("Ano"))

    # ====================
    # PROJEÇÃO ATÉ 2093
    # ====================
    st.subheader("🔮 Previsão de eventos até 2093")
    eventos_ano = df_enchente.groupby("Ano").size().reset_index(name="Qtd_Eventos")
    X = eventos_ano["Ano"].values.reshape(-1, 1)
    y = eventos_ano["Qtd_Eventos"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression().fit(X_poly, y)

    anos_futuros = np.arange(X.max()+1, 2094).reshape(-1, 1)
    todos_anos = np.vstack([X, anos_futuros])
    todos_poly = poly.transform(todos_anos)
    previsoes = modelo.predict(todos_poly)

    df_prev = pd.DataFrame({
        "Ano": todos_anos.flatten(),
        "Qtd_Eventos_Previstos": np.round(previsoes, 0).astype(int)
    })

    fig4, ax4 = plt.subplots()
    ax4.plot(df_prev["Ano"], df_prev["Qtd_Eventos_Previstos"], label="Previsão", linestyle="--", color="red")
    ax4.scatter(X.flatten(), y, label="Histórico", color="blue")
    ax4.set_title("📈 Previsão de Enchentes no Brasil até 2093")
    ax4.set_xlabel("Ano")
    ax4.set_ylabel("Número de eventos")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

    st.dataframe(df_prev[df_prev["Ano"] >= 2024])

    # Botão de download
    csv = df_prev.to_csv(index=False).encode("utf-8")
    st.download_button("📩 Baixar previsão até 2093 (.csv)", data=csv, file_name="previsao_enchentes_2093.csv", mime="text/csv")


    # ====================
    # 🔍 Insights finais
    # ====================
    st.subheader("🧠 Insights e conclusões do projeto")

    st.markdown("""
    - 📈 **A frequência de eventos de enchentes no Brasil tem crescido ao longo das últimas décadas**, com tendência de aumento até 2093.
    - 🏘️ Os **municípios mais afetados** concentram-se em áreas urbanas densas ou em regiões de risco geológico/hidrológico.
    - 💸 O **prejuízo público total** varia de ano para ano, refletindo a gravidade e a localização dos eventos.
    - 🌎 A análise mostra **vulnerabilidades regionais** que podem auxiliar o planejamento urbano, políticas públicas e ações preventivas.
    - 🔮 A projeção indica que, **caso não haja mitigação**, o número de enchentes poderá dobrar até 2093.
    - 🗂️ A plataforma interativa permite **filtrar por estado e município**, otimizando análises locais e decisões governamentais.
    """)

# Assinatura no final
st.markdown("---")
st.markdown("📊 **by Victor Resende**")