import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import google.generativeai as genai
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(
    page_title="Analisador Laboratorial Avançado",
    page_icon="🔬",
    layout="wide"
)

# --- Funções Auxiliares ---

@st.cache_data
def load_data(uploaded_file):
    """Carrega dados de Excel ou CSV."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Tenta converter colunas com 'data' no nome para datetime
        for col in df.columns:
            if 'data' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

def safe_plot_bar(df, x, y, title):
    """Cria um gráfico de barras com ajuste automático de eixo Y para evitar cortes."""
    if df.empty or y not in df.columns:
        st.warning("Não há dados para plotar.")
        return

    fig = px.bar(df, x=x, y=y, title=title, text=y)
    
    # Adiciona 15% de "folga" no topo do eixo Y para o texto não cortar
    y_max = df[y].max()
    fig.update_layout(
        yaxis_range=[0, y_max * 1.15],
        xaxis={'categoryorder':'total descending'}
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


def get_gemini_query(api_key, user_prompt, columns):
    """Traduz linguagem natural em uma query Pandas usando a API Gemini."""
    if not api_key:
        st.error("API Key do Gemini não fornecida. Insira na barra lateral.")
        return None
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt de sistema detalhado para instruir a IA
        system_prompt = f"""
        Você é um assistente de análise de dados especialista em Pandas.
        Sua tarefa é traduzir o pedido do usuário em uma ÚNICA string de consulta (query) válida para `df.query()`.

        As colunas disponíveis no DataFrame são:
        {columns}

        REGRAS ESTRITAS:
        1.  Responda APENAS com a string da query. Sem explicações, sem "Aqui está:", sem nada.
        2.  Use os nomes exatos das colunas.
        3.  Se um nome de coluna tiver espaços, números ou caracteres especiais, use crases (ex: `Nome da Coluna`).
        4.  Para comparações de texto (string), SEMPRE converta para minúsculas usando `.str.lower()` e compare com texto em minúsculas.
            Exemplo CORRETO: `Motivo.str.lower() == 'hemólise'`
            Exemplo CORRETO: `\`Motivo da Recoleta\`.str.lower() == 'amostra coagulada'`
            Exemplo ERRADO: `Motivo == 'Hemólise'`
        5.  Use operadores Python padrão: `==`, `!=`, `>`, `<`, `>=`, `<=`, `and`, `or`, `not`.
        6.  Para checar valores nulos (NaN), use `.isna()` ou `.notna()`. Ex: `Coluna.isna()`

        Pedido do Usuário:
        "{user_prompt}"
        """
        
        response = model.generate_content(system_prompt)
        query_string = response.text.strip().replace("`", "") # Remove crases extras se a IA adicionar
        
        # Uma verificação final para garantir que é uma query
        if '(' not in query_string and '==' not in query_string and '>' not in query_string and '<' not in query_string:
             # Se for muito simples, pode não ter parênteses, mas ainda assim pode ser válida
             pass

        return query_string

    except Exception as e:
        st.error(f"Erro na API do Gemini: {e}")
        return None

# --- Funções de Análise (Reestruturadas) ---

def run_cutoff_analysis(df):
    """
    Nova 'Análise 1': Frequência baseada em cut-offs (Pos/Neg/Ind).
    """
    st.subheader("1. Análise por Cut-off (Positivo/Negativo/Indeterminado)")
    st.markdown("Defina valores de corte para uma coluna numérica para classificar resultados.")

    # --- Filtro de Data Opcional ---
    date_filter_on = st.toggle("Ativar filtro de data?", value=False)
    df_filtered = df.copy()
    
    if date_filter_on:
        date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        if len(date_cols) == 0:
            st.warning("Nenhuma coluna de data detectada. Para filtrar, certifique-se que o nome da coluna contenha 'data'.")
            date_filter_on = False
        else:
            col_date = st.selectbox("Selecione a coluna de data:", date_cols)
            
            # Converte para datetime (necessário para o slider)
            df_filtered[col_date] = pd.to_datetime(df_filtered[col_date])
            
            min_date = df_filtered[col_date].min().date()
            max_date = df_filtered[col_date].max().date()
            
            date_range = st.date_input(
                "Selecione o intervalo de datas:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered[col_date].dt.date >= start_date) & 
                    (df_filtered[col_date].dt.date <= end_date)
                ]
                st.success(f"Dados filtrados por data: {len(df_filtered)} linhas restantes.")
            else:
                st.info("Aguardando seleção de data final.")
                st.stop()
    
    st.markdown("---")

    # --- Definição dos Cut-offs ---
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.warning("Nenhuma coluna numérica encontrada nos dados filtrados.")
        st.stop()
        
    col_num = st.selectbox("Selecione a coluna numérica para análise:", numeric_cols)
    
    st.markdown("Defina os limites. Qualquer valor entre 'Negativo' e 'Positivo' será 'Indeterminado'.")
    
    c1, c2 = st.columns(2)
    with c1:
        # Ex: < 0.9
        cut_neg = st.number_input("Valor de corte para Negativo (ex: < 0.9)", value=0.9)
    with c2:
        # Ex: > 1.1
        cut_pos = st.number_input("Valor de corte para Positivo (ex: > 1.1)", value=1.1)
        
    if cut_neg >= cut_pos:
        st.error("O valor de corte 'Negativo' deve ser MENOR que o 'Positivo'.")
        st.stop()
        
    # Classificação
    conditions = [
        (df_filtered[col_num] < cut_neg),
        (df_filtered[col_num] > cut_pos),
        (df_filtered[col_num] >= cut_neg) & (df_filtered[col_num] <= cut_pos)
    ]
    choices = ['Negativo', 'Positivo', 'Indeterminado']
    
    df_filtered['Classificação'] = np.select(conditions, choices, default='Erro')
    
    # Cálculo de Frequência
    result_df = df_filtered['Classificação'].value_counts().reset_index()
    result_df.columns = ['Classificação', 'Frequência Absoluta']
    
    total = result_df['Frequência Absoluta'].sum()
    result_df['Frequência Relativa (%)'] = ((result_df['Frequência Absoluta'] / total) * 100).round(2)
    
    st.dataframe(result_df, use_container_width=True)
    
    # Plot
    safe_plot_bar(result_df, x='Classificação', y='Frequência Absoluta', title="Frequência de Resultados por Cut-off")


def run_stratification_analysis(df):
    """
    Nova 'Análise 2': Estratificação (bins) com agrupamento/correlação opcional.
    """
    st.subheader("2. Estratificação Quantitativa com Agrupamento")
    st.markdown("Estratifique uma coluna numérica (ex: MICROI) e, opcionalmente, agrupe por outra coluna (ex: Data, Mês, Ano).")
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.warning("Nenhuma coluna numérica encontrada.")
        st.stop()
        
    col_num = st.selectbox("Selecione a coluna numérica para estratificar (ex: MICROI):", numeric_cols)
    
    # --- Definição dos Bins ---
    st.markdown("**Defina os Intervalos (Bins)** (Ex: `0, 10, 20, 50`)")
    bins_str = st.text_input("Pontos de corte (separados por vírgula):")
    
    if not bins_str:
        st.info("Por favor, defina os intervalos (bins) para continuar.")
        st.stop()
        
    try:
        bin_edges = [float(x.strip()) for x in bins_str.split(',')]
        bin_edges = sorted(list(set(bin_edges)))
        
        if len(bin_edges) < 2:
            st.error("Defina pelo menos dois pontos de corte (ex: 0, 10).")
            st.stop()
            
        full_bin_edges = [-np.inf] + bin_edges + [np.inf]
        labels = []
        labels.append(f"< {bin_edges[0]}")
        for i in range(len(bin_edges) - 1):
            labels.append(f"{bin_edges[i]} a {bin_edges[i+1]}")
        labels.append(f"> {bin_edges[-1]}")

        df['Intervalo'] = pd.cut(df[col_num], bins=full_bin_edges, labels=labels, right=False)
        
    except Exception as e:
        st.error(f"Erro ao processar os intervalos: {e}")
        st.stop()

    # --- Agrupamento Opcional ---
    st.markdown("---")
    group_on = st.toggle("Adicionar coluna de agrupamento? (ex: por Data, Mês, Ano, Equipamento)", value=False)
    
    if not group_on:
        # Análise Simples (Original)
        st.markdown(f"**Frequência Simples para '{col_num}'**")
        result_df = df['Intervalo'].value_counts().sort_index().reset_index()
        result_df.columns = ['Intervalo', 'Frequência']
        st.dataframe(result_df, use_container_width=True)
        safe_plot_bar(result_df, x='Intervalo', y='Frequência', title=f"Estratificação de '{col_num}'")
    
    else:
        # Análise Agrupada
        col_group = st.selectbox("Selecione a coluna para agrupar:", [None] + list(df.columns))
        
        if col_group is None:
            st.info("Selecione uma coluna de agrupamento.")
            st.stop()
            
        df_analysis = df[[col_num, 'Intervalo', col_group]].copy()
        
        # Se for data, oferece agrupar por período
        if pd.api.types.is_datetime64_any_dtype(df_analysis[col_group]):
            period_type = st.radio(
                "Agrupar data por:",
                ['Dia', 'Mês', 'Ano'],
                horizontal=True
            )
            
            if period_type == 'Dia':
                df_analysis['Período'] = df_analysis[col_group].dt.to_period('D').astype(str)
            elif period_type == 'Mês':
                df_analysis['Período'] = df_analysis[col_group].dt.to_period('M').astype(str)
            else:
                df_analysis['Período'] = df_analysis[col_group].dt.to_period('Y').astype(str)
            
            group_col_final = 'Período'
        else:
            # Se não for data, apenas usa a coluna como está
            df_analysis[col_group] = df_analysis[col_group].astype(str)
            group_col_final = col_group

        # --- Plots Agrupados ---
        st.markdown(f"**Análise de '{col_num}' por '{group_col_final}'**")

        tab1, tab2, tab3, tab4 = st.tabs(["Frequência Relativa (%)", "Boxplot por Período", "Mediana por Período", "Tabela de Frequência"])

        with tab1:
            st.markdown("Frequência relativa de cada intervalo ao longo do período.")
            crosstab_df = pd.crosstab(df_analysis[group_col_final], df_analysis['Intervalo'])
            # Calcula % por linha (total do período)
            crosstab_pct = crosstab_df.apply(lambda x: (x / x.sum()) * 100, axis=1).reset_index()
            # Derrete (melts) para formato longo, ideal para Plotly
            df_plot_pct = crosstab_pct.melt(id_vars=group_col_final, var_name='Intervalo', value_name='Frequência Relativa (%)')
            
            fig = px.line(df_plot_pct, x=group_col_final, y='Frequência Relativa (%)', color='Intervalo', title="Frequência Relativa dos Intervalos ao Longo do Período", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("Distribuição dos valores (Boxplot) por período.")
            fig_box = px.box(df_analysis, x=group_col_final, y=col_num, title=f"Distribuição de '{col_num}' por '{group_col_final}'")
            st.plotly_chart(fig_box, use_container_width=True)

        with tab3:
            st.markdown("Mediana dos valores ao longo do período.")
            median_df = df_analysis.groupby(group_col_final)[col_num].median().reset_index()
            fig_median = px.line(median_df, x=group_col_final, y=col_num, title=f"Mediana de '{col_num}' por '{group_col_final}'", markers=True)
            fig_median.update_traces(name="Mediana")
            st.plotly_chart(fig_median, use_container_width=True)
            
        with tab4:
            st.markdown("Tabela de frequência absoluta (contagem).")
            crosstab_df_abs = pd.crosstab(df_analysis['Intervalo'], df_analysis[group_col_final])
            st.dataframe(crosstab_df_abs, use_container_width=True)


def run_crosstab_analysis(df):
    """
    Análise 3: Tabela de Co-ocorrência (Cruzamento).
    Agora com conversão para minúsculas.
    """
    st.subheader("3. Tabela de Co-ocorrência (Cruzamento)")
    st.markdown("Cruza duas variáveis categóricas. Comparações de texto serão convertidas para minúsculas.")
    
    col_options = [None] + list(df.columns)
    
    c1, c2 = st.columns(2)
    row_var = c1.selectbox("Selecione a Variável 1 (Linhas):", col_options, key='cb_row')
    col_var = c2.selectbox("Selecione a Variável 2 (Colunas):", col_options, key='cb_col')
        
    if not row_var or not col_var:
        st.info("Selecione duas variáveis.")
        st.stop()
        
    if row_var == col_var:
        st.warning("Selecione duas colunas diferentes.")
        st.stop()
        
    try:
        # Prepara os dados (converte para string e minúsculas se for texto)
        row_data = df[row_var]
        col_data = df[col_var]
        
        if pd.api.types.is_string_dtype(row_data):
            row_data = row_data.str.lower().fillna("N/A")
        if pd.api.types.is_object_dtype(row_data):
             row_data = row_data.astype(str).str.lower().fillna("N/A")
            
        if pd.api.types.is_string_dtype(col_data):
            col_data = col_data.str.lower().fillna("N/A")
        if pd.api.types.is_object_dtype(col_data):
            col_data = col_data.astype(str).str.lower().fillna("N/A")

        crosstab_df = pd.crosstab(row_data, col_data)
        
        st.dataframe(crosstab_df, use_container_width=True)
        fig = px.imshow(crosstab_df, text_auto=True, title=f"Mapa de Calor: {row_var} vs {col_var}", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao criar tabela cruzada: {e}")

def run_query_analysis(df, api_key):
    """
    Análise 4: Frequência por Query com IA (Linguagem Natural).
    """
    st.subheader("4. Frequência de Evento Específico (Query com IA)")
    st.markdown("Faça uma pergunta em linguagem natural sobre seus dados. A IA irá traduzir para uma query.")
    
    if not api_key:
        st.error("Por favor, insira sua API Key do Google Gemini na barra lateral para usar esta funcionalidade.")
        st.stop()
    
    st.info("""
    **Exemplos de Perguntas:**
    - "todos os pacientes com idade acima de 50 e motivo de recoleta hemólise"
    - "resultados positivos para a coluna 'Resultado' onde a 'Idade' é menor que 18"
    - "quantos exames de TSH estão acima de 4.5 ou abaixo de 0.4"
    """)
    
    user_prompt = st.text_area("Escreva sua pergunta em linguagem natural:")
    
    if not st.button("Traduzir Pergunta com IA") or not user_prompt:
        st.stop()
        
    with st.spinner("A IA está traduzindo sua pergunta..."):
        query_string = get_gemini_query(api_key, user_prompt, list(df.columns))
    
    if query_string:
        st.markdown("**Query Traduzida pela IA:**")
        st.code(query_string, language="python")
        
        # Armazena a query no estado da sessão para confirmação
        st.session_state.gemini_query = query_string
        
    if 'gemini_query' in st.session_state and st.session_state.gemini_query:
        if st.button("Confirmar e Executar Query"):
            with st.spinner("Executando query..."):
                try:
                    filtered_df = df.query(st.session_state.gemini_query)
                    count = len(filtered_df)
                    total = len(df)
                    percent = (count / total * 100) if total > 0 else 0
                    
                    st.subheader("Resultados da Consulta")
                    c1, c2 = st.columns(2)
                    c1.metric(label="Frequência do Evento (Contagem)", value=count)
                    c2.metric(label="Percentual do Total", value=f"{percent:.2f}%")
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erro ao executar a query: {e}")
                    st.error(f"Query problemática: {st.session_state.gemini_query}")
                
                # Limpa a query após a execução
                del st.session_state.gemini_query

# --- Interface Principal (Main App) ---

st.title("🔬 Analisador Laboratorial Avançado")
st.markdown("Carregue sua planilha e escolha uma das análises avançadas.")

# --- Sidebar (API Key e Upload) ---
st.sidebar.title("Configuração")
api_key = st.sidebar.text_input(
    "1. Cole sua API Key do Google Gemini:",
    type="password",
    help="Necessária APENAS para a 'Análise 4: Query com IA'."
)

uploaded_file = st.sidebar.file_uploader(
    "2. Carregue seu arquivo (Excel ou CSV)",
    type=["csv", "xls", "xlsx"]
)

# --- Lógica Principal ---
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    # Carrega dados se for um novo arquivo
    if st.session_state.df is None or uploaded_file.name != st.session_state.get('file_name'):
        with st.spinner("Carregando dados..."):
            df = load_data(uploaded_file)
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
                st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado!")
            else:
                st.session_state.df = None
                st.session_state.file_name = None
                st.sidebar.error("Falha ao carregar o arquivo.")
else:
    # Limpa o estado se o arquivo for removido
    st.session_state.df = None
    st.session_state.file_name = None

if st.session_state.df is None:
    st.info("Por favor, carregue um arquivo (Excel ou CSV) na barra lateral para começar.")
    st.stop()

# Se o DataFrame estiver carregado
df = st.session_state.df

with st.expander("Visualizar Dados Carregados (Primeiras 50 linhas)"):
    st.dataframe(df.head(50), use_container_width=True)

st.markdown("---")

# Seleção do Tipo de Análise (com novos nomes)
analysis_type = st.selectbox(
    "Selecione o Tipo de Análise:",
    [
        "Selecione...",
        "1. Análise por Cut-off (Pos/Neg/Ind)",
        "2. Estratificação com Agrupamento",
        "3. Tabela de Co-ocorrência (Cruzamento)",
        "4. Frequência por Query (Linguagem Natural)"
    ]
)

# Chama a função de análise correspondente
if analysis_type == "1. Análise por Cut-off (Pos/Neg/Ind)":
    run_cutoff_analysis(df)

elif analysis_type == "2. Estratificação com Agrupamento":
    run_stratification_analysis(df)

elif analysis_type == "3. Tabela de Co-ocorrência (Cruzamento)":
    run_crosstab_analysis(df)

elif analysis_type == "4. Frequência por Query (Linguagem Natural)":
    run_query_analysis(df, api_key)