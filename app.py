# app.py
import streamlit as st
import pandas as pd
import matplotlib

# Importa as funções dos nossos módulos
from src.ai_analysis import initialize_openai_client, generate_analysis
from src.plotting import create_plot

# --- Configuração Inicial ---
# O Streamlit lerá o .streamlit/config.toml automaticamente
matplotlib.use('Agg')
st.set_page_config(
    page_title="InsightXpress",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Inicialização do App ---
client = initialize_openai_client()

# Inicialização do estado da sessão (sem a chave 'theme')
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'graph_count' not in st.session_state:
    st.session_state.graph_count = 0
if 'follow_up_count' not in st.session_state:
    st.session_state.follow_up_count = 0

MAX_GRAPHS = 2
MAX_FOLLOW_UPS = 2

# --- Interface do Usuário ---
st.title("💡 InsightXpress")
st.markdown("""
**Como usar:**
1. Faça upload de uma planilha Excel (`.xlsx`, `.xls`) com seus dados.
2. Descreva o contexto ou objetivo da análise.
3. Clique em **"Analisar Dados"** e veja a mágica acontecer!
""")
st.info("⚠️ Use apenas dados fictícios. A análise é limitada às primeiras 20 linhas da planilha para fins de demonstração.", icon="ℹ️")

# --- Barra Lateral (Sidebar) para Upload e Controles ---
with st.sidebar:
    st.header("1. Configurações")
    uploaded_file = st.file_uploader(
        "Selecione a planilha Excel",
        type=['xlsx', 'xls'],
        help="O arquivo não pode exceder 2MB."
    )

    # --- INÍCIO DA MODIFICAÇÃO ---
    MAX_CHARS_CONTEXT = 300 # NOVO: Defina o limite de caracteres
    context_input = st.text_area(
        "2. Descreva o contexto da análise",
        placeholder="Ex: Quero entender o padrão de vendas por região e produto.",
        height=100,
        max_chars=MAX_CHARS_CONTEXT # NOVO: Adiciona o limite
    )
    # NOVO: Exibe o contador de caracteres
    st.caption(f"Caracteres: {len(context_input)}/{MAX_CHARS_CONTEXT}")
    # --- FIM DA MODIFICAÇÃO ---

    analyze_button = st.button("Analisar Dados", type="primary", use_container_width=True)

    st.divider()
    st.info("Para alterar o tema, use o menu (☰) > Settings.")

    # --- Rodapé (Footer) ---
    footer_html = """
    <div style="text-align: center; padding-top: 20px; color: grey; font-size: 14px;">
        <p>Feito por <a href="https://www.linkedin.com/in/gustavo-castro-06668b231/" target="_blank" style="color: grey; text-decoration: none;">Gustavo Rabutske</a> | Projeto de Portfólio</p>
    </div>
    """

    # Adiciona o rodapé na barra lateral
    st.sidebar.markdown(footer_html, unsafe_allow_html=True)


# --- Lógica Principal e Exibição de Resultados (Nenhuma mudança aqui) ---

# Lógica de carregamento de arquivo
if uploaded_file:
    try:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("Arquivo muito grande. O limite é de 2MB.")
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.context = context_input
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")
        st.session_state.df = None

# Lógica do botão de análise
if analyze_button:
    if st.session_state.df is None or st.session_state.context.strip() == "":
        st.warning("Por favor, carregue uma planilha e descreva o contexto antes de analisar.")
    elif client is None:
        st.error("Cliente da OpenAI não inicializado. Verifique a chave da API.")
    else:
        st.session_state.analysis_history = []
        st.session_state.graph_count = 0
        st.session_state.follow_up_count = 0
        with st.spinner("A IA está analisando seus dados... Por favor, aguarde."):
            try:
                initial_explanation = generate_analysis(client, st.session_state.df, st.session_state.context)
                st.session_state.analysis_history.append({"type": "analysis", "content": initial_explanation})
                st.rerun() # Adicionado para garantir a atualização da UI
            except (ValueError, RuntimeError) as e:
                st.error(f"Erro na Análise: {e}")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")

# Exibição do histórico de resultados
if st.session_state.analysis_history:
    st.header("Resultados da Análise")
    for item in st.session_state.analysis_history:
        if item['type'] == 'analysis':
            st.markdown(item['content'])
        elif item['type'] == 'graph':
            st.pyplot(item['content'])
        elif item['type'] == 'follow_up':
            with st.chat_message("user"):
                st.write(item['question'])
            with st.chat_message("assistant"):
                st.write(item['answer'])
    st.divider()

    # Seção de Acompanhamento
    if st.session_state.df is not None:
        graph_limit_reached = st.session_state.graph_count >= MAX_GRAPHS
        with st.expander("Gerar Novo Gráfico", expanded=False):
            if graph_limit_reached:
                st.warning(f"Você atingiu o limite de {MAX_GRAPHS} gráficos por análise.")
            else:
                # O resto do código do formulário de gráfico...
                st.markdown(f"Você pode gerar mais **{MAX_GRAPHS - st.session_state.graph_count}** gráfico(s).")
                with st.form("graph_form"):
                    columns = list(st.session_state.df.columns)
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox("Tipo de Gráfico", ["Automático", "Barras", "Linha", "Dispersão", "Pizza"], key="chart_type")
                        col_x = st.selectbox("Eixo X", columns, index=0, key="col_x")
                    with col2:
                        col_y = st.selectbox("Eixo Y", columns, index=min(1, len(columns)-1), key="col_y")
                    submitted = st.form_submit_button("Gerar Gráfico", use_container_width=True)
                    if submitted:
                        with st.spinner("Criando gráfico..."):
                            try:
                                fig = create_plot(st.session_state.df, chart_type, col_x, col_y)
                                st.session_state.analysis_history.append({"type": "graph", "content": fig})
                                st.session_state.graph_count += 1
                                st.rerun()
                            except (ValueError, RuntimeError) as e:
                                st.error(f"Erro ao gerar gráfico: {e}")

        st.subheader("Fazer uma pergunta de acompanhamento")
        follow_up_limit_reached = st.session_state.follow_up_count >= MAX_FOLLOW_UPS
        if follow_up_limit_reached:
            st.warning(f"Você atingiu o limite de {MAX_FOLLOW_UPS} perguntas por análise.")
        else:
            # O resto do código do formulário de pergunta...
            st.markdown(f"Você pode fazer mais **{MAX_FOLLOW_UPS - st.session_state.follow_up_count}** pergunta(s).")
            
            
            with st.form("follow_up_form", clear_on_submit=True):
                MAX_CHARS_QUESTION = 250 
                question = st.text_input(
                    "Sua pergunta sobre os dados:", 
                    key="follow_up_question", 
                    disabled=follow_up_limit_reached,
                    max_chars=MAX_CHARS_QUESTION 
                )
                # NOVO: Exibe o contador de caracteres
                st.caption(f"Caracteres: {len(question)}/{MAX_CHARS_QUESTION}")

                submitted = st.form_submit_button("Enviar Pergunta")
                if submitted and question:
                    with st.spinner("A IA está pensando..."):
                        try:
                            answer = generate_analysis(client, st.session_state.df, st.session_state.context, question=question)
                            st.session_state.analysis_history.append({"type": "follow_up", "question": question, "answer": answer})
                            st.session_state.follow_up_count += 1
                            st.rerun()
                        except (ValueError, RuntimeError) as e:
                            st.error(f"Erro na Pergunta: {e}")

