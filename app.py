import streamlit as st
import pandas as pd
import matplotlib

# Importa as funções dos nossos módulos
from src.ai_analysis import initialize_groq_client, generate_analysis
from src.plotting import create_plot

# --- Configuração Inicial ---
matplotlib.use('Agg')
st.set_page_config(
    page_title="Análise com IA",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Inicialização do App ---
client = initialize_groq_client()

# Inicialização do estado da sessão
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
st.title("💡 Análise de dados com IA")
st.markdown("""
**Como usar:**
1. Faça upload de uma planilha Excel (`.xlsx`, `.xls`) com seus dados.
2. Descreva o contexto ou objetivo da análise.
3. Clique em **"Analisar Dados"** e veja a mágica acontecer com a velocidade da Groq!
""")

# Estrutura de abas para organizar a interface
tab_analise, tab_sobre = st.tabs(["Análise de Dados", "Sobre o Projeto"])

# Conteúdo da aba "Sobre o Projeto"
with tab_sobre:
    st.header("Um Projeto de Portfólio")
    st.markdown("""
        Este projeto foi desenvolvido como um projeto de demonstração para o meu portfólio. 
        Ele exemplifica a construção de uma aplicação web interativa, ponta a ponta, focada em análise de dados e impulsionada por modelos de linguagem de última geração.
        **Esta aplicação não deve ser usada para análise de dados reais nem para tomada de decisões.**
    """)

    st.subheader("Tecnologias Utilizadas")
    st.markdown("""
        - **Linguagem:** Python
        - **Inteligência Artificial:**
            - **Groq API:** Para fornecer inferência em tempo real com altíssima velocidade.
            - **Modelo:** Llama 3 (um dos mais avançados modelos de linguagem abertos).
        - **Frontend e Interface:**
            - **Streamlit:** Para a criação da interface web interativa de forma ágil.
        - **Manipulação de Dados e Visualização:**
            - **Pandas:** Para a leitura e manipulação eficiente das planilhas.
            - **Matplotlib:** Para a geração dos gráficos e visualizações.
        - **Ambiente e Dependências:**
            - **dotenv:** Para o gerenciamento seguro de chaves de API.
    """)
    st.info("O código-fonte deste projeto está disponível no meu GitHub. Visite meu LinkedIn para mais detalhes!", icon="🔗")


# Toda a lógica de exibição de resultados agora fica dentro da aba "Análise de Dados"
with tab_analise:
    st.info("⚠️ Use apenas dados fictícios. A análise é limitada às primeiras 50 linhas da planilha.", icon="ℹ️")

    # --- Barra Lateral (Sidebar) para Upload e Controles ---
    with st.sidebar:
        st.header("1. Configurações")
        uploaded_file = st.file_uploader(
            "Selecione a planilha Excel",
            type=['xlsx', 'xls'],
            help="O arquivo não pode exceder 2MB."
        )
        
        MAX_CHARS_CONTEXT = 300
        context_input = st.text_area(
            "2. Descreva o contexto da análise",
            placeholder="Ex: Quero entender o padrão de vendas por região e produto.",
            height=100,
            max_chars=MAX_CHARS_CONTEXT,
            key="context_input"
        )
        # LINHA REMOVIDA: A linha abaixo criava um contador de caracteres duplicado.
        # st.caption(f"Caracteres: {len(st.session_state.get('context_input', ''))}/{MAX_CHARS_CONTEXT}")

        analyze_button = st.button("Analisar Dados", type="primary", use_container_width=True)

        st.divider()
        st.info("Para alterar o tema, use o menu (☰) > Settings.")

        footer_html = """
        <div style="text-align: center; padding-top: 20px; color: grey; font-size: 14px;">
            <p>Feito por <a href="https://www.linkedin.com/in/gustavo-castro-06668b231/" target="_blank" style="color: grey; text-decoration: none;">Gustavo Rabutske</a> | Projeto de Portfólio</p>
        </div>
        """
        st.sidebar.markdown(footer_html, unsafe_allow_html=True)


    # --- Lógica Principal e Exibição de Resultados ---

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
            st.error("Cliente da Groq não inicializado. Verifique a chave da API no arquivo .env.")
        else:
            st.session_state.analysis_history = []
            st.session_state.graph_count = 0
            st.session_state.follow_up_count = 0
            with st.spinner("A IA da Groq está analisando seus dados... Por favor, aguarde."):
                try:
                    initial_explanation = generate_analysis(client, st.session_state.df, st.session_state.context)
                    st.session_state.analysis_history.append({"type": "analysis", "content": initial_explanation})
                    st.rerun() 
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
                st.markdown(f"Você pode fazer mais **{MAX_FOLLOW_UPS - st.session_state.follow_up_count}** pergunta(s).")
                
                with st.form("follow_up_form", clear_on_submit=True):
                    MAX_CHARS_QUESTION = 250 
                    question = st.text_input(
                        "Sua pergunta sobre os dados:", 
                        key="follow_up_question",
                        disabled=follow_up_limit_reached,
                        max_chars=MAX_CHARS_QUESTION 
                    )

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