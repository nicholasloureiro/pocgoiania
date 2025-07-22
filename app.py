import os
from datetime import datetime, timedelta
import sqlalchemy as sql
import re 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from plotly.subplots import make_subplots
from openai import OpenAI
import asyncio
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.agents import SQLDatabaseAgent

# Configuração da página
st.set_page_config(
    page_title="Análise de Fraude - Unimed",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para estilo melhorado
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 4px solid #3b82f6;
        padding-bottom: 1rem;
    }
    
    .tab-header {
        font-size: 2rem;
        font-weight: 600;
        color: #374151;
        margin: 1rem 0;
        text-align: center;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #ef4444;
        padding-left: 1rem;
    }
    
    .subsection-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #4b5563;
        margin: 1rem 0 0.5rem 0;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .alert-critical {
        background: #fef2f2;
        border: 2px solid #fecaca;
        color: #991b1b;
        padding: 1.2rem;
        border-radius: 0.7rem;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .alert-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .alert-medium {
        background: #fffbeb;
        border: 1px solid #fed7aa;
        color: #92400e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .insight-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        color: #0c4a6e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .story-box {
        background: #f9fafb;
        border-left: 4px solid #6366f1;
        padding: 1.2rem;
        margin: 1rem 0;
        font-style: italic;
    }
    
    .key-finding {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Configuração de conexão com banco
DB_CONFIG = {
    'host': os.getenv('host'),
    'database': os.getenv('database'), 
    'user': os.getenv('user'),
    'password': os.getenv('password'),
    'port': 5432
}
MODEL_LIST = ['gpt-4o-mini']



#@st.cache_data
def get_database_connection():
    """Conexão com PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar com o banco: {e}")
        return None

#@st.cache_data
def execute_query(query):
    """Executa query e retorna DataFrame"""
    conn = get_database_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao executar query: {e}")
        conn.close()
        return pd.DataFrame()

def get_overview_metrics():
    """Calcula métricas gerais para o overview"""
    # Queries para métricas principais
    query_total_atendimentos = """
    SELECT COUNT(*) as total FROM unimed.base_pep WHERE "DATA" IS NOT NULL
    """
    
    query_noshow_rate = """
    SELECT 
        ROUND((SUM(CASE WHEN "SITUAÇÃO" ILIKE '%não%' OR "SITUAÇÃO" ILIKE '%nao%' THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 1) as taxa_noshow
    FROM unimed.base_pep WHERE "SITUAÇÃO" IS NOT NULL
    """
    
    query_divergencias_count = """
    WITH sgu_formatted AS (
        SELECT "Beneficiário" as carteirinha, "Data Execução" as data_atendimento, "Valor Pagamento" as valor
        FROM unimed.comp10 WHERE "Tipo Lançamento" = 'PRODUÇÃO'
        UNION ALL
        SELECT "Beneficiário" as carteirinha, "Data Execução" as data_atendimento, "Valor Pagamento" as valor
        FROM unimed.comp11 WHERE "Tipo Lançamento" = 'PRODUÇÃO'
        UNION ALL
        SELECT "Beneficiário" as carteirinha, "Data Execução" as data_atendimento, "Valor Pagamento" as valor
        FROM unimed.comp12 WHERE "Tipo Lançamento" = 'PRODUÇÃO'
    ),
    pep_formatted AS (
        SELECT "CARTEIRA" as carteirinha, TO_DATE("DATA", 'DD/MM/YYYY') as data_atendimento
        FROM unimed.base_pep
    )
    SELECT COUNT(*) as divergencias, SUM(s.valor) as valor_total
    FROM sgu_formatted s
    LEFT JOIN pep_formatted p ON (s.carteirinha = p.carteirinha AND s.data_atendimento = p.data_atendimento)
    WHERE p.carteirinha IS NULL
    """
    
    query_conflitos_count = """
    SELECT COUNT(*) as conflitos
    FROM (
        SELECT "PROFISSIONAL ", TO_DATE("DATA", 'DD/MM/YYYY') as data, "ATENDIMENTO"
        FROM unimed.base_pep
        WHERE "SITUAÇÃO" = 'atendida'
        GROUP BY "PROFISSIONAL ", TO_DATE("DATA", 'DD/MM/YYYY'), "ATENDIMENTO"
        HAVING COUNT(*) > 1
    ) x
    """
    
    # Executa queries
    df_total = execute_query(query_total_atendimentos)
    df_noshow = execute_query(query_noshow_rate)
    df_divergencias = execute_query(query_divergencias_count)
    df_conflitos = execute_query(query_conflitos_count)
    
    return {
        'total_atendimentos': df_total['total'].iloc[0] if not df_total.empty else 0,
        'taxa_noshow': df_noshow['taxa_noshow'].iloc[0] if not df_noshow.empty else 0,
        'divergencias': df_divergencias['divergencias'].iloc[0] if not df_divergencias.empty else 0,
        'valor_divergencias': df_divergencias['valor_total'].iloc[0] if not df_divergencias.empty else 0,
        'conflitos': df_conflitos['conflitos'].iloc[0] if not df_conflitos.empty else 0
    }

def render_overview_tab():
    """Renderiza a aba com chat SQL inteligente usando OpenAI com melhor tratamento de erros"""
    st.markdown('<div class="tab-header">🤖 Chat Inteligente SQL</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Assistente IA para Análise de Dados:</strong> Faça perguntas em português sobre os dados 
    da Unimed.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for chat configuration
    with st.sidebar:
        st.markdown("### 🤖 Configuração do Chat")
        
        # OpenAI API Key
        if "OPENAI_API_KEY" not in st.session_state:
            st.session_state["OPENAI_API_KEY"] = ""
            
        api_key = st.text_input(
            "🔑 OpenAI API Key", 
            type="password", 
            value=st.session_state["OPENAI_API_KEY"],
            help="Sua chave da API OpenAI é necessária para o chat funcionar."
        )
        
        if api_key:
            st.session_state["OPENAI_API_KEY"] = api_key
            
            # Test API Key
            try:
                client = OpenAI(api_key=api_key)
                models = client.models.list()
                st.success("✅ API Key válida!")
                
                # Model selection
                model_option = st.selectbox(
                    "🧠 Modelo OpenAI",
                    MODEL_LIST,
                    index=0
                )
                st.session_state["model_option"] = model_option
                
                # Clear messages button in sidebar
                st.markdown("---")
                st.markdown("### Controles do Chat")
                if st.button(" Limpar Conversas", type="secondary", use_container_width=True):
                    if "openai_chat_messages" in st.session_state:
                        del st.session_state["openai_chat_messages"]
                    if "openai_dataframes" in st.session_state:
                        st.session_state.openai_dataframes = []
                    st.rerun()
                
                # Chat statistics
                if "openai_dataframes" in st.session_state:
                    st.metric("📊 Consultas Realizadas", len(st.session_state.openai_dataframes))
                
            except Exception as e:
                st.error(f"❌ API Key inválida: {e}")
                return
        else:
            st.info("👆 Insira sua OpenAI API Key para continuar.")
            return
    
    # Initialize chat components
    if st.session_state.get("OPENAI_API_KEY"):
        # Set up OpenAI client
        client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
        
        # Get database connection with better error handling
        def get_db_connection():
            """Cria conexão com o banco com tratamento robusto de erro"""
            try:
                import sqlalchemy as sql
                # Build connection string from your DB_CONFIG
                connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
                
                # Create engine with better configuration
                engine = sql.create_engine(
                    connection_string,
                    pool_size=5,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    echo=False
                )
                
                # Test connection
                with engine.connect() as conn:
                    conn.execute(sql.text("SELECT 1"))
                
                return engine
            except Exception as e:
                raise Exception(f"Falha na conexão com banco de dados: {str(e)}")
        
        # Get database schema information with improved error handling
        try:
            engine = get_db_connection()
            import sqlalchemy as sql
            
            # Get table schema information
            inspector = sql.inspect(engine)
            schemas_info = {}
            
            # Get tables from unimed schema
            tables = inspector.get_table_names(schema='unimed')
            for table in tables[:5]:  # Limit to first 5 tables for context
                try:
                    columns = inspector.get_columns(table, schema='unimed')
                    # Wrap column names in double quotes and filter out PACIENTE ID
                    wrapped_columns = [f'"{col["name"]}"' for col in columns[:10] if col["name"] != "PACIENTE ID"]
                    schemas_info[f'unimed."{table}"'] = wrapped_columns
                except Exception as e:
                    st.warning(f"⚠️ Não foi possível carregar esquema da tabela {table}: {e}")
                    continue
            
        except Exception as e:
            st.error(f"❌ Erro ao conectar com o banco de dados: {e}")
            return
        
        # Set up memory
        msgs = StreamlitChatMessageHistory(key="openai_chat_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("Olá! Sou seu assistente para análise de dados da Unimed. Posso ajudá-lo a criar consultas SQL e analisar padrões de fraude. Como posso ajudar?")
        
        # Initialize dataframe storage
        if "openai_dataframes" not in st.session_state:
            st.session_state.openai_dataframes = []
        
        # Function to validate and clean SQL
        def validate_and_clean_sql(sql_query):
            """Valida e limpa query SQL antes da execução"""
            try:
                # Remove markdown formatting
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'^```\s*', '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*```$', '', sql_query)
                sql_query = sql_query.strip()
                
                # Basic SQL injection prevention
                dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
                sql_upper = sql_query.upper()
                for keyword in dangerous_keywords:
                    if keyword in sql_upper and not sql_upper.startswith('SELECT'):
                        raise ValueError(f"Operação não permitida: {keyword}")
                
                # Ensure query starts with SELECT
                if not sql_upper.strip().startswith('SELECT'):
                    raise ValueError("Apenas consultas SELECT são permitidas")
                
                # Add LIMIT if not present and query is potentially large
                if 'LIMIT' not in sql_upper and 'COUNT(' not in sql_upper:
                    sql_query += ' LIMIT 1000'
                
                return sql_query
                
            except Exception as e:
                raise ValueError(f"SQL inválido: {str(e)}")
        
        # Function to generate SQL with OpenAI
        def generate_sql_with_openai(user_question, schema_info):
            schema_text = "\n".join([f"Tabela {table}: {', '.join(columns)}" for table, columns in schema_info.items()])
            
            system_prompt = f"""Você é um especialista em SQL para análise de dados de saúde da Unimed. Baseado no esquema de banco de dados abaixo, gere consultas SQL seguindo os padrões estabelecidos.

ESQUEMA DO BANCO DE DADOS: {schema_text}

ESTRUTURA DO BANCO:
- unimed.base_pep: dados de agendamentos/atendimentos (PEP)
  * "CARTEIRA": identificador do beneficiário
  * "PACIENTE": nome do paciente  
  * "DATA": data do agendamento (formato DD/MM/YYYY)
  * "ATENDIMENTO": horário do atendimento
  * "SITUAÇÃO": status ('atendida', 'cancelada', 'nao compareceu')
  * "PROFISSIONAL ": nome do profissional (com espaço no final)
  * "ESPECIALIDADE": especialidade médica
  
- unimed.comp10, unimed.comp11, unimed.comp12: dados SGU de faturamento
  * "Beneficiário": carteirinha do beneficiário
  * "Nome Beneficiário": nome do paciente
  * "Data Execução": data da execução
  * "Hora Execução": hora da execução
  * "Item Desc": descrição do procedimento
  * "Nome Prestador": nome do prestador
  * "Valor Pagamento": valor pago
  * "Tipo Lançamento": filtrar por 'PRODUÇÃO'

REGRAS CRÍTICAS DE FORMATAÇÃO:
1. SEMPRE retorne APENAS código SQL puro, sem explicações, sem markdown, sem comentários extras
2. Use aspas duplas para TODAS as colunas: "CARTEIRA", "DATA", "SITUAÇÃO"
3. Use aspas duplas para nomes de tabelas: unimed."base_pep"
4. Para datas, use: TO_DATE("DATA", 'DD/MM/YYYY')
5. Para strings, use ILIKE para busca case-insensitive
6. SEMPRE use TRIM() em campos que podem ter espaços
7. Use LIMIT 1000 para evitar resultados muito grandes
8. Para SGU, sempre filtre: "Tipo Lançamento" = 'PRODUÇÃO'

EXEMPLO DE RESPOSTA (apenas SQL):
SELECT "CARTEIRA", "PACIENTE", "DATA", "SITUAÇÃO" 
FROM unimed."base_pep" 
WHERE "SITUAÇÃO" = 'não atendida' 
ORDER BY TO_DATE("DATA", 'DD/MM/YYYY') DESC 
LIMIT 1000;

INSTRUÇÕES FINAIS:
- Responda APENAS com código SQL válido
- Não inclua explicações, comentários ou formatação markdown
- Garanta que a query seja otimizada e segura
- Trate casos edge (valores nulos, dados inconsistentes)
"""

            try:
                response = client.chat.completions.create(
                    model=st.session_state.get("model_option", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                raise Exception(f"Erro ao gerar SQL com OpenAI: {e}")
        
        # Function to execute SQL and get results with improved error handling
        def execute_sql_query(sql_query):
            """Executa query SQL com tratamento robusto de erros"""
            try:
                # Validate and clean SQL first
                clean_sql = validate_and_clean_sql(sql_query)
                
                # Execute query with better error handling
                with engine.begin() as conn:
                    # Use text() for raw SQL and fetch results properly
                    result = conn.execute(sql.text(clean_sql))
                    
                    # Convert to DataFrame manually to avoid SQLAlchemy issues
                    columns = result.keys()
                    rows = result.fetchall()
                    
                    # Create DataFrame from rows and columns
                    df = pd.DataFrame(rows, columns=columns)
                    
                    # Basic data type optimization
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Try to convert to numeric if possible
                            try:
                                pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                    
                return df, clean_sql
                
            except ValueError as ve:
                # SQL validation errors
                raise Exception(f"Query inválida: {str(ve)}")
            except sql.exc.SQLAlchemyError as se:
                # Database specific errors
                raise Exception(f"Erro no banco de dados: {str(se)}")
            except Exception as e:
                # Generic errors with more context
                error_msg = str(e)
                if "immutabledict" in error_msg:
                    raise Exception("Erro de compatibilidade SQLAlchemy. Tente uma consulta mais simples.")
                elif "permission denied" in error_msg.lower():
                    raise Exception("Sem permissão para acessar os dados solicitados.")
                elif "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                    raise Exception("Tabela não encontrada. Verifique o nome da tabela.")
                else:
                    raise Exception(f"Erro na execução: {error_msg}")
        
        # Function to generate insights about the results
        def generate_insights_with_openai(question, sql_query, df):
            if df.empty:
                return "📊 Nenhum resultado encontrado para esta consulta. Tente ajustar os critérios de busca."
            
            # Get basic info about the dataframe
            try:
                # Safely get sample data
                sample_data = df.head(3).to_dict('records') if len(df) > 0 else []
                df_info = f"""
Dados retornados:
- {len(df)} registros
- {len(df.columns)} colunas: {', '.join(df.columns.tolist())}
- Sample: {sample_data}
"""
                
                prompt = f"""Baseado na pergunta do usuário e nos resultados da consulta SQL, forneça insights breves e relevantes.

PERGUNTA: {question}

SQL EXECUTADO: {sql_query}

DADOS RETORNADOS:
{df_info}

Forneça 2-3 insights principais em português, de forma concisa e relevante para análise de fraude em saúde. Use emojis para tornar mais visual."""

                response = client.chat.completions.create(
                    model=st.session_state.get("model_option", "gpt-4o-mini"),
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"📈 Análise dos dados concluída com sucesso. ({len(df)} registros encontrados)"
        
        # Display chat history
        def display_chat_history():
            for i, msg in enumerate(msgs.messages):
                with st.chat_message(msg.type):
                    if "DATAFRAME_INDEX:" in msg.content:
                        df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                        if df_index < len(st.session_state.openai_dataframes):
                            df_info = st.session_state.openai_dataframes[df_index]
                            
                            # Enhanced dataframe display
                            st.markdown("### Resultado da Consulta")
                            
                            # Metrics row
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Registros", len(df_info['dataframe']))
                            with col2:
                                st.metric("Colunas", len(df_info['dataframe'].columns))
                            with col3:
                                try:
                                    memory_usage = df_info['dataframe'].memory_usage(deep=True).sum() / 1024
                                    st.metric("Tamanho", f"{memory_usage:.1f} KB")
                                except:
                                    st.metric("Tamanho", "N/A")
                            
                            # Dataframe with better styling
                            st.dataframe(
                                df_info['dataframe'], 
                                use_container_width=True,
                                height=min(400, len(df_info['dataframe']) * 35 + 100)
                            )
                            
                            # Show insights in an attractive box
                            if df_info.get('insights'):
                                st.markdown("### Insights")
                                st.info(df_info['insights'])
                            
                            # Show SQL query in expandable section with copy button
                            with st.expander("Ver SQL Gerado", expanded=False):
                                st.code(df_info['sql_query'], language='sql')
                                
                            # Download button
                            if not df_info['dataframe'].empty:
                                try:
                                    csv = df_info['dataframe'].to_csv(index=False)
                                    st.download_button(
                                        label="Baixar Resultado (CSV)",
                                        data=csv,
                                        file_name=f"gpt_consulta_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        key=f"download_{df_index}"
                                    )
                                except Exception as e:
                                    st.warning(f"⚠️ Erro ao preparar download: {e}")
                    else:
                        st.write(msg.content)
        
        # Main chat area with improved layout
        st.markdown("---")
        
        # Display chat history
        display_chat_history()
        
        # Chat input with suggested question handling
        suggested_question = st.session_state.get("suggested_question", "")
        if suggested_question:
            question = suggested_question
            st.session_state["suggested_question"] = ""
        else:
            question = st.chat_input("💬 Faça sua pergunta sobre os dados da Unimed...")
        
        if question:
            st.chat_message("human").write(question)
            msgs.add_user_message(question)
            with st.spinner("Estamos analisando sua pergunta..."):
                # Add user message
                
                
                # Process question with improved error handling
                try:
                    # Generate SQL with OpenAI
                    sql_query = generate_sql_with_openai(question, schemas_info)
                    
                    # Execute SQL
                    result_df, clean_sql = execute_sql_query(sql_query)
                    
                    # Generate insights
                    insights = generate_insights_with_openai(question, clean_sql, result_df)
                    
                    # Format response
                    response_text = f"""


**Pergunta:** {question}

**Resultado:** {len(result_df)} registro(s) encontrado(s)


"""
                    
                    # Store dataframe with SQL and insights
                    df_index = len(st.session_state.openai_dataframes)
                    st.session_state.openai_dataframes.append({
                        'dataframe': result_df,
                        'sql_query': clean_sql,
                        'question': question,
                        'insights': insights,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                    # Add messages
                    msgs.add_ai_message(response_text)
                    msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                    
                    # Auto-scroll to bottom
                    st.rerun()
                    
                except Exception as e:
                    # Enhanced error handling with specific suggestions
                    error_str = str(e)
                    
                    # Provide specific suggestions based on error type
                    if "inválida" in error_str or "SQL" in error_str:
                        suggestions = """
**💡 Dicas para corrigir a consulta:**
- ✅ Verifique se os nomes das tabelas estão corretos: `base_pep`, `comp10`, `comp11`, `comp12`
- ✅ Use aspas duplas nos campos: `"DATA"`, `"SITUAÇÃO"`, `"CARTEIRA"`
- ✅ Para datas, especifique o formato: "últimos 30 dias", "janeiro de 2024"
- ✅ Seja mais específico: "pacientes com mais de 5 cancelamentos"
"""
                    elif "conexão" in error_str.lower() or "banco" in error_str.lower():
                        suggestions = """
**🔧 Problema de conexão com banco:**
- 🔄 Tente novamente em alguns segundos
- 📞 Contate o administrador se o problema persistir
- 💾 Verifique se o banco de dados está disponível
"""
                    elif "permissão" in error_str.lower():
                        suggestions = """
**🔐 Problema de permissão:**
- 👤 Verifique suas credenciais de acesso
- 📋 Consulte apenas tabelas permitidas: `base_pep`, `comp10-12`
- 🚫 Operações de escrita não são permitidas
"""
                    else:
                        suggestions = """
**💡 Dicas gerais:**
- ✅ Tente uma pergunta mais simples primeiro
- ✅ Use exemplos das perguntas sugeridas abaixo
- ✅ Seja específico sobre campos e períodos
- 🔄 Reformule a pergunta de forma diferente
"""
                    
                    error_response = f"""
### ❌ Erro na Consulta

😔 Tive dificuldade para processar sua pergunta.

**Erro:** {error_str}

{suggestions}

**🔄 Exemplo de pergunta válida:** "Quantos agendamentos temos na base_pep?"
"""
                    msgs.add_ai_message(error_response)
                    st.chat_message("ai").write(error_response)
        
        # Enhanced suggestions section
      
    
    else:
        # Enhanced warning for missing API key
        st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(90deg, #ff6b6b, #ffa500); border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
            <h3>🔑 Configuração Necessária</h3>
            <p>Configure sua OpenAI API Key na barra lateral para usar o chat com GPT.</p>
            <p><small>A API Key é necessária para gerar consultas SQL automaticamente.</small></p>
        </div>
        """, unsafe_allow_html=True)
        
def render_divergencias_tab():
    """Renderiza a aba de divergências SGU x PEP"""
    st.markdown('<div class="tab-header">Divergências SGU x PEP</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>O Problema:</strong> O SGU (Sistema de Gestão Unimed) registra procedimentos faturados que não aparecem como atendidos no PEP (Prontuário Eletrônico do Paciente). Esse descompasso pode indicar:

            - Faturamento de procedimentos não realizados;

            - Falhas na integração entre os sistemas;

            - Inconsistências nos registros administrativoclinicos.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para divergências
    query_divergencias = """
    WITH sgu_formatted AS (
        SELECT 
            "Beneficiário" as carteirinha,
            "Nome Beneficiário" as nome_paciente,
            "Data Execução" as data_atendimento,
            "Hora Execução" as hora_atendimento,
            "Item Desc" as procedimento,
            "Nome Prestador" as prestador,
            "Valor Pagamento" as valor,
            'SGU-OUT' as origem
        FROM unimed.comp10
        WHERE "Tipo Lançamento" = 'PRODUÇÃO'
        
        UNION ALL
        
        SELECT 
            "Beneficiário" as carteirinha,
            "Nome Beneficiário" as nome_paciente,
            "Data Execução" as data_atendimento,
            "Hora Execução" as hora_atendimento,
            "Item Desc" as procedimento,
            "Nome Prestador" as prestador,
            "Valor Pagamento" as valor,
            'SGU-NOV' as origem
        FROM unimed.comp11
        WHERE "Tipo Lançamento" = 'PRODUÇÃO'
        
        UNION ALL
        
        SELECT 
            "Beneficiário" as carteirinha,
            "Nome Beneficiário" as nome_paciente,
            "Data Execução" as data_atendimento,
            "Hora Execução" as hora_atendimento,
            "Item Desc" as procedimento,
            "Nome Prestador" as prestador,
            "Valor Pagamento" as valor,
            'SGU-DEZ' as origem
        FROM unimed.comp12
        WHERE "Tipo Lançamento" = 'PRODUÇÃO'
    ),
    pep_formatted AS (
        SELECT 
            "CARTEIRA" as carteirinha,
            "PACIENTE" as nome_paciente,
            TO_DATE("DATA", 'DD/MM/YYYY') as data_atendimento,
            CASE 
                WHEN "SITUAÇÃO" = 'atendida' THEN 'ATENDIDO'
                ELSE 'NAO_ATENDIDO'
            END as status_atendimento
        FROM unimed.base_pep
    ),
    sgu_sem_pep AS (
        SELECT 
            s.*,
            'EXECUTADO_SGU_SEM_PEP' as situacao,
            'Verificar: SGU registrou execução mas não há agendamento no PEP' as observacao
        FROM sgu_formatted s
        LEFT JOIN pep_formatted p ON (
            s.carteirinha = p.carteirinha 
            AND s.data_atendimento = p.data_atendimento
        )
        WHERE p.carteirinha IS NULL
    ),
    sgu_com_pep_nao_atendido AS (
        SELECT 
            s.*,
            'EXECUTADO_SGU_NAO_ATENDIDO_PEP' as situacao,
            'Crítico: SGU executado mas PEP marcado como não atendido' as observacao
        FROM sgu_formatted s
        INNER JOIN pep_formatted p ON (
            s.carteirinha = p.carteirinha 
            AND s.data_atendimento = p.data_atendimento
        )
        WHERE p.status_atendimento = 'NAO_ATENDIDO'
    )
    
    SELECT * FROM sgu_sem_pep
    UNION ALL
    SELECT * FROM sgu_com_pep_nao_atendido
    ORDER BY valor DESC, data_atendimento DESC
    """
    
    df_divergencias = execute_query(query_divergencias)
    
    if not df_divergencias.empty:
        # Métricas das divergências
        valor_total = df_divergencias['valor'].sum()
        casos_criticos = len(df_divergencias[df_divergencias['situacao'] == 'EXECUTADO_SGU_NAO_ATENDIDO_PEP'])
        
        col0, col1, col2, col3 = st.columns(4)
        
        with col0:
            st.metric("Atendimentos Totais",6.769)

        with col1:
            st.metric("Casos Críticos", f"{casos_criticos:,}".replace(',', '.'))
        

        with col2:
            st.metric("Total de Casos", f"{len(df_divergencias):,}".replace(',', '.'))
        with col3:
            valor_formatado = f"R$ {valor_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            st.metric("Valor Total", valor_formatado)
           
        # Análise por tipo
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de divergências por tipo
            div_by_type = df_divergencias.groupby(['situacao', 'origem']).agg({
                'valor': 'sum',
                'carteirinha': 'count'
            }).reset_index()
            div_by_type.columns = ['situacao', 'origem', 'valor_total', 'quantidade']
            with st.container(border=True):
                fig_div = px.bar(
                    div_by_type,
                    x='origem',
                    y='valor_total',
                    color='situacao',
                    title="Valor das Divergências por Tipo e Mês",
                    labels={'valor_total': 'Valor (R$)', 'origem': 'Mês'},
                    color_discrete_map={
                        'EXECUTADO_SGU_SEM_PEP': '#f59e0b',
                        'EXECUTADO_SGU_NAO_ATENDIDO_PEP': '#ef4444'
                    }
                )
                st.plotly_chart(fig_div, use_container_width=True)
            
        with col2:
            with st.container(border=True):
                top_prestadores = df_divergencias.groupby('prestador')['valor'].sum().nlargest(5)
                
                st.markdown("### Top 5 Prestadores")
                for prestador, valor in top_prestadores.items():
                    st.markdown(f"**{prestador}**: R$ {valor:,.2f}".replace(',', '.'))
            
        # Casos mais críticos
        st.markdown('<h3 class="section-header">Casos Mais Críticos</h3>', unsafe_allow_html=True)
        
        casos_criticos_df = df_divergencias[
            df_divergencias['situacao'] == 'EXECUTADO_SGU_NAO_ATENDIDO_PEP'
        ].nlargest(10, 'valor')

        casos_criticos_df['valor'] = casos_criticos_df['valor'].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        
        if not casos_criticos_df.empty:
            st.markdown("""
            <div class="alert-critical">
            <strong>⚠️ ATENÇÃO:</strong> Estes casos mostram procedimentos faturados no SGU mas marcados como 
            "não atendidos" no PEP. Isso requer investigação imediata pois pode indicar fraude.
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                casos_criticos_df[['nome_paciente', 'prestador', 'procedimento', 'valor', 'data_atendimento']],
                use_container_width=True
            )
    
    else:
        st.info("Nenhuma divergência encontrada entre SGU e PEP.")

def render_conflitos_tab():
    """Renderiza a aba de conflitos de horário"""
    st.markdown('<div class="tab-header">Conflitos de Horário</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Padrão Suspeito:</strong> Profissionais que atendem múltiplos pacientes no mesmo horário podem indicar 
    superfaturamento, registros incorretos ou tentativas de burlar o sistema de controle.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para conflitos
    query_conflitos = """
    WITH conflitos_horario AS (
        SELECT 
            "PROFISSIONAL " as profissional,
            TO_DATE("DATA", 'DD/MM/YYYY') as data,
            "ATENDIMENTO" as horario,
            STRING_AGG("PACIENTE", ' | ') as pacientes,
            COUNT(*) as qtd_conflitos
        FROM unimed.base_pep
       WHERE "SITUAÇÃO" = 'atendida'
        GROUP BY "PROFISSIONAL ", TO_DATE("DATA", 'DD/MM/YYYY'), "ATENDIMENTO"
        HAVING COUNT(*) > 1
    )
    SELECT 
        profissional,
        data,
        horario,
        pacientes,
        qtd_conflitos
    FROM conflitos_horario
    ORDER BY qtd_conflitos DESC, data DESC
    """
    
    df_conflitos = execute_query(query_conflitos)
    
    if not df_conflitos.empty:
        # Métricas de conflitos
        total_conflitos = len(df_conflitos)
        max_conflitos = df_conflitos['qtd_conflitos'].max()
        profissionais_conflito = df_conflitos['profissional'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Conflitos", f"{total_conflitos:,}".replace(',', '.'))
        with col2:
            st.metric("Profissionais Envolvidos", f"{profissionais_conflito:,}".replace(',', '.'))
        with col3:
            st.metric("Máx. Pacientes Simultâneos", f"{max_conflitos}")
        
        # Análise temporal
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Conflitos por dia
            df_diario = df_conflitos.groupby('data').size().reset_index(name='conflitos')
            df_diario = df_diario.sort_values("data").reset_index(drop=True)
            with st.container(border=True):
                fig_temporal = px.line(
                    df_diario,
                    x='data',
                    y='conflitos',
                    title='Evolução dos Conflitos de Horário',
                    markers=True
                )
                fig_temporal.update_traces(line_color='red', marker_color='red')
                st.plotly_chart(fig_temporal, use_container_width=True)
            
        with col2:
            with st.container(border=True):
                if not df_conflitos.empty:
                    df_conflitos['data'] = pd.to_datetime(df_conflitos['data'])
                    df_conflitos['dia_semana'] = df_conflitos['data'].dt.day_name()
                    
                    # Mapeamento dos dias para português
                    dias_pt = {
                        'Monday': 'Segunda-feira',
                        'Tuesday': 'Terça-feira',
                        'Wednesday': 'Quarta-feira',
                        'Thursday': 'Quinta-feira',
                        'Friday': 'Sexta-feira',
                        'Saturday': 'Sábado',
                        'Sunday': 'Domingo'
                    }
                    
                    df_conflitos['dia_semana'] = df_conflitos['dia_semana'].map(dias_pt)
                    
                    calendar_df = df_conflitos.groupby('dia_semana').size().reset_index(name='conflitos')
                    
                    # Ordenar os dias da semana corretamente em português
                    dias_ordem = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
                    calendar_df['dia_semana'] = pd.Categorical(calendar_df['dia_semana'], categories=dias_ordem, ordered=True)
                    calendar_df = calendar_df.sort_values('dia_semana')
                    
                    fig_semana = px.bar(
                        calendar_df,
                        x='dia_semana',
                        y='conflitos',
                        title="Conflitos por Dia da Semana",
                        color='conflitos',
                        color_continuous_scale='Reds',
                        labels={'dia_semana': 'Dia da Semana', 'conflitos': 'Número de Conflitos'}
                    )
                    st.plotly_chart(fig_semana, use_container_width=True)
                else:
                    st.info("Sem dados para análise semanal")
        # Profissionais com mais conflitos
        st.markdown('<h3 class="section-header">Profissionais com Conflitos</h3>', unsafe_allow_html=True)
        
        conflitos_por_profissional = df_conflitos.groupby('profissional').agg({
            'qtd_conflitos': 'sum',
            'data': 'count'
        }).reset_index()
        conflitos_por_profissional.columns = ['profissional', 'total_conflitos', 'dias_com_conflito']
        conflitos_por_profissional = conflitos_por_profissional.sort_values('total_conflitos', ascending=False)
        
        # Identificar casos mais críticos
        casos_criticos = conflitos_por_profissional[conflitos_por_profissional['total_conflitos'] >= 10]
        
        if not casos_criticos.empty:
            st.markdown("""
            <div class="alert-high">
            <strong>Profissionais com Padrão Crítico:</strong> Os profissionais abaixo apresentam 
            um número elevado de conflitos que merece investigação.
            </div>
            """, unsafe_allow_html=True)
            
            for _, row in casos_criticos.head(5).iterrows():
                st.markdown(f"• **{row['profissional']}**: {row['total_conflitos']} conflitos em {row['dias_com_conflito']} dias")
        
        # Tabela detalhada dos piores casos
        with st.expander("Ver Casos Mais Graves (+ pacientes simultâneos)"):
            casos_graves = df_conflitos[df_conflitos['qtd_conflitos'] >= 0]
            if not casos_graves.empty:
                st.dataframe(casos_graves, use_container_width=True)
            else:
                st.info("Nenhum caso com 4+ pacientes simultâneos encontrado.")
    
    else:
        st.success("Nenhum conflito de horário detectado!")

def render_intervalos_tab():
    """Renderiza a aba de intervalos suspeitos"""
    st.markdown('<div class="tab-header">Intervalos Suspeitos</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Tempo é Dinheiro:</strong> Intervalos muito curtos entre inícios de atendimentos (< 30 minutos) podem indicar 
    consultas superficiais, registros fictícios ou tentativas de maximizar faturamento sem qualidade adequada.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para intervalos
    query_intervalos = """
    WITH intervalos_calculados AS (
        SELECT 
            "PROFISSIONAL ",
            "PACIENTE",
            TO_DATE("DATA", 'DD/MM/YYYY') as data,
            "ATENDIMENTO",
            LAG("ATENDIMENTO") OVER (
                PARTITION BY "PROFISSIONAL ", TO_DATE("DATA", 'DD/MM/YYYY') 
                ORDER BY "ATENDIMENTO"
            ) as atendimento_anterior,
            EXTRACT(EPOCH FROM (
                "ATENDIMENTO"::time - LAG("ATENDIMENTO"::time) OVER (
                    PARTITION BY "PROFISSIONAL ", TO_DATE("DATA", 'DD/MM/YYYY') 
                    ORDER BY "ATENDIMENTO"
                )
            ))/60 as intervalo_minutos
        FROM unimed.base_pep
        WHERE "SITUAÇÃO" = 'atendida'
    )
    SELECT 
        "PROFISSIONAL ",
        "PACIENTE",
        data,
        "ATENDIMENTO",
        atendimento_anterior,
        intervalo_minutos,
        CASE 
            WHEN intervalo_minutos <= 15 THEN 'CRÍTICO'
            WHEN intervalo_minutos <= 30 THEN 'SUSPEITO'
            WHEN intervalo_minutos <= 45 THEN 'QUESTIONÁVEL'
            ELSE 'NORMAL'
        END as classificacao
    FROM intervalos_calculados
    WHERE intervalo_minutos IS NOT NULL 
    AND intervalo_minutos <= 45
    ORDER BY intervalo_minutos ASC, data DESC
    """
    
    df_intervalos = execute_query(query_intervalos)
    
    if not df_intervalos.empty:
        # Métricas de intervalos
        criticos = len(df_intervalos[df_intervalos['classificacao'] == 'CRÍTICO'])
        suspeitos = len(df_intervalos[df_intervalos['classificacao'] == 'SUSPEITO'])
        questionaveis = len(df_intervalos[df_intervalos['classificacao'] == 'QUESTIONÁVEL'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Críticos (≤15 min)", f"{criticos:,}".replace(',', '.'))
        with col2:
            st.metric("Suspeitos (16-30 min)", f"{suspeitos:,}".replace(',', '.'))
        with col3:
            st.metric("Questionáveis (31-45 min)", f"{questionaveis:,}".replace(',', '.'))
        
        # Distribuição dos intervalos
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container(border=True):
                fig_box = px.box(
                    df_intervalos,
                    x="PROFISSIONAL ",
                    y="intervalo_minutos",
                    color="classificacao",
                    color_discrete_map={
                        'CRÍTICO': '#dc2626',
                        'SUSPEITO': '#f59e0b',
                        'QUESTIONÁVEL': '#eab308'
                    },
                    title="Distribuição dos Intervalos entre inícios de Atendimentos por Profissional",
                    labels={"intervalo_minutos": "Intervalo (minutos)"}
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                df_intervalos['classificacao'] = df_intervalos['classificacao'].str.upper().str.strip()

                # Recalcula os valores após normalização
                class_count = df_intervalos['classificacao'].value_counts()

                # Define o mapa de cores
                color_map = {
                    'CRÍTICO': '#dc2626',
                    'SUSPEITO': '#f59e0b',
                    'QUESTIONÁVEL': '#eab308'
                }

                # Gera o gráfico de pizza
                fig_pie = px.pie(
                    values=class_count.values,
                    names=class_count.index,
                    title="Distribuição por Classificação",
                    color=class_count.index,  # precisa explicitar que o campo usado será para colorir
                    color_discrete_map=color_map
                )

                st.plotly_chart(fig_pie, use_container_width=True)

        # Profissionais com mais casos suspeitos
        st.markdown('<h3 class="section-header">Ranking de Risco por Profissional</h3>', unsafe_allow_html=True)
        
        ranking = df_intervalos.groupby(['PROFISSIONAL ', 'classificacao']).size().unstack(fill_value=0)
        
        # Garantir que todas as colunas de classificação existam
        for col in ['CRÍTICO', 'SUSPEITO', 'QUESTIONÁVEL']:
            if col not in ranking.columns:
                ranking[col] = 0
                
        ranking['score_risco'] = ranking['CRÍTICO'] * 3 + ranking['SUSPEITO'] * 2 + ranking['QUESTIONÁVEL'] * 1
        ranking = ranking.sort_values('score_risco', ascending=False)
        
        top_risk = ranking.head(10)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Gráfico de barras empilhadas
            top_risk_reset = top_risk.reset_index()
            with st.container(border=True):
                fig_ranking = px.bar(
                    top_risk_reset,
                    x='PROFISSIONAL ',
                    y=['CRÍTICO', 'SUSPEITO', 'QUESTIONÁVEL'],
                    title="Top 10 Profissionais com Intervalos Suspeitos",
                    color_discrete_map={
                        'CRÍTICO': '#dc2626',
                        'SUSPEITO': '#f59e0b',
                        'QUESTIONÁVEL': '#eab308'
                    },
                    labels={'value': 'Número de Casos', 'PROFISSIONAL ': 'Profissional'}
                )
                fig_ranking.update_xaxes(tickangle=45)
                st.plotly_chart(fig_ranking, use_container_width=True)
            
        with col2:
            with st.container(border=True):
                st.markdown("### 🎯 Score de Risco")
                st.caption("Crítico: 3pts | Suspeito: 2pts | Questionável: 1pt")
                
                for prof, score in top_risk['score_risco'].head(5).items():
                
                    criticos = top_risk.loc[prof, 'CRÍTICO'] if 'CRÍTICO' in top_risk.columns else 0
                    st.markdown(f"**{prof.strip()}**: {score} pts")
                    if criticos > 0:
                        st.markdown(f"<span style='color: red'>⚠️ {criticos} casos críticos</span>", unsafe_allow_html=True)
            
        # Casos mais críticos para investigação
        casos_criticos = df_intervalos[df_intervalos['classificacao'] == 'CRÍTICO'].head(20)
        
        if not casos_criticos.empty:
            st.markdown('<h3 class="section-header">Casos Críticos para Investigação Imediata</h3>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="alert-critical">
            <strong>AÇÃO REQUERIDA:</strong> Os casos abaixo apresentam intervalos ≤ 15 minutos entre atendimentos. 
            Isso é fisicamente impossível para consultas adequadas e requer verificação imediata.
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                casos_criticos[['PROFISSIONAL ', 'data', 'atendimento_anterior', 'ATENDIMENTO', 'intervalo_minutos']],
                use_container_width=True
            )
    
    else:
        st.success("✅ Nenhum intervalo suspeito detectado!")

def render_noshow_tab():
    """Renderiza a aba de análise de no-show"""
    st.markdown('<div class="tab-header">Análise de No-Show</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Comportamentos Anômalos:</strong> Taxas extremas de no-show (muito altas ou muito baixas) podem indicar 
    problemas operacionais, registros incorretos ou tentativas de manipulação do sistema.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para análise de no-show
    query_noshow = """
    WITH medico_stats AS (
        SELECT 
            "PROFISSIONAL " as medico,
            "ESPECIALIDADE",
            COUNT(*) as total_agendamentos,
            SUM(CASE 
                WHEN "SITUAÇÃO" = 'não atendida' OR "SITUAÇÃO" ILIKE '%não%' OR "SITUAÇÃO" ILIKE '%nao%'
                THEN 1 ELSE 0 
            END) as total_no_shows,
            SUM(CASE 
                WHEN "SITUAÇÃO" = 'atendida' OR "SITUAÇÃO" ILIKE '%atend%'
                THEN 1 ELSE 0 
            END) as total_atendidos,
            COUNT(DISTINCT "PACIENTE ID") as pacientes_unicos
        FROM unimed.base_pep
        WHERE "PROFISSIONAL " IS NOT NULL AND TRIM("PROFISSIONAL ") != ''
        GROUP BY "PROFISSIONAL ", "ESPECIALIDADE"
    )
    SELECT 
        medico,
        "ESPECIALIDADE",
        total_agendamentos,
        total_no_shows,
        total_atendidos,
        pacientes_unicos,
        CASE 
            WHEN total_agendamentos > 0 
            THEN ROUND((total_no_shows::numeric / total_agendamentos::numeric) * 100, 2)
            ELSE 0 
        END as taxa_no_show_percent,
        CASE 
            WHEN total_agendamentos < 10 THEN 'POUCOS_DADOS'
            WHEN total_no_shows = 0 THEN 'ZERO_NOSHOW'
            WHEN (total_no_shows::numeric / total_agendamentos::numeric) >= 0.5 THEN 'CRÍTICO'
            WHEN (total_no_shows::numeric / total_agendamentos::numeric) >= 0.35 THEN 'ALTO'
            WHEN (total_no_shows::numeric / total_agendamentos::numeric) >= 0.25 THEN 'MÉDIO'
            WHEN (total_no_shows::numeric / total_agendamentos::numeric) >= 0.1 THEN 'BAIXO'
            ELSE 'MUITO_BAIXO'
        END as classificacao_risco
    FROM medico_stats
    WHERE total_agendamentos >= 5
    ORDER BY taxa_no_show_percent DESC
    """
    
    df_noshow = execute_query(query_noshow)
    
    if not df_noshow.empty:
        # Métricas gerais
        media_taxa = df_noshow['taxa_no_show_percent'].mean()
        casos_criticos = len(df_noshow[df_noshow['classificacao_risco'] == 'CRÍTICO'])
        casos_zero = len(df_noshow[df_noshow['classificacao_risco'] == 'ZERO_NOSHOW'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Taxa Média", f"{media_taxa:.1f}%")
        with col2:
            st.metric("Casos Críticos (≥50%)", f"{casos_criticos}")
        with col3:
            st.metric("Zero No-Show", f"{casos_zero}")
        with col4:
            st.metric("Total Profissionais", f"{len(df_noshow)}")
        
        # Análise de distribuição
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container(border=True):
                fig_scatter = px.scatter(
                    df_noshow,
                    x='total_agendamentos',
                    y='taxa_no_show_percent',
                    size='pacientes_unicos',
                    color='classificacao_risco',
                    hover_name='medico',
                    title='Relação entre Volume de Agendamentos e Taxa de No-Show',
                    labels={
                        'total_agendamentos': 'Total de Agendamentos',
                        'taxa_no_show_percent': 'Taxa de No-Show (%)',
                        'pacientes_unicos': 'Pacientes Únicos'
                    },
                    color_discrete_map={
                        'CRÍTICO': '#dc2626',
                        'ALTO': '#ea580c',
                        'MÉDIO': '#f59e0b',
                        'BAIXO': '#84cc16',
                        'MUITO_BAIXO': '#22c55e',
                        'ZERO_NOSHOW': '#3b82f6',
                        'POUCOS_DADOS': '#9ca3af'
                    }
                )
                fig_scatter.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Distribuição por classificação
            # Normaliza os valores para evitar diferenças por espaços ou 
            with st.container(border=True):
                df_noshow['classificacao_risco'] = df_noshow['classificacao_risco'].str.upper().str.strip()

                # Recalcula os valores após a normalização
                class_dist = df_noshow['classificacao_risco'].value_counts()

                # Mapa de cores conforme os rótulos esperados
                color_map = {
                    'CRÍTICO': '#dc2626',
                    'ALTO': '#ea580c',
                    'MÉDIO': '#f59e0b',
                    'BAIXO': '#84cc16',
                    'MUITO_BAIXO': '#22c55e',
                    'ZERO_NOSHOW': '#3b82f6',
                    'POUCOS_DADOS': '#9ca3af'
                }

                # Cria o gráfico de pizza com mapeamento explícito de cores
                fig_class = px.pie(
                    values=class_dist.values,
                    names=class_dist.index,
                    title="Distribuição por Risco",
                    color=class_dist.index,
                    color_discrete_map=color_map
                )

                st.plotly_chart(fig_class, use_container_width=True)

        
        # Casos que requerem atenção
        st.markdown('<h3 class="section-header">🎯 Casos que Requerem Atenção</h3>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Críticos", "Suspeitos", "Outliers"])
        
        with tab1:
            criticos = df_noshow[df_noshow['classificacao_risco'] == 'CRÍTICO']
            if not criticos.empty:
                st.markdown("""
                <div class="alert-critical">
                <strong>TAXA CRÍTICA (≥50%):</strong> Estes profissionais apresentam taxas extremamente altas 
                que podem indicar problemas graves ou manipulação de registros.
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    criticos[['medico', 'ESPECIALIDADE', 'total_agendamentos', 'taxa_no_show_percent']],
                    use_container_width=True
                )
            else:
                st.success("Nenhum caso crítico encontrado!")
        
        with tab2:
            # Zero no-shows podem ser suspeitos também
            zero_noshow = df_noshow[df_noshow['classificacao_risco'] == 'ZERO_NOSHOW']
            if not zero_noshow.empty:
                st.markdown("""
                <div class="alert-medium">
                <strong>ZERO NO-SHOW:</strong> Embora positivo, zero no-shows em volumes altos pode indicar 
                registros incorretos ou práticas questionáveis.
                </div>
                """, unsafe_allow_html=True)
                
                zero_alto_volume = zero_noshow[zero_noshow['total_agendamentos'] >= 50]
                if not zero_alto_volume.empty:
                    st.dataframe(
                        zero_alto_volume[['medico', 'ESPECIALIDADE', 'total_agendamentos', 'pacientes_unicos']],
                        use_container_width=True
                    )
                else:
                    st.info("Todos os casos de zero no-show são de baixo volume.")
            else:
                st.info("Nenhum profissional com zero no-show encontrado.")
        
        with tab3:
            # Análise de outliers estatísticos
            if len(df_noshow) >= 4:  # Precisamos de pelo menos 4 pontos para calcular quartis
                Q1 = df_noshow['taxa_no_show_percent'].quantile(0.25)
                Q3 = df_noshow['taxa_no_show_percent'].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Evitar divisão por zero
                    limite_superior = Q3 + 1.5 * IQR
                    limite_inferior = Q1 - 1.5 * IQR
                    
                    outliers_altos = df_noshow[df_noshow['taxa_no_show_percent'] > limite_superior]
                    outliers_baixos = df_noshow[df_noshow['taxa_no_show_percent'] < limite_inferior]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Outliers Altos")
                        if not outliers_altos.empty:
                            st.dataframe(
                                outliers_altos[['medico', 'taxa_no_show_percent']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("Nenhum outlier alto detectado.")
                    
                    with col2:
                        st.subheader("Outliers Baixos")
                        if not outliers_baixos.empty:
                            st.dataframe(
                                outliers_baixos[['medico', 'taxa_no_show_percent']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("Nenhum outlier baixo detectado.")
                else:
                    st.info("Dados insuficientes para análise de outliers (IQR = 0)")
            else:
                st.info("Dados insuficientes para análise de outliers (< 4 profissionais)")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_brazilian_gender_classifier():
    """
    Create a Brazilian name-based gender classifier using common patterns
    and the actual names from your database
    """
    
    # Common Brazilian female names (add more from your database analysis)
    female_names = {
        'MARIA', 'ANA', 'LUCIA', 'FERNANDA', 'BEATRIZ', 'CARLA', 'PAULA', 'SANDRA', 
        'ADRIANA', 'PATRICIA', 'JULIANA', 'CLAUDIA', 'SILVIA', 'MARCIA', 'MONICA', 
        'CRISTINA', 'ROSA', 'JOANA', 'HELENA', 'ISABELA', 'GABRIELA', 'RAFAELA',
        'LETICIA', 'CAMILA', 'AMANDA', 'RENATA', 'VANESSA', 'DANIELA', 'PRISCILA',
        'FABIANA', 'LUCIANA', 'SIMONE', 'ELIANE', 'SOLANGE', 'TATIANA', 'VIVIANE',
        'ANDREIA', 'MICHELE', 'FRANCINE', 'CAROLINE', 'ALINE', 'JANAINA', 'KARINA'
    }
    
    # Common Brazilian male names (add more from your database analysis)
    male_names = {
        'ANTONIO', 'PEDRO', 'CARLOS', 'PAULO', 'JOÃO', 'JOSE', 'FRANCISCO', 'LUIS',
        'MARCOS', 'ROBERTO', 'RICARDO', 'FERNANDO', 'SERGIO', 'ALEXANDRE', 'ANDRE',
        'MARCELO', 'LEONARDO', 'EDUARDO', 'RAFAEL', 'RODRIGO', 'DANIEL', 'MARCIO',
        'FABIO', 'GABRIEL', 'GUSTAVO', 'BRUNO', 'DIEGO', 'FELIPE', 'LUCAS', 'MATEUS',
        'THIAGO', 'VINICIUS', 'WELLINGTON', 'ANDERSON', 'JEFFERSON', 'LEANDRO',
        'RENATO', 'CLAUDIO', 'FLAVIO', 'JULIO', 'MAURICIO', 'NELSON', 'WAGNER'
    }
    
    # Brazilian feminine name endings
    feminine_endings = {
        'A', 'ANA', 'INA', 'INE', 'INHA', 'ETTE', 'ELLY', 'ELLY', 'ELLA', 'ICIA',
        'ENCIA', 'ANTA', 'ILDA', 'UNDA'
    }
    
    # Brazilian masculine name endings  
    masculine_endings = {
        'O', 'OS', 'OR', 'AR', 'ER', 'IR', 'SON', 'TON', 'ALDO', 'ARDO', 'ERTO',
        'ILDO', 'UNDO', 'INHO', 'ISMO', 'IANO'
    }
    
    return {
        'female_names': female_names,
        'male_names': male_names,
        'feminine_endings': feminine_endings,
        'masculine_endings': masculine_endings
    }

def detect_gender_brazilian(full_name, classifier_data):
    """
    Detect gender using Brazilian name patterns and your database names
    """
    if pd.isna(full_name) or full_name == '':
        return 'desconhecido'
    
    # Extract and clean first name
    name_parts = str(full_name).upper().strip().split()
    if not name_parts:
        return 'desconhecido'
    
    first_name = name_parts[0]
    
    # Direct lookup in known names
    if first_name in classifier_data['female_names']:
        return 'feminino'
    if first_name in classifier_data['male_names']:
        return 'masculino'
    
    # Pattern matching for Brazilian names
    # Check feminine endings
    for ending in classifier_data['feminine_endings']:
        if first_name.endswith(ending):
            return 'feminino'
    
    # Check masculine endings
    for ending in classifier_data['masculine_endings']:
        if first_name.endswith(ending):
            return 'masculino'
    
    # Special Brazilian patterns
    if first_name.endswith('A') and len(first_name) > 2:
        return 'feminino'
    elif first_name.endswith('O') and len(first_name) > 2:
        return 'masculino'
    
    return 'desconhecido'

def get_name_analysis_query():
    """Get extended query to analyze all unique first names from your database"""
    return """
    SELECT 
        UPPER(TRIM(substring("PACIENTE", 1, CASE 
            WHEN strpos("PACIENTE", ' ') > 0 
            THEN strpos("PACIENTE", ' ') - 1 
            ELSE length("PACIENTE") 
        END))) as primeiro_nome,
        COUNT(*) as frequencia
    FROM unimed.base_pep 
    WHERE "PACIENTE" IS NOT NULL 
        AND "PACIENTE" != ''
        AND length(trim("PACIENTE")) > 2
    GROUP BY primeiro_nome
    HAVING COUNT(*) >= 2  -- Only names that appear at least twice
    ORDER BY frequencia DESC
    """

def execute_population_query_with_gender():
    """Execute the comprehensive population analysis query with gender detection"""
    query = """
    WITH paciente_base AS (
        SELECT 
            "PACIENTE",
            "CARTEIRA",
            "NASCIMENTO",
            "DATA",
            "SITUAÇÃO",
            "ESPECIALIDADE",
            "PROFISSIONAL ",
            "CHEGADA",
            "ATENDIMENTO",
            TO_DATE("DATA", 'DD/MM/YYYY') as data_formatada,
            -- Calcula idade
            EXTRACT(YEAR FROM AGE(DATE '2025-08-01', TO_DATE("NASCIMENTO", 'DD/MM/YYYY'))) as idade,
            -- Tempo de espera (quando possível calcular)
            CASE 
                WHEN "CHEGADA" IS NOT NULL AND "ATENDIMENTO" IS NOT NULL 
                     AND "CHEGADA" != '' AND "ATENDIMENTO" != ''
                     AND "CHEGADA" != 'nada' AND "ATENDIMENTO" != 'nada'
                THEN EXTRACT(EPOCH FROM (CAST("ATENDIMENTO" AS TIME) - CAST("CHEGADA" AS TIME)))/60
                ELSE NULL
            END as tempo_espera_minutos
        FROM unimed.base_pep
        WHERE "DATA" IS NOT NULL
          AND "NASCIMENTO" IS NOT NULL
          AND "PACIENTE" IS NOT NULL
    ),
    paciente_metricas AS (
        SELECT 
            "PACIENTE",
            "CARTEIRA", 
            MAX(idade) as idade,
            
            -- MÉTRICAS DE VOLUME
            COUNT(*) as total_agendamentos,
            COUNT(DISTINCT "ESPECIALIDADE") as especialidades_diferentes,
            COUNT(DISTINCT "PROFISSIONAL ") as medicos_diferentes,
            
            -- MÉTRICAS DE COMPORTAMENTO
            SUM(CASE WHEN "SITUAÇÃO" = 'atendida' THEN 1 ELSE 0 END) as total_atendidos,
            SUM(CASE WHEN "SITUAÇÃO" = 'cancelada' THEN 1 ELSE 0 END) as total_cancelados,
            SUM(CASE WHEN "SITUAÇÃO" = 'nao compareceu' THEN 1 ELSE 0 END) as total_no_shows,
            
            -- TAXAS DE COMPORTAMENTO
            ROUND((SUM(CASE WHEN "SITUAÇÃO" = 'atendida' THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 2) as taxa_comparecimento,
            ROUND((SUM(CASE WHEN "SITUAÇÃO" = 'nao compareceu' THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 2) as taxa_no_show,
            ROUND((SUM(CASE WHEN "SITUAÇÃO" = 'cancelada' THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 2) as taxa_cancelamento,
            
            -- MÉTRICAS TEMPORAIS
            MIN(data_formatada) as primeiro_agendamento,
            MAX(data_formatada) as ultimo_agendamento,
            MAX(data_formatada) - MIN(data_formatada) + 1 as periodo_acompanhamento_dias,
            
            -- MÉTRICAS DE TEMPO DE ESPERA
            AVG(tempo_espera_minutos) as tempo_espera_medio_minutos,
            STDDEV(tempo_espera_minutos) as tempo_espera_desvio_padrao,
            COUNT(CASE WHEN tempo_espera_minutos IS NOT NULL THEN 1 END) as agendamentos_com_tempo_espera,
            
            -- ESPECIALIDADE MAIS FREQUENTE
            MODE() WITHIN GROUP (ORDER BY "ESPECIALIDADE") as especialidade_principal
            
        FROM paciente_base
        GROUP BY  "PACIENTE", "CARTEIRA"
    ),
    paciente_intervalos AS (
        SELECT 
            "CARTEIRA",
            -- Calcula intervalos entre consultas
            AVG(CASE 
                WHEN intervalo_dias > 0 THEN intervalo_dias 
                ELSE NULL 
            END) as media_dias_entre_consultas,
            STDDEV(CASE 
                WHEN intervalo_dias > 0 THEN intervalo_dias 
                ELSE NULL 
            END) as desvio_dias_entre_consultas,
            MIN(CASE 
                WHEN intervalo_dias > 0 THEN intervalo_dias 
                ELSE NULL 
            END) as menor_intervalo_dias,
            MAX(intervalo_dias) as maior_intervalo_dias
        FROM (
            SELECT 
               "CARTEIRA",
                data_formatada,
                data_formatada - LAG(data_formatada) OVER (
                    PARTITION BY "CARTEIRA"
                    ORDER BY data_formatada
                ) as intervalo_dias
            FROM paciente_base
        ) intervalos
        GROUP BY "CARTEIRA"
    ),
    paciente_clustering_features AS (
        SELECT 
            m."PACIENTE",
            m."CARTEIRA",
            m.idade,
            
            -- FEATURES PARA CLUSTERING
            -- Volume de uso
            m.total_agendamentos,
            m.especialidades_diferentes,
            m.medicos_diferentes,
            
            -- Comportamento
            m.taxa_comparecimento,
            m.taxa_no_show,
            m.taxa_cancelamento,
            
            -- Padrões temporais
            m.periodo_acompanhamento_dias,
            COALESCE(i.media_dias_entre_consultas, 0) as media_dias_entre_consultas,
            COALESCE(i.desvio_dias_entre_consultas, 0) as variabilidade_intervalos,
            
            -- Experiência de atendimento
            COALESCE(m.tempo_espera_medio_minutos, 0) as tempo_espera_medio,
            m.agendamentos_com_tempo_espera,
            
            -- Especialidade principal
            m.especialidade_principal,
            
            -- FEATURES DERIVADAS
            -- Intensidade de uso (agendamentos por dia de acompanhamento)
            CASE 
                WHEN m.periodo_acompanhamento_dias > 0 
                THEN ROUND(m.total_agendamentos::numeric / m.periodo_acompanhamento_dias::numeric, 4)
                ELSE 0 
            END as intensidade_uso,
            
            -- Diversidade de cuidados (especialidades / total agendamentos)
            ROUND(m.especialidades_diferentes::numeric / m.total_agendamentos::numeric, 4) as diversidade_especialidades,
            
            -- Fidelidade médica (agendamentos / médicos diferentes)
            ROUND(m.total_agendamentos::numeric / m.medicos_diferentes::numeric, 2) as fidelidade_medica,
            
            -- Classificação de risco baseada em no-show
            CASE 
                WHEN m.taxa_no_show >= 50 THEN 'ALTO_RISCO'
                WHEN m.taxa_no_show >= 20 THEN 'MEDIO_RISCO'
                WHEN m.taxa_no_show > 0 THEN 'BAIXO_RISCO'
                ELSE 'SEM_RISCO'
            END as categoria_risco_no_show,
            
            -- Categoria de idade
            CASE 
                WHEN m.idade < 18 THEN 'CRIANCA_ADOLESCENTE (0-18)'
                WHEN m.idade < 35 THEN 'JOVEM_ADULTO (19-35)'
                WHEN m.idade < 60 THEN 'ADULTO (36-59)'
              
                ELSE 'IDOSO (60+)'
            END as categoria_idade
            
        FROM paciente_metricas m
        LEFT JOIN paciente_intervalos i ON m."CARTEIRA" = i."CARTEIRA"
        WHERE m.total_agendamentos >= 2  -- Apenas pacientes com pelo menos 2 agendamentos
    )
    
    -- Dados finais para clustering
    SELECT 
        "PACIENTE",
        "CARTEIRA",
        
        -- FEATURES NUMÉRICAS PARA CLUSTERING (normalizar antes do K-means)
        idade,
        total_agendamentos,
        especialidades_diferentes,
        medicos_diferentes,
        taxa_comparecimento,
        taxa_no_show,
        taxa_cancelamento,
        periodo_acompanhamento_dias,
        media_dias_entre_consultas,
        variabilidade_intervalos,
        tempo_espera_medio,
        intensidade_uso,
        diversidade_especialidades,
        fidelidade_medica,
        
        -- FEATURES CATEGÓRICAS (para análise pós-clustering)
        especialidade_principal,
        categoria_risco_no_show,
        categoria_idade,
        
        -- MÉTRICAS ADICIONAIS PARA INTERPRETAÇÃO
        agendamentos_com_tempo_espera
        
    FROM paciente_clustering_features
    ORDER BY total_agendamentos DESC
    """
    
    return execute_query(query)

def perform_patient_clustering(df, n_clusters=3):
    """Perform K-means clustering on patient data"""
    
    # Select numerical features for clustering
    clustering_features = [
        'idade',  'tempo_espera_medio','media_dias_entre_consultas',
        'taxa_comparecimento', 'taxa_no_show',
        'intensidade_uso', 'diversidade_especialidades', 
    ]
    
    # Prepare data for clustering
    X = df[clustering_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    # Calculate cluster characteristics
    cluster_summary = df.groupby('cluster').agg({
        'idade': 'mean',
        'total_agendamentos': 'mean',
        'taxa_no_show': 'mean',
        'intensidade_uso': 'mean',
        'PACIENTE': 'count'
    }).round(2)
    
    cluster_summary.columns = ['Idade Média', 'Consultas Médias', 'Taxa No-Show %', 'Intensidade Uso', 'Qtd Pacientes']
    
    return df, cluster_summary

def render_pacientes_tab():
    """Renderiza a aba de análise de pacientes com análise populacional e detecção de gênero brasileira"""
    st.markdown('<div class="tab-header">Análise de Pacientes</div>', unsafe_allow_html=True)
    
 
    
    # Initialize Brazilian gender classifier
    classifier_data = create_brazilian_gender_classifier()
    
    # Option to analyze name patterns from database
    
    
    # Execute population query
    with st.spinner('Carregando dados populacionais...'):
        df_population = execute_population_query_with_gender()
    
    if df_population.empty:
        st.error("Nenhum dado encontrado para análise populacional.")
        return
    
    # Add Brazilian gender detection
    with st.spinner('Detectando gênero usando padrões brasileiros...'):
        df_population['genero'] = df_population['PACIENTE'].apply(
            lambda x: detect_gender_brazilian(x, classifier_data)
        )
    
    # Population Overview
    st.markdown('<h3 class="section-header">Visão Geral da População</h3>', unsafe_allow_html=True)
    
    # Key metrics
    total_patients = len(df_population)
    avg_age = df_population['idade'].mean()
    avg_appointments = df_population['total_agendamentos'].mean()
    overall_no_show_rate = df_population['taxa_no_show'].mean()
    gender_accuracy = len(df_population[df_population['genero'] != 'desconhecido']) / total_patients * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Pacientes", f"{total_patients:,}".replace(',', '.'))
    with col2:
        st.metric("Idade Média", f"{avg_age:.1f} anos")
    with col3:
        st.metric("Consultas Médias", f"{avg_appointments:.1f}")
    with col4:
        st.metric("Taxa No-Show Média", f"{overall_no_show_rate:.1f}%")
    with col5:
        st.metric("Precisão Gênero", f"{gender_accuracy:.1f}%")
    
  
    st.markdown('<h3 class="section-header"> Análise de Gênero</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
   
        with st.container(border=True):
            gender_dist = df_population['genero'].value_counts().reset_index()
            gender_dist.columns = ['genero', 'count']

            fig_gender = px.pie(
                gender_dist,
                values='count',
                names='genero',
                color='genero',  # <--- explicitly define the color reference
                title="Distribuição por Gênero",
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )

            st.plotly_chart(fig_gender, use_container_width=True)

        
        
    
    with col2:
        with st.container(border=True):
            fig_age_gender = px.histogram(
                df_population,
                x='idade',
                color='genero',
                nbins=20,
                title="Distribuição de Idade por Gênero",
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            st.plotly_chart(fig_age_gender, use_container_width=True)
  
    # Behavioral Analysis by Gender
    st.markdown('<h3 class="section-header">Análise Comportamental por Gênero</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            behavior_analysis = df_population.groupby(['categoria_idade', 'genero'])['taxa_no_show'].mean().reset_index()
            fig_behavior = px.bar(
                behavior_analysis,
                x='categoria_idade',
                y='taxa_no_show',
                color='genero',
                title="Taxa de No-Show por Idade e Gênero",
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            fig_behavior.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_behavior, use_container_width=True)
    
    with col2:
       with st.container(border=True):
            appointments_analysis = df_population.groupby(['categoria_idade', 'genero'])['total_agendamentos'].mean().reset_index()
            fig_appointments = px.bar(
                appointments_analysis,
                x='categoria_idade',
                y='total_agendamentos',
                color='genero',
                title="Média de Consultas por Idade e Gênero",
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            fig_appointments.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_appointments, use_container_width=True)
        
   
    with st.spinner('Realizando análise de clusters...'):
        df_clustered, cluster_summary = perform_patient_clustering(df_population)
    
   
    
    
    # Specialty Analysis by Gender
    st.markdown('<h3 class="section-header">Análise por Especialidade e Gênero</h3>', unsafe_allow_html=True)
    
    specialty_gender = df_population.groupby(['especialidade_principal', 'genero']).size().reset_index(name='count')
    top_specialties = df_population['especialidade_principal'].value_counts().head(10).index
    specialty_gender_filtered = specialty_gender[specialty_gender['especialidade_principal'].isin(top_specialties)]

    # Order by total counts - create the same order as value_counts()
    specialty_order = df_population['especialidade_principal'].value_counts().head(10).index.tolist()

    fig_specialty_gender = px.bar(
        specialty_gender_filtered,
        x='especialidade_principal',
        y='count',
        color='genero',
        title="Top 10 Especialidades por Gênero",
        color_discrete_map={
            'feminino': '#ff69b4',
            'masculino': '#4169e1',
            'desconhecido': '#808080'
        },
        category_orders={'especialidade_principal': specialty_order}  # Add this line
    )
    fig_specialty_gender.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_specialty_gender, use_container_width=True)
    #aqui
    # Add this section after the "Análise por Especialidade e Gênero" section in render_pacientes_tab()

    st.markdown('---')
    st.markdown('<h3 class="section-header">Análise de Tempos de Espera e Intervalos entre Consultas</h3>', unsafe_allow_html=True)
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)

    with col1:
        # Distribution of average waiting time
        # Filter out zero values for better visualization
        wait_time_data = df_population[df_population['tempo_espera_medio'] > 0]['tempo_espera_medio']
        with st.container(border=True):
            fig_wait_time = px.histogram(
                wait_time_data,
                x=wait_time_data,
                nbins=30,
                title="Distribuição do Tempo Médio de Espera (minutos)",
                labels={'x': 'Tempo de Espera (minutos)', 'count': 'Número de Pacientes'}
            )
            fig_wait_time.update_traces(marker_color='#1f77b4')
            fig_wait_time.update_layout(
                showlegend=False,
                xaxis_title="Tempo de Espera (minutos)",
                yaxis_title="Número de Pacientes"
            )
            st.plotly_chart(fig_wait_time, use_container_width=True)
            
            
            st.markdown("**Estatísticas do Tempo de Espera:**")
            wait_stats = wait_time_data.describe()
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Média", f"{wait_stats['mean']:.1f} min")
            with col1_2:
                st.metric("Mediana", f"{wait_stats['50%']:.1f} min")
            with col1_3:
                st.metric("Máximo", f"{wait_stats['max']:.1f} min")

    with col2:
        # Distribution of average days between consultations
        with st.container(border=True):
            interval_data = df_population[df_population['media_dias_entre_consultas'] > 0]['media_dias_entre_consultas']
            
            fig_interval = px.histogram(
                interval_data,
                x=interval_data,
                nbins=30,
                title="Distribuição da Média de Dias entre Consultas",
                labels={'x': 'Dias entre Consultas', 'count': 'Número de Pacientes'}
            )
            fig_interval.update_traces(marker_color='#2ca02c')
            fig_interval.update_layout(
                showlegend=False,
                xaxis_title="Dias entre Consultas",
                yaxis_title="Número de Pacientes"
            )
            st.plotly_chart(fig_interval, use_container_width=True)
            
            # Summary statistics for consultation intervals
            st.markdown("**Estatísticas do Intervalo entre Consultas:**")
            interval_stats = interval_data.describe()
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Média", f"{interval_stats['mean']:.1f} dias")
            with col2_2:
                st.metric("Mediana", f"{interval_stats['50%']:.1f} dias")
            with col2_3:
                st.metric("Máximo", f"{interval_stats['max']:.1f} dias")

    # Analysis by gender and age category
    st.markdown('<h4 class="section-header">Análise Comparativa por Gênero e Idade</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Average wait time by gender and age category
        wait_time_analysis = df_population[df_population['tempo_espera_medio'] > 0].groupby(['categoria_idade', 'genero'])['tempo_espera_medio'].mean().reset_index()
        with st.container(border=True):
            fig_wait_gender = px.bar(
                wait_time_analysis,
                x='categoria_idade',
                y='tempo_espera_medio',
                color='genero',
                title="Tempo Médio de Espera por Idade e Gênero",
                labels={'tempo_espera_medio': 'Tempo de Espera (min)'},
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            fig_wait_gender.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_wait_gender, use_container_width=True)

    with col2:
        with st.container(border=True):
            interval_analysis = df_population[df_population['media_dias_entre_consultas'] > 0].groupby(['categoria_idade', 'genero'])['media_dias_entre_consultas'].mean().reset_index()
            
            fig_interval_gender = px.bar(
                interval_analysis,
                x='categoria_idade',
                y='media_dias_entre_consultas',
                color='genero',
                title="Média de Dias entre Consultas por Idade e Gênero",
                labels={'media_dias_entre_consultas': 'Dias entre Consultas'},
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            fig_interval_gender.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_interval_gender, use_container_width=True)

   
    st.markdown('<h4 class="section-header">Distribuições Detalhadas</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            wait_time_box_data = df_population[df_population['tempo_espera_medio'] > 0]
            fig_box_wait = px.box(
                wait_time_box_data,
                x='categoria_idade',
                y='tempo_espera_medio',
                title="Distribuição do Tempo de Espera por Categoria de Idade",
                labels={'tempo_espera_medio': 'Tempo de Espera (min)'}
            )
            fig_box_wait.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box_wait, use_container_width=True)

    with col2:
        with st.container(border=True):
            interval_box_data = df_population[df_population['media_dias_entre_consultas'] > 0]
            fig_box_interval = px.box(
                interval_box_data,
                x='categoria_idade',
                y='media_dias_entre_consultas',
                title="Distribuição do Intervalo entre Consultas por Categoria de Idade",
                labels={'media_dias_entre_consultas': 'Dias entre Consultas'}
            )
            fig_box_interval.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box_interval, use_container_width=True)

  
    st.markdown('<h4 class="section-header">Correlação entre Tempo de Espera e Intervalo entre Consultas</h4>', unsafe_allow_html=True)

    # Filter data for meaningful visualization
    scatter_data = df_population[
        (df_population['tempo_espera_medio'] > 0) & 
        (df_population['media_dias_entre_consultas'] > 0) &
        (df_population['tempo_espera_medio'] < df_population['tempo_espera_medio'].quantile(0.95)) &
        (df_population['media_dias_entre_consultas'] < df_population['media_dias_entre_consultas'].quantile(0.95))
    ]

    fig_scatter = px.scatter(
        scatter_data,
        x='media_dias_entre_consultas',
        y='tempo_espera_medio',
        color='genero',
        size='total_agendamentos',
        title="Relação entre Intervalo de Consultas e Tempo de Espera",
        labels={
            'media_dias_entre_consultas': 'Média de Dias entre Consultas',
            'tempo_espera_medio': 'Tempo Médio de Espera (min)',
            'total_agendamentos': 'Total de Agendamentos'
        },
        color_discrete_map={
            'feminino': '#ff69b4',
            'masculino': '#4169e1',
            'desconhecido': '#808080'
        },
        hover_data=['PACIENTE', 'idade', 'categoria_idade']
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Additional insights
    st.markdown('<h4 class="section-header">Insights Principais</h4>', unsafe_allow_html=True)

    # Calculate key insights
    avg_wait_time = df_population[df_population['tempo_espera_medio'] > 0]['tempo_espera_medio'].mean()
    avg_interval = df_population[df_population['media_dias_entre_consultas'] > 0]['media_dias_entre_consultas'].mean()

    # Patients with long wait times
    long_wait_patients = df_population[df_population['tempo_espera_medio'] > 60].shape[0]
    long_wait_pct = (long_wait_patients / len(df_population)) * 100

    # Patients with frequent consultations
    frequent_patients = df_population[df_population['media_dias_entre_consultas'] < 30].shape[0]
    frequent_pct = (frequent_patients / len(df_population)) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Tempo Médio de Espera Geral",
            f"{avg_wait_time:.1f} min",
            help="Média geral do tempo de espera entre chegada e atendimento"
        )
    with col2:
        st.metric(
            "Intervalo Médio entre Consultas",
            f"{avg_interval:.1f} dias",
            help="Média geral de dias entre consultas consecutivas"
        )
    with col3:
        st.metric(
            "Pacientes com Espera > 1h",
            f"{long_wait_pct:.1f}%",
            f"{long_wait_patients} pacientes",
            help="Percentual de pacientes com tempo médio de espera superior a 60 minutos"
        )
    with col4:
        st.metric(
            "Consultas Frequentes (< 30 dias)",
            f"{frequent_pct:.1f}%",
            f"{frequent_patients} pacientes",
            help="Percentual de pacientes com intervalo médio menor que 30 dias entre consultas"
        )
    #aqui
    st.markdown('---')
    st.markdown('<h3 class="section-header">Análises Detalhadas</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab4, tab5 = st.tabs([
        "Alto Risco", 
        "Super Usuários", 
        "Análise Gênero",
        "Dados Completos"
    ])
    
    with tab1:
        high_risk = df_clustered[df_clustered['categoria_risco_no_show'] == 'ALTO_RISCO']
        if not high_risk.empty:
            st.markdown(f"**{len(high_risk)} pacientes de alto risco identificados:**")
            
            # High risk by gender
            risk_gender = high_risk['genero'].value_counts()
            
            
            st.dataframe(
                high_risk[['PACIENTE', 'genero', 'idade', 'total_agendamentos', 'taxa_no_show', 'especialidade_principal']].sort_values('taxa_no_show',ascending=False).head(20),
                use_container_width=True
            )
        else:
            st.success("✅ Nenhum paciente de alto risco identificado!")
    
    with tab2:
        super_users = df_clustered[df_clustered['total_agendamentos'] >= 15].sort_values('total_agendamentos', ascending=False)
        if not super_users.empty:
            st.markdown(f"**{len(super_users)} super usuários identificados:**")
            
            # Super users by gender
            super_gender = super_users['genero'].value_counts()
            st.markdown("**Super Usuários por Gênero:**")
            for gender, count in super_gender.items():
                st.metric(f"{gender.title()}", count)
            
            st.dataframe(
                super_users[['PACIENTE', 'genero', 'idade', 'total_agendamentos', 'especialidades_diferentes', 'taxa_no_show']].head(20),
                use_container_width=True
            )
        else:
            st.info("Nenhum super usuário identificado.")
    
  
    with tab4:
        st.markdown("**Análise Detalhada por Gênero:**")
        
        # Gender comparison metrics
        gender_comparison = df_clustered.groupby('genero').agg({
            'idade': ['mean', 'std'],
            'total_agendamentos': ['mean', 'std'], 
            'taxa_no_show': ['mean', 'std'],
            'especialidades_diferentes': 'mean',
            'tempo_espera_medio': 'mean',
            'intensidade_uso': 'mean'
        }).round(2)
        
        st.dataframe(gender_comparison, use_container_width=True)
        
        # Gender behavior patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot for appointments by gender
            fig_box_appointments = px.box(
                df_clustered,
                x='genero',
                y='total_agendamentos',
                title="Distribuição de Consultas por Gênero",
                color='genero',
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            st.plotly_chart(fig_box_appointments, use_container_width=True)
        
        with col2:
            # Box plot for no-show rates by gender
            fig_box_noshow = px.box(
                df_clustered,
                x='genero',
                y='taxa_no_show',
                title="Distribuição de Taxa No-Show por Gênero",
                color='genero',
                color_discrete_map={
                    'feminino': '#ff69b4',
                    'masculino': '#4169e1',
                    'desconhecido': '#808080'
                }
            )
            st.plotly_chart(fig_box_noshow, use_container_width=True)
        
        # Top specialties by gender
        
    
    with tab5:
        st.markdown("**Dataset completo com análise de gênero brasileira:**")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", len(df_clustered))
        with col2:
            accuracy = len(df_clustered[df_clustered['genero'] != 'desconhecido']) / len(df_clustered) * 100
            st.metric("Precisão Gênero", f"{accuracy:.1f}%")
        with col3:
            clusters = df_clustered['cluster'].nunique()
            st.metric("Clusters Identificados", clusters)
        
        # Display full dataset
        st.dataframe(df_clustered, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Baixar dados completos (CSV)",
                data=csv,
                file_name=f"analise_populacional_completa_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export only gender classification
            gender_export = df_clustered[['PACIENTE', 'genero']].copy()
            gender_csv = gender_export.to_csv(index=False)
            st.download_button(
                label="👥 Baixar classificação de gênero (CSV)",
                data=gender_csv,
                file_name=f"classificacao_genero_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    # Header principal
    st.markdown('<h1 class="main-header">Sistema de Detecção de Fraudes - Unimed</h1>', unsafe_allow_html=True)
    
    # Sidebar para filtros globais
    with st.sidebar:
               
     
        st.markdown("---")
   
    
    # Sistema de abas
    tabs = st.tabs([
        "Perfil de Pacientes", 
        "Divergências SGU/PEP", 
        "Conflitos de Horário", 
        "Intervalos Suspeitos", 
        "Análise de No-Show", 
        "Análise com IA"
    ])
    
    with tabs[5]:
        render_overview_tab()
    
    with tabs[1]:
        render_divergencias_tab()
    
    with tabs[2]:
        render_conflitos_tab()
    
    with tabs[3]:
        render_intervalos_tab()
    
    with tabs[4]:
        render_noshow_tab()
    
    with tabs[0]:
        render_pacientes_tab()

if __name__ == "__main__":
    main()
