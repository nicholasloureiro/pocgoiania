import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from plotly.subplots import make_subplots

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
    """Renderiza a aba de visão geral"""
    st.markdown('<div class="tab-header">📊 Panorama Executivo</div>', unsafe_allow_html=True)
    
    # Story introduction
    st.markdown("""
    <div class="story-box">
    <strong>História dos Dados:</strong> Esta análise examina padrões suspeitos nos sistemas de atendimento da Unimed, 
    cruzando dados entre SGU (faturamento) e PEP (agendamentos) para identificar possíveis inconsistências, 
    fraudes ou ineficiências operacionais que podem estar custando milhares de reais mensalmente.
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principais
    metrics = get_overview_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Total de Agendamentos",
            value=f"{metrics['total_atendimentos']:,}".replace(',', '.'),
            help="Total de registros na base PEP"
        )
    
    with col2:
        st.metric(
            label="❌ Taxa de No-Show",
            value=f"{metrics['taxa_noshow']}%",
            delta=f"-{metrics['taxa_noshow']}%" if metrics['taxa_noshow'] > 20 else None,
            delta_color="inverse",
            help="Percentual de não comparecimentos"
        )
    
    with col3:
        st.metric(
            label="⚠️ Divergências SGU x PEP",
            value=f"{metrics['divergencias']:,}".replace(',', '.'),
            delta=f"R$ {metrics['valor_divergencias']:,.0f}".replace(',', '.') if metrics['valor_divergencias'] else None,
            delta_color="inverse",
            help="Casos onde SGU executou mas PEP não atendeu"
        )
    
    with col4:
        st.metric(
            label="🚨 Conflitos de Horário",
            value=f"{metrics['conflitos']:,}".replace(',', '.'),
            help="Profissionais atendendo no mesmo horário"
        )
    
    # Alertas críticos
    if metrics['valor_divergencias'] and metrics['valor_divergencias'] > 50000:
        st.markdown(f"""
        <div class="alert-critical">
        🚨 <strong>ALERTA CRÍTICO:</strong> Foram identificadas divergências no valor de R$ {metrics['valor_divergencias']:,.2f} 
        entre os sistemas SGU e PEP. Isso representa um risco financeiro significativo que requer investigação imediata.
        </div>
        """, unsafe_allow_html=True)
    
    if metrics['taxa_noshow'] > 25:
        st.markdown(f"""
        <div class="alert-high">
        <strong>Taxa de No-Show Elevada:</strong> A taxa atual de {metrics['taxa_noshow']}% está acima do recomendado (15-20%). 
        Isso pode indicar problemas operacionais ou comportamentos suspeitos.
        </div>
        """, unsafe_allow_html=True)
    
    # Key findings summary
    st.markdown('<h3 class="section-header">🎯 Principais Descobertas</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="key-finding">
        <strong>💰 Impacto Financeiro:</strong><br>
        • Divergências identificadas representam potencial perda de receita<br>
        • Conflitos de horário podem indicar superfaturamento<br>
        • No-shows excessivos impactam eficiência operacional
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="key-finding">
        <strong>🔍 Padrões Identificados:</strong><br>
        • Intervalos suspeitos entre atendimentos<br>
        • Profissionais com comportamentos atípicos<br>
        • Inconsistências sistemáticas entre sistemas
        </div>
        """, unsafe_allow_html=True)
    
    # Recomendações
    st.markdown('<h3 class="section-header">📋 Próximos Passos Recomendados</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **Ações Imediatas (1-7 dias):**
    1. 🔍 Investigar casos de maior valor nas divergências SGU x PEP
    2. 👥 Auditar profissionais com conflitos de horário frequentes
    3. 📞 Implementar verificação adicional para intervalos < 30min entre atendimentos
    
    **Ações de Médio Prazo (1-4 semanas):**
    1. 🤖 Automatizar alertas para padrões suspeitos identificados
    2. 📊 Estabelecer KPIs de monitoramento contínuo
    3. 🔗 Melhorar integração entre sistemas SGU e PEP
    """)

def render_divergencias_tab():
    """Renderiza a aba de divergências SGU x PEP"""
    st.markdown('<div class="tab-header">⚖️ Divergências SGU x PEP</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>O Problema:</strong> O SGU (sistema de faturamento) registra procedimentos executados que não aparecem 
    como atendidos no PEP (sistema de agendamentos). Isso pode indicar faturamento de procedimentos não realizados, 
    falhas de integração ou inconsistências nos registros.
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Valor Total", f"R$ {valor_total:,.2f}".replace(',', '.'))
        with col2:
            st.metric("📊 Total de Casos", f"{len(df_divergencias):,}".replace(',', '.'))
        with col3:
            st.metric("🚨 Casos Críticos", f"{casos_criticos:,}".replace(',', '.'))
        
        # Análise por tipo
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de divergências por tipo
            div_by_type = df_divergencias.groupby(['situacao', 'origem']).agg({
                'valor': 'sum',
                'carteirinha': 'count'
            }).reset_index()
            div_by_type.columns = ['situacao', 'origem', 'valor_total', 'quantidade']
            
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
            # Top prestadores com divergências
            top_prestadores = df_divergencias.groupby('prestador')['valor'].sum().nlargest(5)
            
            st.markdown("### 🏥 Top Prestadores")
            for prestador, valor in top_prestadores.items():
                st.markdown(f"**{prestador}**: R$ {valor:,.2f}".replace(',', '.'))
        
        # Casos mais críticos
        st.markdown('<h3 class="section-header">🚨 Casos Mais Críticos</h3>', unsafe_allow_html=True)
        
        casos_criticos_df = df_divergencias[
            df_divergencias['situacao'] == 'EXECUTADO_SGU_NAO_ATENDIDO_PEP'
        ].nlargest(10, 'valor')
        
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
    st.markdown('<div class="tab-header">⏰ Conflitos de Horário</div>', unsafe_allow_html=True)
    
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
       -- WHERE "SITUAÇÃO" = 'atendida'
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
            st.metric("🚨 Total de Conflitos", f"{total_conflitos:,}".replace(',', '.'))
        with col2:
            st.metric("👥 Profissionais Envolvidos", f"{profissionais_conflito:,}".replace(',', '.'))
        with col3:
            st.metric("📊 Máx. Pacientes Simultâneos", f"{max_conflitos}")
        
        # Análise temporal
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Conflitos por dia
            df_diario = df_conflitos.groupby('data').size().reset_index(name='conflitos')
            df_diario = df_diario.sort_values("data").reset_index(drop=True)
            
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
            # Conflitos por dia da semana
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
        st.markdown('<h3 class="section-header">👨‍⚕️ Profissionais com Conflitos</h3>', unsafe_allow_html=True)
        
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
            <strong>⚠️ Profissionais com Padrão Crítico:</strong> Os profissionais abaixo apresentam 
            um número elevado de conflitos que merece investigação.
            </div>
            """, unsafe_allow_html=True)
            
            for _, row in casos_criticos.head(5).iterrows():
                st.markdown(f"• **{row['profissional']}**: {row['total_conflitos']} conflitos em {row['dias_com_conflito']} dias")
        
        # Tabela detalhada dos piores casos
        with st.expander("Ver Casos Mais Graves (1+ pacientes simultâneos)"):
            casos_graves = df_conflitos[df_conflitos['qtd_conflitos'] >= 0]
            if not casos_graves.empty:
                st.dataframe(casos_graves, use_container_width=True)
            else:
                st.info("Nenhum caso com 4+ pacientes simultâneos encontrado.")
    
    else:
        st.success("✅ Nenhum conflito de horário detectado!")

def render_intervalos_tab():
    """Renderiza a aba de intervalos suspeitos"""
    st.markdown('<div class="tab-header">⏱️ Intervalos Suspeitos</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Tempo é Dinheiro:</strong> Intervalos muito curtos entre atendimentos (< 30 minutos) podem indicar 
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
            st.metric("🚨 Críticos (≤15 min)", f"{criticos:,}".replace(',', '.'))
        with col2:
            st.metric("⚠️ Suspeitos (16-30 min)", f"{suspeitos:,}".replace(',', '.'))
        with col3:
            st.metric("❓ Questionáveis (31-45 min)", f"{questionaveis:,}".replace(',', '.'))
        
        # Distribuição dos intervalos
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Box plot dos intervalos por profissional
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
                title="Distribuição dos Intervalos entre Atendimentos por Profissional",
                labels={"intervalo_minutos": "Intervalo (minutos)"}
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Contagem por classificação
            class_count = df_intervalos['classificacao'].value_counts()
            
            fig_pie = px.pie(
                values=class_count.values,
                names=class_count.index,
                title="Distribuição por Classificação",
                color_discrete_map={
                    'CRÍTICO': '#dc2626',
                    'SUSPEITO': '#f59e0b',
                    'QUESTIONÁVEL': '#eab308'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Profissionais com mais casos suspeitos
        st.markdown('<h3 class="section-header">👨‍⚕️ Ranking de Risco por Profissional</h3>', unsafe_allow_html=True)
        
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
            st.markdown("### 🎯 Score de Risco")
            st.caption("Crítico: 3pts | Suspeito: 2pts | Questionável: 1pt")
            
            for prof, score in top_risk['score_risco'].head(5).items():
                criticos = top_risk.loc[prof, 'CRÍTICO'] if 'CRÍTICO' in top_risk.columns else 0
                st.markdown(f"**{prof}**: {score} pts")
                if criticos > 0:
                    st.markdown(f"<span style='color: red'>⚠️ {criticos} casos críticos</span>", unsafe_allow_html=True)
        
        # Casos mais críticos para investigação
        casos_criticos = df_intervalos[df_intervalos['classificacao'] == 'CRÍTICO'].head(20)
        
        if not casos_criticos.empty:
            st.markdown('<h3 class="section-header">🚨 Casos Críticos para Investigação Imediata</h3>', unsafe_allow_html=True)
            
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
    st.markdown('<div class="tab-header">❌ Análise de No-Show</div>', unsafe_allow_html=True)
    
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
            st.metric("📊 Taxa Média", f"{media_taxa:.1f}%")
        with col2:
            st.metric("🚨 Casos Críticos (≥50%)", f"{casos_criticos}")
        with col3:
            st.metric("🤔 Zero No-Show", f"{casos_zero}")
        with col4:
            st.metric("👥 Total Profissionais", f"{len(df_noshow)}")
        
        # Análise de distribuição
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot: agendamentos vs taxa de no-show
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
            class_dist = df_noshow['classificacao_risco'].value_counts()
            
            fig_class = px.pie(
                values=class_dist.values,
                names=class_dist.index,
                title="Distribuição por Risco",
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
            st.plotly_chart(fig_class, use_container_width=True)
        
        # Casos que requerem atenção
        st.markdown('<h3 class="section-header">🎯 Casos que Requerem Atenção</h3>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🚨 Críticos", "🤔 Suspeitos", "📊 Outliers"])
        
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
                st.success("✅ Nenhum caso crítico encontrado!")
        
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
                        st.subheader("📈 Outliers Altos")
                        if not outliers_altos.empty:
                            st.dataframe(
                                outliers_altos[['medico', 'taxa_no_show_percent']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("Nenhum outlier alto detectado.")
                    
                    with col2:
                        st.subheader("📉 Outliers Baixos")
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
                WHEN m.idade < 18 THEN 'CRIANCA_ADOLESCENTE'
                WHEN m.idade < 35 THEN 'JOVEM_ADULTO'
                WHEN m.idade < 50 THEN 'ADULTO'
                WHEN m.idade < 65 THEN 'ADULTO_MADURO'
                ELSE 'IDOSO'
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
    st.markdown('<div class="tab-header">👥 Análise de Pacientes</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="story-box">
    <strong>Análise Populacional Brasileira:</strong> Sistema inteligente de análise de pacientes com 
    detecção de gênero baseada em nomes brasileiros reais do seu banco de dados e clustering avançado.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Brazilian gender classifier
    classifier_data = create_brazilian_gender_classifier()
    
    # Option to analyze name patterns from database
    with st.expander("🔍 Analisar Padrões de Nomes do Banco de Dados"):
        if st.button("Executar Análise de Nomes"):
            with st.spinner('Analisando nomes únicos do banco de dados...'):
                try:
                    df_names = execute_query(get_name_analysis_query())
                    if not df_names.empty:
                        st.markdown("**Top 50 Primeiros Nomes Mais Frequentes:**")
                        st.dataframe(df_names.head(50), use_container_width=True)
                        
                        # Show gender classification for top names
                        df_names['genero_detectado'] = df_names['primeiro_nome'].apply(
                            lambda x: detect_gender_brazilian(x, classifier_data)
                        )
                        
                        gender_summary = df_names['genero_detectado'].value_counts()
                        st.markdown("**Classificação de Gênero dos Nomes:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("👩 Feminino", gender_summary.get('feminino', 0))
                        with col2:
                            st.metric("👨 Masculino", gender_summary.get('masculino', 0))
                        with col3:
                            st.metric("❓ Desconhecido", gender_summary.get('desconhecido', 0))
                        with col4:
                            st.metric("📊 Total Nomes", len(df_names))
                            
                except Exception as e:
                    st.error(f"Erro ao analisar nomes: {str(e)}")
    
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
    st.markdown('<h3 class="section-header">📊 Visão Geral da População</h3>', unsafe_allow_html=True)
    
    # Key metrics
    total_patients = len(df_population)
    avg_age = df_population['idade'].mean()
    avg_appointments = df_population['total_agendamentos'].mean()
    overall_no_show_rate = df_population['taxa_no_show'].mean()
    gender_accuracy = len(df_population[df_population['genero'] != 'desconhecido']) / total_patients * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Pacientes", f"{total_patients:,}".replace(',', '.'))
    with col2:
        st.metric("🎂 Idade Média", f"{avg_age:.1f} anos")
    with col3:
        st.metric("📅 Consultas Médias", f"{avg_appointments:.1f}")
    with col4:
        st.metric("🚫 Taxa No-Show Média", f"{overall_no_show_rate:.1f}%")
    with col5:
        st.metric("🎯 Precisão Gênero", f"{gender_accuracy:.1f}%")
    
    # Gender Analysis
    st.markdown('<h3 class="section-header">🧬 Análise de Gênero (Brasileira)</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        gender_dist = df_population['genero'].value_counts()
        fig_gender = px.pie(
            values=gender_dist.values,
            names=gender_dist.index,
            title="Distribuição por Gênero",
            color_discrete_map={
                'feminino': '#ff69b4',
                'masculino': '#4169e1',
                'desconhecido': '#808080'
            }
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        
        # Gender stats
        st.markdown("**Estatísticas por Gênero:**")
        gender_stats = df_population.groupby('genero').agg({
            'idade': 'mean',
            'total_agendamentos': 'mean',
            'taxa_no_show': 'mean'
        }).round(2)
        st.dataframe(gender_stats, use_container_width=True)
    
    with col2:
        # Age distribution by gender
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
    st.markdown('<h3 class="section-header">🎯 Análise Comportamental por Gênero</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # No-show rate by gender and age group
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
        # Average appointments by gender and age
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
    st.markdown('<h3 class="section-header">🏥 Análise por Especialidade e Gênero</h3>', unsafe_allow_html=True)
    
    specialty_gender = df_population.groupby(['especialidade_principal', 'genero']).size().reset_index(name='count')
    top_specialties = df_population['especialidade_principal'].value_counts().head(10).index
    specialty_gender_filtered = specialty_gender[specialty_gender['especialidade_principal'].isin(top_specialties)]
    
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
        }
    )
    fig_specialty_gender.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_specialty_gender, use_container_width=True)

    #aqui
    # Add this section after the "Análise por Especialidade e Gênero" section in render_pacientes_tab()

    # Wait Time and Consultation Interval Analysis
    st.markdown('<h3 class="section-header">⏱️ Análise de Tempos de Espera e Intervalos entre Consultas</h3>', unsafe_allow_html=True)

    # Create two columns for the charts
    col1, col2 = st.columns(2)

    with col1:
        # Distribution of average waiting time
        # Filter out zero values for better visualization
        wait_time_data = df_population[df_population['tempo_espera_medio'] > 0]['tempo_espera_medio']
        
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
        
        # Summary statistics for wait time
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
        # Filter out zero values for better visualization
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
    st.markdown('<h4 class="section-header">📊 Análise Comparativa por Gênero e Idade</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Average wait time by gender and age category
        wait_time_analysis = df_population[df_population['tempo_espera_medio'] > 0].groupby(['categoria_idade', 'genero'])['tempo_espera_medio'].mean().reset_index()
        
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
        # Average days between consultations by gender and age category
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

    # Box plots for better understanding of distributions
    st.markdown('<h4 class="section-header">📈 Distribuições Detalhadas</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Box plot for wait times by age category
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
        # Box plot for consultation intervals by age category
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

    # Scatter plot to show correlation
    st.markdown('<h4 class="section-header">🔗 Correlação entre Tempo de Espera e Intervalo entre Consultas</h4>', unsafe_allow_html=True)

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
    st.markdown('<h4 class="section-header">💡 Insights Principais</h4>', unsafe_allow_html=True)

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
            "⏱️ Tempo Médio de Espera Geral",
            f"{avg_wait_time:.1f} min",
            help="Média geral do tempo de espera entre chegada e atendimento"
        )
    with col2:
        st.metric(
            "📅 Intervalo Médio entre Consultas",
            f"{avg_interval:.1f} dias",
            help="Média geral de dias entre consultas consecutivas"
        )
    with col3:
        st.metric(
            "⏰ Pacientes com Espera > 1h",
            f"{long_wait_pct:.1f}%",
            f"{long_wait_patients} pacientes",
            help="Percentual de pacientes com tempo médio de espera superior a 60 minutos"
        )
    with col4:
        st.metric(
            "🔄 Consultas Frequentes (< 30 dias)",
            f"{frequent_pct:.1f}%",
            f"{frequent_patients} pacientes",
            help="Percentual de pacientes com intervalo médio menor que 30 dias entre consultas"
        )
    #aqui
    # Detailed Analysis Tabs
    st.markdown('<h3 class="section-header">🔍 Análises Detalhadas</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alto Risco", 
        "📈 Super Usuários", 
        "🎯 Clusters",
        "👥 Análise Gênero",
        "📋 Dados Completos"
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
    
    with tab3:
        selected_cluster = st.selectbox("Selecione um cluster para análise:", 
                                      sorted(df_clustered['cluster'].unique()))
        
        cluster_data = df_clustered[df_clustered['cluster'] == selected_cluster]
        
        st.markdown(f"**Cluster {selected_cluster} - {len(cluster_data)} pacientes:**")
        
        # Cluster characteristics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Idade Média", f"{cluster_data['idade'].mean():.1f} anos")
        with col2:
            st.metric("Consultas Médias", f"{cluster_data['total_agendamentos'].mean():.1f}")
        with col3:
            st.metric("Taxa No-Show", f"{cluster_data['taxa_no_show'].mean():.1f}%")
        with col4:
            main_gender = cluster_data['genero'].mode()[0] if not cluster_data['genero'].empty else 'N/A'
            st.metric("Gênero Predominante", main_gender.title())
        
        # Gender distribution in cluster
        cluster_gender_dist = cluster_data['genero'].value_counts()
        fig_cluster_gender = px.pie(
            values=cluster_gender_dist.values,
            names=cluster_gender_dist.index,
            title=f"Distribuição de Gênero - Cluster {selected_cluster}",
            color_discrete_map={
                'feminino': '#ff69b4',
                'masculino': '#4169e1',
                'desconhecido': '#808080'
            }
        )
        st.plotly_chart(fig_cluster_gender, use_container_width=True)
        
        # Sample patients from cluster
        st.dataframe(
            cluster_data[['PACIENTE', 'genero', 'idade', 'total_agendamentos', 'taxa_no_show', 'especialidade_principal']].head(10),
            use_container_width=True
        )
    
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
            st.metric("📊 Total de Registros", len(df_clustered))
        with col2:
            accuracy = len(df_clustered[df_clustered['genero'] != 'desconhecido']) / len(df_clustered) * 100
            st.metric("🎯 Precisão Gênero", f"{accuracy:.1f}%")
        with col3:
            clusters = df_clustered['cluster'].nunique()
            st.metric("🎯 Clusters Identificados", clusters)
        
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
    st.markdown('<h1 class="main-header">🔍 Sistema de Detecção de Fraudes - Unimed</h1>', unsafe_allow_html=True)
    
    # Sidebar para filtros globais
    with st.sidebar:
        st.header("🎛️ Filtros Globais")
        
        # Filtro de período
        periodo = st.selectbox(
            "📅 Período de Análise",
            ["Últimos 3 meses", "Outubro 2024", "Novembro 2024", "Dezembro 2024", "Período customizado"]
        )
        
        if periodo == "Período customizado":
            data_inicio = st.date_input("Data inicial")
            data_fim = st.date_input("Data final")
        
        
        
        nivel_risco = st.selectbox(
            "⚠️ Nível de Risco",
            ["Todos", "Crítico", "Alto", "Médio", "Baixo"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Legendas")
        st.markdown("""
        🚨 **Crítico**: Requer ação imediata  
        ⚠️ **Alto**: Investigação prioritária  
        ❓ **Médio**: Monitoramento necessário  
        ✅ **Baixo**: Dentro da normalidade
        """)
    
    # Sistema de abas
    tabs = st.tabs([
        "📊 Visão Geral", 
        "⚖️ Divergências SGU/PEP", 
        "⏰ Conflitos de Horário", 
        "⏱️ Intervalos Suspeitos", 
        "❌ Análise de No-Show", 
        "👥 Perfil de Pacientes"
    ])
    
    with tabs[0]:
        render_overview_tab()
    
    with tabs[1]:
        render_divergencias_tab()
    
    with tabs[2]:
        render_conflitos_tab()
    
    with tabs[3]:
        render_intervalos_tab()
    
    with tabs[4]:
        render_noshow_tab()
    
    with tabs[5]:
        render_pacientes_tab()

if __name__ == "__main__":
    main()
