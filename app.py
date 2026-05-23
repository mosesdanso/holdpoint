import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from src.ingest import load_schedule, validate_columns
from src.features import engineer_features

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Holdpoint",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0d0d;
    color: #e8e0d0;
}

/* Main container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Header */
.hp-header {
    border-bottom: 1px solid #f5a623;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}

.hp-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f5a623;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}

.hp-tagline {
    font-family: 'Barlow', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: #888;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Section headers */
.hp-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #f5a623;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-left: 3px solid #f5a623;
    padding-left: 12px;
    margin-bottom: 1.2rem;
    margin-top: 2rem;
}

/* Health banner */
.hp-banner {
    padding: 1.2rem 1.5rem;
    border-radius: 2px;
    margin-bottom: 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.3px;
}

.hp-banner-red {
    background-color: #2a0a0a;
    border-left: 4px solid #e74c3c;
    color: #f5a5a5;
}

.hp-banner-amber {
    background-color: #1f1500;
    border-left: 4px solid #f5a623;
    color: #f5d78a;
}

.hp-banner-green {
    background-color: #051a0a;
    border-left: 4px solid #2ecc71;
    color: #8af5b5;
}

/* Metric cards */
.hp-metric {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-top: 3px solid #f5a623;
    padding: 1.2rem;
    border-radius: 2px;
    text-align: center;
}

.hp-metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.hp-metric-label {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
}

.metric-red { color: #e74c3c; }
.metric-amber { color: #f5a623; }
.metric-green { color: #2ecc71; }
.metric-white { color: #e8e0d0; }

/* Interpretation bullets */
.hp-insight {
    background: #151515;
    border: 1px solid #222;
    border-left: 3px solid #f5a623;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-radius: 0 2px 2px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Upload zone */
.uploadedFile {
    background: #1a1a1a !important;
    border: 1px dashed #333 !important;
    border-radius: 2px !important;
}

/* Divider */
.hp-divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

/* Dataframe */
.dataframe {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="hp-header">
    <div class="hp-title">HOLDPOINT</div>
    <div class="hp-tagline">Flag the risk. Before the delay.</div>
</div>
""", unsafe_allow_html=True)

# --- UPLOAD ---
st.markdown('<div class="hp-section">Upload Schedule</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload your project schedule — CSV or Excel",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df, features = engineer_features(df)

        model_path = 'models/holdpoint_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            df['risk_score'] = model.predict_proba(df[features])[:, 1]
        else:
            df['risk_score'] = (
                (df['start_delay'].clip(0) * 0.5) +
                (df['planned_duration'] / (df['planned_duration'].max() + 1) * 0.3) +
                ((1 - df['float_days'] / (df['float_days'].max() + 1)) * 0.1) +
                (df['is_critical'] * 0.1)
            ).clip(0, 1)

        def risk_label(score):
            if score >= 0.7: return 'High'
            elif score >= 0.4: return 'Medium'
            return 'Low'

        df['risk_level'] = df['risk_score'].apply(risk_label)
        high = df[df['risk_level'] == 'High']
        medium = df[df['risk_level'] == 'Medium']
        low = df[df['risk_level'] == 'Low']
        total = len(df)
        avg_start_delay = df['start_delay'].mean()
        critical_delayed = df[(df['is_critical'] == 1) & (df['risk_level'] == 'High')]

        # --- EXECUTIVE SUMMARY ---
        st.markdown('<div class="hp-section">Programme Health</div>', unsafe_allow_html=True)

        if len(high) == 0:
            banner_class = "hp-banner-green"
            banner_text = "✓ Programme is in good health. No tasks are currently at high risk of delay."
        elif len(high) <= total * 0.2:
            banner_class = "hp-banner-amber"
            banner_text = "⚠ Isolated risk detected. A small number of tasks require immediate attention."
        else:
            banner_class = "hp-banner-red"
            banner_text = "✕ Programme under significant stress. Multiple tasks are at high risk of delay."

        st.markdown(f'<div class="hp-banner {banner_class}">{banner_text}</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="hp-metric"><div class="hp-metric-value metric-white">{total}</div><div class="hp-metric-label">Total Tasks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="hp-metric"><div class="hp-metric-value metric-red">{len(high)}</div><div class="hp-metric-label">High Risk</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="hp-metric"><div class="hp-metric-value metric-amber">{len(medium)}</div><div class="hp-metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="hp-metric"><div class="hp-metric-value metric-green">{len(low)}</div><div class="hp-metric-label">Low Risk</div></div>', unsafe_allow_html=True)

        st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

        # --- INTERPRETATION ---
        st.markdown('<div class="hp-section">What the data is telling you</div>', unsafe_allow_html=True)

        interpretations = []

        if len(critical_delayed) > 0:
            tasks = ', '.join(critical_delayed['task_name'].tolist())
            interpretations.append(f"<strong>Critical path at risk:</strong> {tasks} {'are' if len(critical_delayed) > 1 else 'is'} on the critical path and flagged high risk. Any further delay here directly extends your project end date.")

        if avg_start_delay > 2:
            interpretations.append(f"<strong>Start delay pattern:</strong> Tasks are starting an average of {avg_start_delay:.1f} days late. This is systemic — likely resource availability, procurement lag, or predecessor dependency failures.")

        if len(high) > total * 0.3:
            interpretations.append(f"<strong>Programme-wide stress:</strong> Over 30% of tasks are high risk. The baseline schedule may be too aggressive or resource allocation is insufficient across the board.")

        tasks_no_float = df[(df['float_days'] == 0) & (df['risk_level'] != 'Low')]
        if len(tasks_no_float) > 0:
            interpretations.append(f"<strong>Zero float buffer:</strong> {len(tasks_no_float)} task(s) have no float and elevated risk. These cannot absorb any further delay.")

        if not interpretations:
            interpretations.append("No major systemic issues detected. Monitor medium risk tasks to prevent escalation.")

        for point in interpretations:
            st.markdown(f'<div class="hp-insight">{point}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

        # --- MITIGATION PATH ---
        st.markdown('<div class="hp-section">Recommended Mitigation Path</div>', unsafe_allow_html=True)

        recommendations = []

        if len(critical_delayed) > 0:
            for _, row in critical_delayed.iterrows():
                recommendations.append({
                    'Priority': '1 — Immediate',
                    'Task': row['task_name'],
                    'Action': 'Crash this activity — add resource or extend working hours to recover lost time. Review all predecessor tasks.',
                    'Reason': 'Critical path — high delay risk'
                })

        non_critical_high = high[high['is_critical'] == 0]
        for _, row in non_critical_high.iterrows():
            recommendations.append({
                'Priority': '2 — This week',
                'Task': row['task_name'],
                'Action': f"Investigate start delay of {int(row['start_delay'])} days. Confirm resource assignment and resolve predecessor blockers.",
                'Reason': 'High risk — may become critical if unresolved'
            })

        for _, row in medium.iterrows():
            recommendations.append({
                'Priority': '3 — Monitor',
                'Task': row['task_name'],
                'Action': 'Add to weekly look-ahead. Flag for early warning if start delay increases.',
                'Reason': 'Medium risk — manageable if caught now'
            })

        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.success("No immediate actions required. Continue monitoring.")

        st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

        # --- CHART ---
        st.markdown('<div class="hp-section">Task Risk Scores</div>', unsafe_allow_html=True)

        fig = px.bar(
            df.sort_values('risk_score', ascending=False),
            x='task_name',
            y='risk_score',
            color='risk_level',
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f5a623', 'Low': '#2ecc71'},
            labels={'risk_score': 'Risk Score', 'task_name': 'Task'}
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#0d0d0d',
            font=dict(family='Barlow, sans-serif', color='#888', size=12),
            xaxis=dict(tickangle=-45, gridcolor='#1a1a1a', linecolor='#222'),
            yaxis=dict(gridcolor='#1a1a1a', linecolor='#222', range=[0, 1]),
            legend=dict(bgcolor='#111', bordercolor='#222', borderwidth=1),
            margin=dict(t=20, b=80),
            showlegend=True
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

        # --- FULL TABLE ---
        st.markdown('<div class="hp-section">Full Task Detail</div>', unsafe_allow_html=True)
        st.dataframe(
            df[['task_name', 'planned_duration', 'start_delay', 'float_days', 'is_critical', 'risk_score', 'risk_level']]
            .sort_values('risk_score', ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.markdown("""
    <div style="border: 1px dashed #2a2a2a; padding: 3rem; text-align: center; border-radius: 2px; margin-top: 1rem;">
        <div style="font-family: 'Space Mono', monospace; color: #333; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;">
            Upload a project schedule to begin
        </div>
        <div style="color: #2a2a2a; font-size: 0.8rem; margin-top: 0.5rem;">
            CSV or Excel — use the sample file in /data to test
        </div>
    </div>
    """, unsafe_allow_html=True)