import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
from src.ingest import load_schedule, validate_columns
from src.features import engineer_features

st.set_page_config(page_title="Holdpoint", page_icon="🚧", layout="wide")

st.title("🚧 Holdpoint")
st.markdown("**Flag the risk. Before the delay.**")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your project schedule (CSV or Excel)", type=["csv", "xlsx"])

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
            if score >= 0.7:
                return 'High'
            elif score >= 0.4:
                return 'Medium'
            return 'Low'

        df['risk_level'] = df['risk_score'].apply(risk_label)
        color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}

        high = df[df['risk_level'] == 'High']
        medium = df[df['risk_level'] == 'Medium']
        low = df[df['risk_level'] == 'Low']
        total = len(df)
        avg_start_delay = df['start_delay'].mean()
        critical_delayed = df[(df['is_critical'] == 1) & (df['risk_level'] == 'High')]

        # --- SECTION 1: EXECUTIVE SUMMARY ---
        st.subheader("📋 Executive Summary")

        if len(high) == 0:
            health = "Your programme is in good health. No tasks are currently at high risk of delay."
            health_color = "#2ecc71"
        elif len(high) <= total * 0.2:
            health = "Your programme has isolated risk. A small number of tasks require immediate attention."
            health_color = "#f39c12"
        else:
            health = "Your programme is under significant stress. Multiple tasks are at high risk of delay."
            health_color = "#e74c3c"

        st.markdown(f"""
        <div style='background-color:{health_color}22; border-left: 4px solid {health_color}; padding: 16px; border-radius: 4px; margin-bottom: 16px;'>
        <strong>{health}</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total)
        col2.metric("🔴 High Risk", len(high))
        col3.metric("🟡 Medium Risk", len(medium))
        col4.metric("🟢 Low Risk", len(low))

        st.markdown("---")

        # --- SECTION 2: INTERPRETATION ---
        st.subheader("🔍 What the data is telling you")

        interpretations = []

        if len(critical_delayed) > 0:
            tasks = ', '.join(critical_delayed['task_name'].tolist())
            interpretations.append(f"**Critical path at risk:** {tasks} {'are' if len(critical_delayed) > 1 else 'is'} on the critical path and flagged high risk. Any further delay here directly extends your project end date.")

        if avg_start_delay > 2:
            interpretations.append(f"**Start delay pattern detected:** Tasks are starting an average of {avg_start_delay:.1f} days late. This is a systemic issue — likely resource availability, procurement lag, or predecessor dependency failures.")

        if len(high) > total * 0.3:
            interpretations.append(f"**Programme-wide stress:** Over 30% of your tasks are high risk. This suggests the baseline schedule may be too aggressive or that resource allocation is insufficient across the board.")

        tasks_no_float = df[(df['float_days'] == 0) & (df['risk_level'] != 'Low')]
        if len(tasks_no_float) > 0:
            interpretations.append(f"**No float buffer:** {len(tasks_no_float)} task(s) have zero float and elevated risk. These have no room to absorb further delay.")

        if not interpretations:
            interpretations.append("No major systemic issues detected. Monitor medium risk tasks to prevent escalation.")

        for point in interpretations:
            st.markdown(f"- {point}")

        st.markdown("---")

        # --- SECTION 3: RECOMMENDED MITIGATION PATH ---
        st.subheader("✅ Recommended mitigation path")

        recommendations = []

        if len(critical_delayed) > 0:
            for _, row in critical_delayed.iterrows():
                recommendations.append({
                    'Priority': '1 — Immediate',
                    'Task': row['task_name'],
                    'Action': 'Review resources and predecessor tasks. Consider crashing this activity — add resource or extend working hours to recover lost time.',
                    'Reason': 'Critical path task at high delay risk'
                })

        non_critical_high = high[high['is_critical'] == 0]
        for _, row in non_critical_high.iterrows():
            recommendations.append({
                'Priority': '2 — This week',
                'Task': row['task_name'],
                'Action': f"Investigate start delay of {row['start_delay']} days. Confirm resource assignment and resolve any predecessor blockers.",
                'Reason': 'High risk, non-critical — may become critical if unresolved'
            })

        for _, row in medium.iterrows():
            recommendations.append({
                'Priority': '3 — Monitor',
                'Task': row['task_name'],
                'Action': 'Add to weekly look-ahead. Flag for early warning if start delay increases.',
                'Reason': 'Medium risk — manageable if caught early'
            })

        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.success("No immediate actions required. Continue monitoring.")

        st.markdown("---")

        # --- SECTION 4: RISK CHART ---
        st.subheader("📊 Task Risk Scores")

        fig = px.bar(
            df.sort_values('risk_score', ascending=False),
            x='task_name',
            y='risk_score',
            color='risk_level',
            color_discrete_map=color_map,
            labels={'risk_score': 'Risk Score', 'task_name': 'Task'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # --- SECTION 5: FULL TASK TABLE ---
        st.subheader("📁 Full Task Detail")
        st.dataframe(
            df[['task_name', 'planned_duration', 'start_delay', 'float_days', 'is_critical', 'risk_score', 'risk_level']]
            .sort_values('risk_score', ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a project schedule (CSV or Excel) to begin your delay risk analysis.")