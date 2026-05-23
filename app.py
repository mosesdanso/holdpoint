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
            df['risk_score'] = (df['start_delay'] / (df['planned_duration'] + 1)).clip(0, 1)

        def risk_label(score):
            if score >= 0.7:
                return 'High'
            elif score >= 0.4:
                return 'Medium'
            return 'Low'

        df['risk_level'] = df['risk_score'].apply(risk_label)

        color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}

        st.subheader("Risk Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk Tasks", len(df[df['risk_level'] == 'High']))
        col2.metric("Medium Risk Tasks", len(df[df['risk_level'] == 'Medium']))
        col3.metric("Low Risk Tasks", len(df[df['risk_level'] == 'Low']))

        st.markdown("---")

        fig = px.bar(
            df.sort_values('risk_score', ascending=False),
            x='task_name',
            y='risk_score',
            color='risk_level',
            color_discrete_map=color_map,
            title='Task Risk Scores',
            labels={'risk_score': 'Risk Score', 'task_name': 'Task'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Task Detail")
        st.dataframe(
            df[['task_name', 'planned_duration', 'start_delay', 'risk_score', 'risk_level']]
            .sort_values('risk_score', ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a schedule file to begin.")