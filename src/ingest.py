import pandas as pd

def load_schedule(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def validate_columns(df):
    required = ['task_name', 'planned_start', 'planned_end', 'actual_start', 'actual_end']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True