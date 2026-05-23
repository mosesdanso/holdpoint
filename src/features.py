import pandas as pd

def engineer_features(df):
    df = df.copy()
    
    df['planned_start'] = pd.to_datetime(df['planned_start'])
    df['planned_end'] = pd.to_datetime(df['planned_end'])
    df['actual_start'] = pd.to_datetime(df['actual_start'])
    df['actual_end'] = pd.to_datetime(df['actual_end'])
    
    df['planned_duration'] = (df['planned_end'] - df['planned_start']).dt.days
    df['start_delay'] = (df['actual_start'] - df['planned_start']).dt.days
    df['float_days'] = df['float_days'] if 'float_days' in df.columns else 0
    df['is_critical'] = df['is_critical'].astype(int) if 'is_critical' in df.columns else 0
    df['delayed'] = ((df['actual_end'] - df['planned_end']).dt.days > 0).astype(int)
    
    features = ['planned_duration', 'start_delay', 'float_days', 'is_critical']
    return df, features