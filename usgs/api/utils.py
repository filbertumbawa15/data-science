import pandas as pd

def calculate_quartal(data):
    data['mag'] = pd.to_numeric(data['mag'], errors='coerce')
    q1 = data['mag'].quantile(0.35)
    q3 = data['mag'].quantile(0.65)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 - (1.5 * iqr)
    return upper_bound, lower_bound