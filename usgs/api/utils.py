from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from io import BytesIO
import pandas as pd
import base64


def calculate_quartal(data):
    data["mag"] = pd.to_numeric(data["mag"], errors="coerce")
    q1 = data["mag"].quantile(0.35)
    q3 = data["mag"].quantile(0.65)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 - (1.5 * iqr)
    return upper_bound, lower_bound


def fill_null_with_mean(data):
    for column in data.columns:
        if data[column].dtype != 'object':
            mean_val = data[column].mean()
            data[column].fillna(mean_val, inplace=True)
        else:
            rata_magT = data[column].value_counts().mean()
            data[column].fillna(rata_magT, inplace=True)
    return data

def visualizerDataElbow(data):
    tes_model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(tes_model, k=(2,10))
    visualizer.fit(data)
    img = BytesIO()
    visualizer.show(outpath=img, format='png')
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()