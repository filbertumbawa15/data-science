from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt
import pandas as pd
import json
from .utils import fill_null_with_mean
from .utils import visualizerDataElbow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from io import BytesIO
import base64


# @api_view(["POST"])
def preprocessing(request):
    #Ini tahap preprocessing nya (membaca dataset nya)
    df = pd.read_csv("usgs_main.csv")
    before_cleaning = df.isnull().sum().to_dict()
    #Menampilkan value sebelum di cleaning missing value
    df = fill_null_with_mean(df) #INi adalah function yang kami buat di utils.py, yaitu untuk melakukan isnull data
    after_cleaning = df.isnull().sum().to_dict()
    #Menampilkan value setelah di cleaning missing value 

    d = (
        df.dtypes.to_frame("dtypes")
        .reset_index()
        .set_index("index")["dtypes"]
        .astype(str)
        .to_dict()
    )

    with open("types.json", "w") as f:
        json.dump(d, f)

    with open("types.json", "r") as f:
        data_types = json.load(f)
    
    X = df[['mag', 'depth', 'rms']].values

    X = StandardScaler().fit_transform(X)

    kElbow = visualizerDataElbow(X)

    img = BytesIO()
    kElbow.show(outpath=img, format='png')
    img.seek(0)

    model_elbow = base64.b64encode(img.getvalue()).decode()

    k = kElbow.elbow_value_
    randomizer = 42
    model_kmeans = KMeans(n_clusters=k, random_state=randomizer)
    model_kmeans.fit(X)

    labels_kmeans = model_kmeans.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='rainbow')

    plt.scatter(model_kmeans.cluster_centers_[:,0], model_kmeans.cluster_centers_[:,1], color='black')  

    imgPlot = BytesIO()
    plt.savefig(imgPlot, format='png')
    img.seek(0)

    cluster_image = base64.b64encode(imgPlot.getvalue()).decode()

    df['cluster'] = model_kmeans.labels_

    response_data = {
        "data_head": df.head().to_dict("records"),
        "data_info": data_types,
        "count_data": int(df[df.columns[0]].count()),
        "dataset_file": "usgs_main.csv",
        "total_attributes": len(df.columns),
        "value_max": json.dumps(
            {
                "max_mag": pd.to_numeric(df["mag"], errors="coerce").max(),
                "max_depth": pd.to_numeric(df["depth"], errors="coerce").max(),
                "max_rms": pd.to_numeric(df["rms"], errors="coerce").max(),
            }
        ),
        "value_min": json.dumps(
            {
                "min_mag": pd.to_numeric(df["mag"], errors="coerce").min(),
                "min_depth": pd.to_numeric(df["depth"], errors="coerce").min(),
                "min_rms": pd.to_numeric(df["rms"], errors="coerce").min(),
            }
        ),
        "before_cleaning": before_cleaning,
        "after_cleaning": after_cleaning,
        "elbow_model": model_elbow,
        "cluster_image": cluster_image,
        "data_cluster": df.head(30).to_dict("records"),
    }
    return render(request, "base.html", response_data)
    # return JsonResponse(response_data)
