from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json
from .utils import fill_null_with_mean
from .utils import visualizerDataElbow
from sklearn.preprocessing import StandardScaler


# @api_view(["POST"])
def preprocessing(request):
    df = pd.read_csv("usgs_main.csv")
    before_cleaning = df.isnull().sum().to_dict()
    df = fill_null_with_mean(df)
    after_cleaning = df.isnull().sum().to_dict()

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

    model_elbow = visualizerDataElbow(X)

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
    }
    return render(request, "base.html", response_data)
    # return JsonResponse(response_data)
