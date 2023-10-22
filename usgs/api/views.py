from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json
from .utils import calculate_quartal


# @api_view(["POST"])
def preprocessing(request):
    df = pd.read_csv("usgs_main.csv")
    df = df.fillna("")

    # print(max(list(df['mag'])))

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

    upper_bound, lower_bound = calculate_quartal(df)

    df_no_outlier = df[(df['mag'] > lower_bound) & (df['mag'] < upper_bound)]

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
    }
    return render(request, "base.html", response_data)
    # return JsonResponse(response_data)
