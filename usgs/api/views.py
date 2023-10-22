from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json


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

    s = pd.Series(df['mag'], name='legs', index=df)
    print(s.max())

    response_data = {
        "data_head": df.head().to_dict("records"),
        "data_info": data_types,
        "count_data": int(df[df.columns[0]].count()),
        # "value_mag": {"max": df['mag'].max(),"min": df['mag'].max()},
        # "value_depth": {"max": df['depth'].max(),"min": df['depth'].max()},
        # "value_rms": {"max": df['rms'].max(),"min": df['rms'].max()},
    }
    return render(request, 'base.html', response_data)
    # return JsonResponse(response_data)