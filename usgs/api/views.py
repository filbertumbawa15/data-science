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
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from io import BytesIO
import base64

# @api_view(["POST"])
def preprocessing(request):
    global dataJson
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

    dataJson = df

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
    }
    return render(request, "base.html", response_data)
    # return JsonResponse(response_data)

@csrf_exempt
def paginate_data(request):
    page = int(request.POST.get('page', 1))
    limit = 10

    start_index = (page - 1) * limit
    end_index = page * limit

    search_value = request.POST.get('search[value]', '')

    filtered_data = [item for item in dataJson.to_dict("records") if search_value in str(item)]

    paginate_data = filtered_data[start_index:end_index]

    return JsonResponse({
        'data': paginate_data,
        'draw': int(request.POST.get('draw', 1)),
        'recordsTotal': len(dataJson.to_dict("records")),
        'recordsFiltered': len(filtered_data),
    })

@csrf_exempt
def initPrediction(request):
    global model_knn
    data_prediction = dataJson[['mag', 'depth', 'rms', 'latitude', 'longitude', 'type', 'cluster']]
    data_prediction['type'] = data_prediction['type'].map({'earthquake':0, 'quarry blast':1, 'explosion':2, 'ice quake':3, 'chemical explosion':4, 'other event':5})

    X = data_prediction.drop(labels='cluster', axis=1).values
    y = data_prediction['cluster'].values

    sm = SMOTENC(random_state=42, categorical_features=[5])
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Standardization Feature
    X_train_scaled = scaler.transform(X_train)
    hyperparameters = {'n_neighbors': list(range(2, 21))}

    model_knn = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=8, refit=True)

    model_knn.fit(X_train_scaled, y_train)

    k_params = model_knn.best_params_
    k = k_params['n_neighbors']
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)

    y_pred = model_knn.predict(X_test)

    return JsonResponse({
        'result': "Already Done",
    })

@csrf_exempt
def prediction(request):
    inputMag = float(request.POST.get('inputMag'))
    inputDepth = float(request.POST.get('inputDepth'))
    inputRms = float(request.POST.get('inputRms'))
    inputLatitude = float(request.POST.get('inputLatitude'))
    inputLongitude = float(request.POST.get('inputLongitude'))
    inputType = int(request.POST.get('inputType'))
    # hasil_prediksi = model_knn.predict([[1.24, 1.610000, 0.04, 38.759666, -122.719666, 0]])
    hasil_prediksi = model_knn.predict([[inputMag, inputDepth, inputRms, inputLatitude, inputLongitude, inputType]])
    hasil_prediksi = int(hasil_prediksi)
    keterangan = ''

    if hasil_prediksi == 0:
        if inputType == 0:
            keterangan = "Guncangan masih dalam tahap aman"
        elif inputType == 1:
            keterangan = "Lokasi area tambang, masih dalam tahap aman"
        elif inputType == 2:
            keterangan = "Adanya erupsi, gempa masih dalam tahap aman"
        elif inputType == 3:
            keterangan = "Area lempeng es, masih dalam tahap aman"
        elif inputType == 4:
            keterangan = "Ujicoba nuklir, masih dalam tahap aman"
        else:
            keterangan = "Lokasi tidak terdeteksi"

    elif hasil_prediksi == 1:
        if inputType == 0:
            keterangan = "Guncangan masih dalam tahap aman"
        elif inputType == 1:
            keterangan = "Lokasi area tambang, masih dalam tahap aman"
        elif inputType == 2:
            keterangan = "Adanya erupsi, gempa masih dalam tahap aman"
        elif inputType == 3:
            keterangan = "Area lempeng es, masih dalam tahap aman"
        elif inputType == 4:
            keterangan = "Ujicoba nuklir, masih dalam tahap aman"
        else:
            keterangan = "Lokasi tidak terdeteksi"

    elif hasil_prediksi == 2:
        if inputType == 0:
            keterangan = "Guncangan lumayan kencang, lakukan evakuasi"
        elif inputType == 1:
            keterangan = "Lokasi area tambang, mohon lakukan evakuasi"
        elif inputType == 2:
            keterangan = "Lokasi area terjadinya erupsi gunung merapi, silahkan lakukan evakuasi agar tidak terkena dampak"
        elif inputType == 3:
            keterangan = "Kerusakan di area pemukiman dingin, harap lakukan evakuasi"
        elif inputType == 4:
            keterangan = "Ujicoba nuklir yang lumayan tinggi, silahkan menjauh dahulu dari area tersebut"
        else:
            keterangan = "Lokasi tidak terdeteksi"

    elif hasil_prediksi == 3:
        if inputType == 0:
            keterangan = "Guncangan dalam tahap berpotensi tsunami"
        elif inputType == 1:
            keterangan = "Lokasi tambang dalam tahap tingkat tinggi, mohon lakukan evakuasi agar tidak terjadi longsor diarea tersebut"
        elif inputType == 2:
            keterangan = "Sudah memasuki tahap bahaya, silahkan lakukan evakuasi"
        elif inputType == 3:
            keterangan = "Retakan es yang lumayan, mohon menjauh dari area tersebut"
        elif inputType == 4:
            keterangan = "Ujicoba nukir tahap tinggi, silahkan lakukan evakuasi"
        else:
            keterangan = "Lokasi tidak terdeteksi"

    elif hasil_prediksi == 4:
        if inputType == 0:
            keterangan = "Guncangan berpotensi tsunami, silahkan lakukan evakuasi secepatnya"
        elif inputType == 1:
            keterangan = "Lokasi area tambang tahap tinggi, harap lakukan evakusi segera agar tidak terjadi bencana"
        elif inputType == 2:
            keterangan = "Adanya letusan magma tingkat tinggi, segera lakukan evakuasi agar tidak terjadi dampak yang berbahaya"
        elif inputType == 3:
            keterangan = "Retakan es dalam tahap tinggi, segera lakukan evakuasi secepatnya"
        elif inputType == 4:
            keterangan = "Tahap bahaya, silahkan lakukan evakuasi dan segera lakukan penindakan agar tidak terjadi bahaya"
        else:
            keterangan = "Lokasi tidak terdeteksi"

    return JsonResponse({
        'hasil_prediksi': keterangan,
    })