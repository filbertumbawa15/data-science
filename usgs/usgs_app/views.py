from django.shortcuts import render

# Create your views here.

def preprocessing(request):
    # data_file = request.FILES["file"]
    # df = pd.read_csv(data_file)
    # df = df.fillna('')

    return render(request, 'base.html')
# http://127.0.0.1:8000/api/preprocessing

    # d = df.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()

    # with open('types.json', 'w') as f:
    #     json.dump(d, f)

    # with open('types.json', 'r') as f:
    #     data_types = json.load(f)

    # response_data = {
    #     "data_head": df.head().to_dict('records'),
    #     # "data_describe": df.describe().to_dict('records'),
    #     "data_info": data_types,
    # }
    # return Response(response_data)