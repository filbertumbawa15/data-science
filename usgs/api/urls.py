from django.urls import path, include
from api import views
from rest_framework.urlpatterns import format_suffix_patterns

app_name = "api"
urlpatterns = [
    path("preprocessing/", views.preprocessing, name='Home'),
]
