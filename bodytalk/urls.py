from django.urls import path
from . import views

app_name = 'bodytalk'
urlpatterns = [
    path('', views.disease_recomm),
]