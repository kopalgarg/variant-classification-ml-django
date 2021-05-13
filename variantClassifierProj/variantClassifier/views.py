from django.http import HttpResponse
from django.shortcuts import render
import joblib
def home(request):
    return render(request, "home.html")

def result(request):
    # load model
    dtree = joblib.load('final_model.sav')
    input = request.GET('file')
    input.head()
    return render(request, "result.html")
