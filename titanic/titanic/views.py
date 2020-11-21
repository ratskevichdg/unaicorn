from django.shortcuts import render
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def result(request):

    #plclass, sex, age, sibsp, parch, fare, embarked, title

    pclass = request.GET['pclass']
    sex = request.GET['sex']
    age = request.GET['age']
    sibsp = request.GET['sibsp']
    parch = request.GET['parch']
    fare = request.GET['fare']
    embarked = request.GET['embarked']
    title = request.GET['title']
    prediction = ml_predict.prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title)
    return render(request, 'result.html', {'prediction':prediction})