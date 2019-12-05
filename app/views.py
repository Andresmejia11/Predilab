"""
Definition of views.
"""
from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render
from django.http import HttpRequest
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import AverageForm
import io
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render
from matplotlib.backends.backend_agg import FigureCanvasAgg
from random import sample
import matplotlib.pyplot as plt
from django.views.generic import View 
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from prettytable import PrettyTable
import warnings
import psycopg2
import os
def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def informe(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/informe.html',
        {
            'title':'',
            'year':datetime.now().year,
        }
    )
def pag11(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/pag1.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def UsoM(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/UsoM.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def pred2(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/pred2.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def pr(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/PruebaHTML.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )
def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'Informe De Resultados',
            'message':'Informe De Resultados',
            'year':datetime.now().year,
        }
    )
#========================Modelo Dos======================================
def averageProm(request):
    warnings.simplefilter('ignore')
    #os.environ.get('Password')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT * FROM tmachinefinal ORDER BY añoi,numsem ASC"
    df = pd.read_sql(query, conn)
    Data=df
    frames = [Data]
    datos = pd.concat(frames)
    datos.head()
    X = datos.drop(['nrep'], axis=1).copy()
    y = datos[['nrep']]
    X_train, X_test, y_train, w_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
  
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    y_ranfor = RanFor.predict(X_test)
    y_ranfor = y_ranfor.round()
    
    #____________________________________________________________________________________________________________
    X = datos.drop(['multimetro'], axis=1).copy()
    y = datos[['multimetro']]

    X_train, X_test, y_train, A_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
 
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    A_ranfor = RanFor.predict(X_test)
    A_ranfor = A_ranfor.round()
    
#______________________________________________________________________________________________________________________________________________
    X = datos.drop(['otros'], axis=1).copy()
    y = datos[['otros']]
    X_train, X_test, y_train, B_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    B_ranfor = RanFor.predict(X_test)
    B_ranfor = B_ranfor.round()
    
  #________________________________________________________________________________________________________________________________
    X = datos.drop(['genosc'], axis=1).copy()
    y = datos[['genosc']]
    X_train, X_test, y_train, C_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    C_ranfor = RanFor.predict(X_test)
    C_ranfor = C_ranfor.round()
   
    #____________________________________________________________________________________________________________________________________
    X = datos.drop(['fuente'], axis=1).copy()
    y = datos[['fuente']]
    X_train, X_test, y_train, D_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    D_ranfor = RanFor.predict(X_test)
    D_ranfor = D_ranfor.round()
   
#_________________________________________________________________________________________________________________________
    X = datos.drop(['sondas'], axis=1).copy()
    y = datos[['sondas']]
    X_train, X_test, y_train, E_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    E_ranfor = RanFor.predict(X_test)
    E_ranfor = E_ranfor.round()
    
#_____________________________________________________________________________________________
    X = datos.drop(['fusible'], axis=1).copy()
    y = datos[['fusible']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    F_ranfor = RanFor.predict(X_test)
    F_ranfor = F_ranfor.round()
    
    #----------------------------------------------
    #queryset = Predilabtable3.objects.all()
    Reds2n=  int(w_test.iloc[1]['nrep'])
    Reds1n=  int(w_test.iloc[2]['nrep'])
    sum1=   int(y_ranfor[1:2])
    sum2=   int(y_ranfor[2:3])
    sum3=   int(y_ranfor[3:4])
    sum4=   int(y_ranfor[4:5])
    #-----MUl------
    Reds2m=  int(A_test.iloc[1]['multimetro'])
    Reds1m=  int(A_test.iloc[2]['multimetro'])
    sum12=   int(A_ranfor[1:2])
    sum22=   int(A_ranfor[2:3])
    sum32=   int(A_ranfor[3:4])
    sum42=   int(A_ranfor[4:5])
    #-----otros------
    Reds2o=  int(B_test.iloc[1]['otros'])
    Reds1o=  int(B_test.iloc[2]['otros'])
    sum13=   int(B_ranfor[1:2])
    sum23=   int(B_ranfor[2:3])
    sum33=   int(B_ranfor[3:4])
    sum43=   int(B_ranfor[4:5])
    #-----genosc------
    Reds2g=  int(C_test.iloc[1]['genosc'])
    Reds1g=  int(C_test.iloc[2]['genosc'])
    sum14=   int(C_ranfor[1:2])
    sum24=   int(C_ranfor[2:3])
    sum34=   int(C_ranfor[3:4])
    sum44=   int(C_ranfor[4:5])
    #-----fuente-----
    Reds2f=  int(D_test.iloc[1]['fuente'])
    Reds1f=  int(D_test.iloc[2]['fuente'])
    sum15=   int(D_ranfor[1:2])
    sum25=   int(D_ranfor[2:3])
    sum35=   int(D_ranfor[3:4])
    sum45=   int(D_ranfor[4:5])
    #-----sondas-----
    Reds2s=  int(E_test.iloc[1]['sondas'])
    Reds1s=  int(E_test.iloc[2]['sondas'])
    sum16=   int(E_ranfor[1:2])
    sum26=   int(E_ranfor[2:3])
    sum36=   int(E_ranfor[3:4])
    sum46=   int(E_ranfor[4:5])
    #-----FUSI-----
    Reds2=    int(y_test.iloc[1]['fusible'])
    Reds1=    int(y_test.iloc[2]['fusible'])
    sum17=   int(F_ranfor[1:2])
    sum27=   int(F_ranfor[2:3])
    sum37=   int(F_ranfor[3:4])
    sum47=   int(F_ranfor[4:5])
    form = AverageForm()
    context = {
        'form': form,
        #----------nrep
        'sm2n':Reds2n,
        'sm1n':Reds1n,
        'average1': sum1,
        'average2': sum2,
        'average3': sum3,
        'average4': sum4,
        #--------------Mul
        'sm2m':Reds2m,
        'sm1m':Reds1m,
        'average_11': sum12,
        'average_12': sum22,
        'average_13': sum32,
        'average_14': sum42,
        #--------------Otr
        'sm2o':Reds2o,
        'sm1o':Reds1o,
        'average_11o': sum13,
        'average_12o': sum23,
        'average_13o': sum33,
        'average_14o': sum43,
        #--------------GENoS
        'sm2g':Reds2g,
        'sm1g':Reds1g,
        'average_11g': sum14,
        'average_12g': sum24,
        'average_13g': sum34,
        'average_14g': sum44,
         #--------------FUEN
        'sm2f':Reds2f,
        'sm1f':Reds1f,
        'average_11f': sum15,
        'average_12f': sum25,
        'average_13f': sum35,
        'average_14f': sum45,
        #--------------sondas
        'sm2s':Reds2s,
        'sm1s':Reds1s,
        'average_11s': sum16,
        'average_12s': sum26,
        'average_13s': sum36,
        'average_14s': sum46,
        #--------------FUS
        'sm2': Reds2,
        'sm1': Reds1,
        'average_11u': sum17,
        'average_12u': sum27,
        'average_13u': sum37,
        'average_14u': sum47,
    }
    return render(request, 'app/PruebaHTML.html', context)
#------------------------Modelo Uno--------------------------------------
def UNO(request):
    warnings.simplefilter('ignore')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT * FROM tmachinefinal ORDER BY añoi,numsem ASC"
    df = pd.read_sql(query, conn)
    Data=df
    frames = [Data]
    datos = pd.concat(frames)
    datos.head()
    X = datos.drop(['nrep'], axis=1).copy()
    y = datos[['nrep']]
    X_train, X_test, y_train, w_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='sgd', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    mlpPredict = mlp.predict(X_test)
    mlpPredict = mlpPredict.round()

    #____________________________________________________________________________________________________________
    X = datos.drop(['multimetro'], axis=1).copy()
    y = datos[['multimetro']]
    X_train, X_test, y_train, A_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    A_mlpPredict = mlp.predict(X_test)
    A_mlpPredict = A_mlpPredict.round()
    
#______________________________________________________________________________________________________________________________________________
    X = datos.drop(['otros'], axis=1).copy()
    y = datos[['otros']]
    X_train, X_test, y_train, B_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=500,
    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    B_mlpPredict = mlp.predict(X_test)
    B_mlpPredict = B_mlpPredict.round()
    
  #________________________________________________________________________________________________________________________________
    X = datos.drop(['genosc'], axis=1).copy()
    y = datos[['genosc']]
    X_train, X_test, y_train, C_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=500,
    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    C_mlpPredict = mlp.predict(X_test)
    C_mlpPredict = C_mlpPredict.round()
    
    #____________________________________________________________________________________________________________________________________
    X = datos.drop(['fuente'], axis=1).copy()
    y = datos[['fuente']]
    X_train, X_test, y_train, D_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='constant', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    D_mlpPredict = mlp.predict(X_test)
    D_mlpPredict = D_mlpPredict.round()
    
#_________________________________________________________________________________________________________________________
    X = datos.drop(['sondas'], axis=1).copy()
    y = datos[['sondas']]
    X_train, X_test, y_train, E_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=500,
    learning_rate='constant', learning_rate_init= 0.01)
    mlp.fit(X_train, y_train)
    E_mlpPredict = mlp.predict(X_test)
    E_mlpPredict = E_mlpPredict.round()
   
#_____________________________________________________________________________________________
    X = datos.drop(['fusible'], axis=1).copy()
    y = datos[['fusible']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    F_mlpPredict = mlp.predict(X_test)
    F_mlpPredict = F_mlpPredict.round()
   
    #----------------------------------------------
    #queryset = Predilabtable3.objects.all()
    Red1=    int(mlpPredict[1:2])
    Red2=    int(mlpPredict[2:3])
    Red3=    int(mlpPredict[3:4])
    Red4=    int(mlpPredict[4:5])
    Reds2n=  int(w_test.iloc[1]['nrep'])
    Reds1n=  int(w_test.iloc[2]['nrep'])
    #-----MUl------
    Red12=    int(A_mlpPredict[1:2])
    Red22=    int(A_mlpPredict[2:3])
    Red32=    int(A_mlpPredict[3:4])
    Red42=    int(A_mlpPredict[4:5])
    Reds2m=  int(A_test.iloc[1]['multimetro'])
    Reds1m=  int(A_test.iloc[2]['multimetro'])
    #-----otros------
    Red13=    int(B_mlpPredict[1:2])
    Red23=    int(B_mlpPredict[2:3])
    Red33=    int(B_mlpPredict[3:4])
    Red43=    int(B_mlpPredict[4:5])
    Reds2o=  int(B_test.iloc[1]['otros'])
    Reds1o=  int(B_test.iloc[2]['otros'])
    #-----genosc------
    Red14=    int(C_mlpPredict[1:2])
    Red24=    int(C_mlpPredict[2:3])
    Red34=    int(C_mlpPredict[3:4])
    Red44=    int(C_mlpPredict[4:5])
    Reds2g=  int(C_test.iloc[1]['genosc'])
    Reds1g=  int(C_test.iloc[2]['genosc'])
   #-----fuente-----
    Red15=    int(D_mlpPredict[1:2])
    Red25=    int(D_mlpPredict[2:3])
    Red35=    int(D_mlpPredict[3:4])
    Red45=    int(D_mlpPredict[4:5])
    Reds2f=  int(D_test.iloc[1]['fuente'])
    Reds1f=  int(D_test.iloc[2]['fuente'])
    #-----sondas-----
    Red16=    int(E_mlpPredict[1:2])
    Red26=    int(E_mlpPredict[2:3])
    Red36=    int(E_mlpPredict[3:4])
    Red46=    int(E_mlpPredict[4:5])
    Reds2s=  int(E_test.iloc[1]['sondas'])
    Reds1s=  int(E_test.iloc[2]['sondas'])
    #-----FUSI-----
    Red17=    int(F_mlpPredict[1:2])
    Red27=    int(F_mlpPredict[2:3])
    Red37=    int(F_mlpPredict[3:4])
    Red47=    int(F_mlpPredict[4:5])
    Reds2=    int(y_test.iloc[1]['fusible'])
    Reds1=    int(y_test.iloc[2]['fusible'])
    form = AverageForm()
    context = {
        'form': form,
        #----------nrep
        'average':   Red1,
        'average11': Red2,
        'average12': Red3,
        'average13': Red4,
        'sm2n':Reds2n,
        'sm1n':Reds1n,
        #--------------Mul
        'average_1': Red12,
        'average_2': Red22,
        'average_3': Red32,
        'average_4': Red42,
        'sm2m':Reds2m,
        'sm1m':Reds1m,
        #--------------Otr
        'average_1o': Red13,
        'average_2o': Red23,
        'average_3o': Red33,
        'average_4o': Red43,
        'sm2o':Reds2o,
        'sm1o':Reds1o,
        #--------------GENoS
        'average_1g': Red14,
        'average_2g': Red24,
        'average_3g': Red34,
        'average_4g': Red44,
        'sm2g':Reds2g,
        'sm1g':Reds1g,
         #--------------FUEN
        'average_1f': Red15,
        'average_2f': Red25,
        'average_3f': Red35,
        'average_4f': Red45,
        'sm2f':Reds2f,
        'sm1f':Reds1f,
        #--------------sondas
        'average_1s': Red16,
        'average_2s': Red26,
        'average_3s': Red36,
        'average_4s': Red46,
        'sm2s':Reds2s,
        'sm1s':Reds1s,
        #--------------FUS
        'average_1u': Red17,
        'average_2u': Red27,
        'average_3u': Red37,
        'average_4u': Red47,
        'sm2': Reds2,
        'sm1': Reds1,
    }
    return render(request, 'app/pag1.html', context)
#=======================Informe de resultados==============================
def Dos(request):
    warnings.simplefilter('ignore')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT * FROM tmachinefinal ORDER BY añoi,numsem ASC"
    df = pd.read_sql(query, conn)
  
    Data=df
    frames = [Data]
    datos = pd.concat(frames)
    datos.head()
    X = datos.drop(['nrep'], axis=1).copy()
    y = datos[['nrep']]
    X_train, X_test, y_train, w_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='sgd', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    mlpPredict = mlp.predict(X_test)
    mlpPredict = mlpPredict.round()
       
        
  
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    y_ranfor = RanFor.predict(X_test)
    y_ranfor = y_ranfor.round()
   
    
    
    #____________________________________________________________________________________________________________
    X = datos.drop(['multimetro'], axis=1).copy()
    y = datos[['multimetro']]
    X_train, X_test, y_train, A_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    A_mlpPredict = mlp.predict(X_test)
    A_mlpPredict = A_mlpPredict.round()
    
       

    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    A_ranfor = RanFor.predict(X_test)
    A_ranfor = A_ranfor.round()
    
   
  
#______________________________________________________________________________________________________________________________________________
    X = datos.drop(['otros'], axis=1).copy()
    y = datos[['otros']]
    X_train, X_test, y_train, B_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=500,
    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    B_mlpPredict = mlp.predict(X_test)
    B_mlpPredict = B_mlpPredict.round()
   
   
      
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    B_ranfor = RanFor.predict(X_test)
    B_ranfor = B_ranfor.round()
    
   
    
  #________________________________________________________________________________________________________________________________
    X = datos.drop(['genosc'], axis=1).copy()
    y = datos[['genosc']]
    X_train, X_test, y_train, C_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=500,
    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    C_mlpPredict = mlp.predict(X_test)
    C_mlpPredict = C_mlpPredict.round()
   
   
  
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    C_ranfor = RanFor.predict(X_test)
    C_ranfor = C_ranfor.round()
   
   
    
#____________________________________________________________________________________________________________________________________
    X = datos.drop(['fuente'], axis=1).copy()
    y = datos[['fuente']]
    X_train, X_test, y_train, D_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='constant', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    D_mlpPredict = mlp.predict(X_test)
    D_mlpPredict = D_mlpPredict.round()
    
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    D_ranfor = RanFor.predict(X_test)
    D_ranfor = D_ranfor.round()
    
    
 
#_________________________________________________________________________________________________________________________
    X = datos.drop(['sondas'], axis=1).copy()
    y = datos[['sondas']]
    X_train, X_test, y_train, E_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=500,
    learning_rate='constant', learning_rate_init= 0.01)
    mlp.fit(X_train, y_train)
    E_mlpPredict = mlp.predict(X_test)
    E_mlpPredict = E_mlpPredict.round()
    
    
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    E_ranfor = RanFor.predict(X_test)
    E_ranfor = E_ranfor.round()
   
     
#_____________________________________________________________________________________________
    X = datos.drop(['fusible'], axis=1).copy()
    y = datos[['fusible']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    F_mlpPredict = mlp.predict(X_test)
    F_mlpPredict = F_mlpPredict.round()
    
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    F_ranfor = RanFor.predict(X_test)
    F_ranfor = F_ranfor.round()
   
    
    #----------------------------------------------
    X = datos.drop(['tipo_0'], axis=1).copy()
    y = datos[['tipo_0']]
    X_train, X_test, y_train, m_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    m_mlpPredict = mlp.predict(X_test)
    m_mlpPredict = m_mlpPredict.round()
   
    
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    m_ranfor = RanFor.predict(X_test)
    m_ranfor = m_ranfor.round()
    
   
    
    #__________________________________________________________________________________________________________________________
    X = datos.drop(['tipo_1'], axis=1).copy()
    y = datos[['tipo_1']]
    X_train, X_test, y_train, n_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    n_mlpPredict = mlp.predict(X_test)
    n_mlpPredict = n_mlpPredict.round()
    
  
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    n_ranfor = RanFor.predict(X_test)
    n_ranfor = n_ranfor.round()
    
   
    
    #_______________________________________________________________________________________________
    X = datos.drop(['tipo_2'], axis=1).copy()
    y = datos[['tipo_2']]
    X_train, X_test, y_train, o_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    o_mlpPredict = mlp.predict(X_test)
    o_mlpPredict = o_mlpPredict.round()
    
   
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    o_ranfor = RanFor.predict(X_test)
    o_ranfor = o_ranfor.round()
    
    
    
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_3'], axis=1).copy()
    y = datos[['tipo_3']]
    X_train, X_test, y_train, p_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    p_mlpPredict = mlp.predict(X_test)
    p_mlpPredict = p_mlpPredict.round()
    
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    p_ranfor = RanFor.predict(X_test)
    p_ranfor = p_ranfor.round()
    
    
    
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_4'], axis=1).copy()
    y = datos[['tipo_4']]
    X_train, X_test, y_train, q_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    q_mlpPredict = mlp.predict(X_test)
    q_mlpPredict = q_mlpPredict.round()
    
    
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    q_ranfor = RanFor.predict(X_test)
    q_ranfor = q_ranfor.round()
    
   
   
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_5'], axis=1).copy()
    y = datos[['tipo_5']]
    X_train, X_test, y_train, r_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    r_mlpPredict = mlp.predict(X_test)
    r_mlpPredict = r_mlpPredict.round()
    
   
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    r_ranfor = RanFor.predict(X_test)
    r_ranfor = r_ranfor.round()
    
   

    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_6'], axis=1).copy()
    y = datos[['tipo_6']]
    X_train, X_test, y_train, s_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    s_mlpPredict = mlp.predict(X_test)
    s_mlpPredict = s_mlpPredict.round()
    
   
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    s_ranfor = RanFor.predict(X_test)
    s_ranfor = s_ranfor.round()
   
   
   
    #____________________________________________________________________________________________
    X = datos.drop(['tipo_7'], axis=1).copy()
    y = datos[['tipo_7']]
    X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    t_mlpPredict = mlp.predict(X_test)
    t_mlpPredict = t_mlpPredict.round()
    
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    t_ranfor = RanFor.predict(X_test)
    t_ranfor = t_ranfor.round()
   
    
   
    #____________________________________________________________________________________________
    Red1=    int(mlpPredict[1:2])
    Red2=    int(mlpPredict[2:3])
    Red3=    int(mlpPredict[3:4])
    Red4=    int(mlpPredict[4:5])
    Reds2n=  int(w_test.iloc[1]['nrep'])
    Reds1n=  int(w_test.iloc[2]['nrep'])
    sum1=   int(y_ranfor[1:2])
    sum2=   int(y_ranfor[2:3])
    sum3=   int(y_ranfor[3:4])
    sum4=   int(y_ranfor[4:5])
   #-----MUl------
    Red12=    int(A_mlpPredict[1:2])
    Red22=    int(A_mlpPredict[2:3])
    Red32=    int(A_mlpPredict[3:4])
    Red42=    int(A_mlpPredict[4:5])
    Reds2m=  int(A_test.iloc[1]['multimetro'])
    Reds1m=  int(A_test.iloc[2]['multimetro'])
    sum12=   int(A_ranfor[1:2])
    sum22=   int(A_ranfor[2:3])
    sum32=   int(A_ranfor[3:4])
    sum42=   int(A_ranfor[4:5])
    #-----otros------
    Red13=    int(B_mlpPredict[1:2])
    Red23=    int(B_mlpPredict[2:3])
    Red33=    int(B_mlpPredict[3:4])
    Red43=    int(B_mlpPredict[4:5])
    Reds2o=  int(B_test.iloc[1]['otros'])
    Reds1o=  int(B_test.iloc[2]['otros'])
    sum13=   int(B_ranfor[1:2])
    sum23=   int(B_ranfor[2:3])
    sum33=   int(B_ranfor[3:4])
    sum43=   int(B_ranfor[4:5])
    #-----genosc------
    Red14=    int(C_mlpPredict[1:2])
    Red24=    int(C_mlpPredict[2:3])
    Red34=    int(C_mlpPredict[3:4])
    Red44=    int(C_mlpPredict[4:5])
    Reds2g=  int(C_test.iloc[1]['genosc'])
    Reds1g=  int(C_test.iloc[2]['genosc'])
    sum14=   int(C_ranfor[1:2])
    sum24=   int(C_ranfor[2:3])
    sum34=   int(C_ranfor[3:4])
    sum44=   int(C_ranfor[4:5])
    #-----fuente-----
    Red15=    int(D_mlpPredict[1:2])
    Red25=    int(D_mlpPredict[2:3])
    Red35=    int(D_mlpPredict[3:4])
    Red45=    int(D_mlpPredict[4:5])
    Reds2f=  int(D_test.iloc[1]['fuente'])
    Reds1f=  int(D_test.iloc[2]['fuente'])
    sum15=   int(D_ranfor[1:2])
    sum25=   int(D_ranfor[2:3])
    sum35=   int(D_ranfor[3:4])
    sum45=   int(D_ranfor[4:5])
    #-----sondas-----
    Red16=    int(E_mlpPredict[1:2])
    Red26=    int(E_mlpPredict[2:3])
    Red36=    int(E_mlpPredict[3:4])
    Red46=    int(E_mlpPredict[4:5])
    Reds2s=  int(E_test.iloc[1]['sondas'])
    Reds1s=  int(E_test.iloc[2]['sondas'])
    sum16=   int(E_ranfor[1:2])
    sum26=   int(E_ranfor[2:3])
    sum36=   int(E_ranfor[3:4])
    sum46=   int(E_ranfor[4:5])
    #-----FUSI-----
    Red17=    int(F_mlpPredict[1:2])
    Red27=    int(F_mlpPredict[2:3])
    Red37=    int(F_mlpPredict[3:4])
    Red47=    int(F_mlpPredict[4:5])
    Reds2=    int(y_test.iloc[1]['fusible'])
    Reds1=    int(y_test.iloc[2]['fusible'])
    sum17=   int(F_ranfor[1:2])
    sum27=   int(F_ranfor[2:3])
    sum37=   int(F_ranfor[3:4])
    sum47=   int(F_ranfor[4:5])
     #___tipo_0
    Rem17=    int(m_mlpPredict[1:2])
    Rem27=    int(m_mlpPredict[2:3])
    Rem37=    int(m_mlpPredict[3:4])
    Rem47=    int(m_mlpPredict[4:5])
    Rem2=    int(m_test.iloc[1]['tipo_0'])
    Rem1=    int(m_test.iloc[2]['tipo_0'])
    su17=   int(m_ranfor[1:2])
    su27=   int(m_ranfor[2:3])
    su37=   int(m_ranfor[3:4])
    su47=   int(m_ranfor[4:5])
    #___tipo_1
    Ren17=    int(n_mlpPredict[1:2])
    Ren27=    int(n_mlpPredict[2:3])
    Ren37=    int(n_mlpPredict[3:4])
    Ren47=    int(n_mlpPredict[4:5])
    Ren2=    int(n_test.iloc[1]['tipo_1'])
    Ren1=    int(n_test.iloc[2]['tipo_1'])
    su17n=   int(n_ranfor[1:2])
    su27n=   int(n_ranfor[2:3])
    su37n=   int(n_ranfor[3:4])
    su47n=   int(n_ranfor[4:5])
    #___tipo_2
    Reo17=    int(o_mlpPredict[1:2])
    Reo27=    int(o_mlpPredict[2:3])
    Reo37=    int(o_mlpPredict[3:4])
    Reo47=    int(o_mlpPredict[4:5])
    Reo2=    int(o_test.iloc[1]['tipo_2'])
    Reo1=    int(o_test.iloc[2]['tipo_2'])
    su17o=   int(o_ranfor[1:2])
    su27o=   int(o_ranfor[2:3])
    su37o=   int(o_ranfor[3:4])
    su47o=   int(o_ranfor[4:5])
    #___tipo_3
    Rep17=    int(p_mlpPredict[1:2])
    Rep27=    int(p_mlpPredict[2:3])
    Rep37=    int(p_mlpPredict[3:4])
    Rep47=    int(p_mlpPredict[4:5])
    Rep2=    int(p_test.iloc[1]['tipo_3'])
    Rep1=    int(p_test.iloc[2]['tipo_3'])
    su17p=   int(p_ranfor[1:2])
    su27p=   int(p_ranfor[2:3])
    su37p=   int(p_ranfor[3:4])
    su47p=   int(p_ranfor[4:5])
     #___tipo_4
    Req17=    int(q_mlpPredict[1:2])
    Req27=    int(q_mlpPredict[2:3])
    Req37=    int(q_mlpPredict[3:4])
    Req47=    int(q_mlpPredict[4:5])
    Req2=    int(q_test.iloc[1]['tipo_4'])
    Req1=    int(q_test.iloc[2]['tipo_4'])
    su17q=   int(q_ranfor[1:2])
    su27q=   int(q_ranfor[2:3])
    su37q=   int(q_ranfor[3:4])
    su47q=   int(q_ranfor[4:5])
     #___tipo_5
    Rer17=    int(r_mlpPredict[1:2])
    Rer27=    int(r_mlpPredict[2:3])
    Rer37=    int(r_mlpPredict[3:4])
    Rer47=    int(r_mlpPredict[4:5])
    Rer2=    int(r_test.iloc[1]['tipo_5'])
    Rer1=    int(r_test.iloc[2]['tipo_5'])
    su17r=   int(r_ranfor[1:2])
    su27r=   int(r_ranfor[2:3])
    su37r=   int(r_ranfor[3:4])
    su47r=   int(r_ranfor[4:5])
    #___tipo_6
    Res17=    int(s_mlpPredict[1:2])
    Res27=    int(s_mlpPredict[2:3])
    Res37=    int(s_mlpPredict[3:4])
    Res47=    int(s_mlpPredict[4:5])
    Res2=    int(s_test.iloc[1]['tipo_6'])
    Res1=    int(s_test.iloc[2]['tipo_6'])
    su17s=   int(s_ranfor[1:2])
    su27s=   int(s_ranfor[2:3])
    su37s=   int(s_ranfor[3:4])
    su47s=   int(s_ranfor[4:5])
    #___tipo_7
    Ret17=    int(t_mlpPredict[1:2])
    Ret27=    int(t_mlpPredict[2:3])
    Ret37=    int(t_mlpPredict[3:4])
    Ret47=    int(t_mlpPredict[4:5])
    Ret2=    int(t_test.iloc[1]['tipo_7'])
    Ret1=    int(t_test.iloc[2]['tipo_7'])
    su17t=   int(t_ranfor[1:2])
    su27t=   int(t_ranfor[2:3])
    su37t=   int(t_ranfor[3:4])
    su47t=   int(t_ranfor[4:5])
    #---------------------
    scoreN= round( float(metrics.r2_score(w_test, mlpPredict)),4)
    scoreM= round( float(metrics.r2_score(A_test, A_mlpPredict)),4)
    scoreO= round( float(metrics.r2_score(B_test, B_mlpPredict)),4)
    scoreG= round( float(metrics.r2_score(C_test, C_mlpPredict)),4)
    scoreF= round( float(metrics.r2_score(D_test, D_mlpPredict)),4)
    scoreS= round( float(metrics.r2_score(E_test, E_mlpPredict)),4)
    scoreU= round( float(metrics.r2_score(y_test, F_mlpPredict)),4)
    MSEN= round( float(metrics.mean_squared_error(w_test, mlpPredict)),4)
    MSEM= round( float(metrics.mean_squared_error(A_test, A_mlpPredict)),4)
    MSEO= round( float(metrics.mean_squared_error(B_test, B_mlpPredict)),4)
    MSEG= round( float(metrics.mean_squared_error(C_test, C_mlpPredict)),4)
    MSEF= round( float(metrics.mean_squared_error(D_test, D_mlpPredict)),4)
    MSES= round( float(metrics.mean_squared_error(E_test, E_mlpPredict)),4)
    MSEU= round( float( metrics.mean_squared_error(y_test, F_ranfor)),4)
    scoreNt= round( float(metrics.r2_score(m_test, m_mlpPredict)),4)
    scoreMt= round( float(metrics.r2_score(n_test, n_mlpPredict)),4)
    scoreOt= round( float(metrics.r2_score(o_test, o_mlpPredict)),4)
    scoreGt= round( float(metrics.r2_score(p_test, p_mlpPredict)),4)
    scoreFt= round( float(metrics.r2_score(q_test, q_mlpPredict)),4)
    scoreSt= round( float(metrics.r2_score(r_test, r_mlpPredict)),4)
    scoreUt= round( float(metrics.r2_score(s_test, s_mlpPredict)),4)
    scoreVt= round( float(metrics.r2_score(t_test, t_mlpPredict)),4)
    MSENt= round( float(metrics.mean_squared_error(m_test, m_mlpPredict)),4)
    MSEMt= round( float(metrics.mean_squared_error(n_test, n_mlpPredict)),4)
    MSEOt= round( float(metrics.mean_squared_error(o_test, o_mlpPredict)),4)
    MSEGt= round( float(metrics.mean_squared_error(p_test, p_mlpPredict)),4)
    MSEFt= round( float(metrics.mean_squared_error(q_test, q_mlpPredict)),4)
    MSESt= round( float(metrics.mean_squared_error(r_test, r_mlpPredict)),4)
    MSEUt= round( float( metrics.mean_squared_error(s_test, s_mlpPredict)),4)
    MSEVt= round( float( metrics.mean_squared_error(t_test, t_mlpPredict)),4)
    #------------------------------------------------------------------------random
    scorN= round( float(metrics.r2_score(w_test, y_ranfor)),4)
    scorM= round( float(metrics.r2_score(A_test, A_ranfor)),4)
    scorO= round( float(metrics.r2_score(B_test, B_ranfor)),4)
    scorG= round( float(metrics.r2_score(C_test, C_ranfor)),4)
    scorF= round( float(metrics.r2_score(D_test, D_ranfor)),4)
    scorS= round( float(metrics.r2_score(E_test, E_ranfor)),4)
    scorU= round( float(metrics.r2_score(y_test, F_ranfor)),4)
    MSN= round( float(metrics.mean_squared_error(w_test, y_ranfor)),4)
    MSM= round( float(metrics.mean_squared_error(A_test, A_ranfor)),4)
    MSO= round( float(metrics.mean_squared_error(B_test, B_ranfor)),4)
    MSG= round( float(metrics.mean_squared_error(C_test, C_ranfor)),4)
    MSF= round( float(metrics.mean_squared_error(D_test, D_ranfor)),4)
    MSS= round( float(metrics.mean_squared_error(E_test, E_ranfor)),4)
    MSU= round( float(metrics.mean_squared_error(y_test, F_ranfor)),4)
    scorNt= round( float(metrics.r2_score(m_test, m_ranfor)),4)
    scorMt= round( float(metrics.r2_score(n_test, n_ranfor)),4)
    scorOt= round( float(metrics.r2_score(o_test, o_ranfor)),4)
    scorGt= round( float(metrics.r2_score(p_test, p_ranfor)),4)
    scorFt= round( float(metrics.r2_score(q_test, q_ranfor)),4)
    scorSt= round( float(metrics.r2_score(r_test, r_ranfor)),4)
    scorUt= round( float(metrics.r2_score(s_test, s_ranfor)),4)
    scorVt= round( float(metrics.r2_score(t_test, t_ranfor)),4)
    MSNt= round( float(metrics.mean_squared_error(m_test, m_ranfor)),4)
    MSMt= round( float(metrics.mean_squared_error(n_test, n_ranfor)),4)
    MSOt= round( float(metrics.mean_squared_error(o_test, o_ranfor)),4)
    MSGt= round( float(metrics.mean_squared_error(p_test, p_ranfor)),4)
    MSFt= round( float(metrics.mean_squared_error(q_test, q_ranfor)),4)
    MSSt= round( float(metrics.mean_squared_error(r_test, r_ranfor)),4)
    MSUt= round( float(metrics.mean_squared_error(s_test, s_ranfor)),4)
    MSVt= round( float(metrics.mean_squared_error(t_test, t_ranfor)),4)
    form = AverageForm()
    context = {
        'form': form,
        #----------nrep
        'average':   Red1,
        'average11': Red2,
        'average12': Red3,
        'average13': Red4,
        'sm2n':Reds2n,
        'sm1n':Reds1n,
        'average1': sum1,
        'average2': sum2,
        'average3': sum3,
        'average4': sum4,
        #--------------Mul
        'average_1': Red12,
        'average_2': Red22,
        'average_3': Red32,
        'average_4': Red42,
        'sm2m':Reds2m,
        'sm1m':Reds1m,
        'average_11': sum12,
        'average_12': sum22,
        'average_13': sum32,
        'average_14': sum42,
        #--------------Otr
        'average_1o': Red13,
        'average_2o': Red23,
        'average_3o': Red33,
        'average_4o': Red43,
        'sm2o':Reds2o,
        'sm1o':Reds1o,
        'average_11o': sum13,
        'average_12o': sum23,
        'average_13o': sum33,
        'average_14o': sum43,
        #--------------GENoS
        'average_1g': Red14,
        'average_2g': Red24,
        'average_3g': Red34,
        'average_4g': Red44,
        'sm2g':Reds2g,
        'sm1g':Reds1g,
        'average_11g': sum14,
        'average_12g': sum24,
        'average_13g': sum34,
        'average_14g': sum44,
         #--------------FUEN
        'average_1f': Red15,
        'average_2f': Red25,
        'average_3f': Red35,
        'average_4f': Red45,
        'sm2f':Reds2f,
        'sm1f':Reds1f,
        'average_11f': sum15,
        'average_12f': sum25,
        'average_13f': sum35,
        'average_14f': sum45,
        #--------------sondas
        'average_1s': Red16,
        'average_2s': Red26,
        'average_3s': Red36,
        'average_4s': Red46,
        'sm2s':Reds2s,
        'sm1s':Reds1s,
        'average_11s': sum16,
        'average_12s': sum26,
        'average_13s': sum36,
        'average_14s': sum46,
        #--------------FUS
        'average_1u': Red17,
        'average_2u': Red27,
        'average_3u': Red37,
        'average_4u': Red47,
        'sm2': Reds2,
        'sm1': Reds1,
        'average_11u': sum17,
        'average_12u': sum27,
        'average_13u': sum37,
        'average_14u': sum47,
        #----------------------------------
        'scoreN':scoreN,
        'scoreM':scoreM,
        'scoreO':scoreO,
        'scoreG':scoreG,
        'scoreF':scoreF,
        'scoreS':scoreS,
        'scoreU':scoreU,
        'MSEN':MSEN,
        'MSEM':MSEM,
        'MSEO':MSEO,
        'MSEG':MSEG,
        'MSEF':MSEF,
        'MSES':MSES,
        'MSEU':MSEU,
        #----------------------------------
        'scorN':scorN,
        'scorM':scorM,
        'scorO':scorO,
        'scorG':scorG,
        'scorF':scorF,
        'scorS':scorS,
        'scorU':scorU,
        'MSN':MSN,
        'MSM':MSM,
        'MSO':MSO,
        'MSG':MSG,
        'MSF':MSF,
        'MSS':MSS,
        'MSU':MSU,
        #--------------ti_0
        'averag_1m': Rem17,
        'averag_2m': Rem27,
        'averag_3m': Rem37,
        'averag_4m': Rem47,
        'smm2': Rem2,
        'smm1': Rem1,
       'averag_11m': su17,
        'averag_12m': su27,
        'averag_13m': su37,
        'averag_14m': su47,
        #--------------ti_1
        'averag_1n': Ren17,
        'averag_2n': Ren27,
        'averag_3n': Ren37,
        'averag_4n': Ren47,
        'smn2': Ren2,
        'smn1': Ren1,
        'averag_11n': su17n,
        'averag_12n': su27n,
        'averag_13n': su37n,
        'averag_14n': su47n,
         #--------------ti_2
        'averag_1o': Reo17,
        'averag_2o': Reo27,
        'averag_3o': Reo37,
        'averag_4o': Reo47,
        'smo2': Reo2,
        'smo1': Reo1,
        'averag_11o': su17o,
        'averag_12o': su27o,
        'averag_13o': su37o,
        'averag_14o': su47o,
        #--------------ti_3
        'averag_1p': Rep17,
        'averag_2p': Rep27,
        'averag_3p': Rep37,
        'averag_4p': Rep47,
        'smp2': Rep2,
        'smp1': Rep1,
        'averag_11p': su17p,
        'averag_12p': su27p,
        'averag_13p': su37p,
        'averag_14p': su47p,
        #--------------ti_4
        'averag_1q': Req17,
        'averag_2q': Req27,
        'averag_3q': Req37,
        'averag_4q': Req47,
        'smq2': Req2,
        'smq1': Req1,
        'averag_11q': su17q,
        'averag_12q': su27q,
        'averag_13q': su37q,
        'averag_14q': su47q,
        #--------------ti_5
        'averag_1r': Rer17,
        'averag_2r': Rer27,
        'averag_3r': Rer37,
        'averag_4r': Rer47,
        'smr2': Rer2,
        'smr1': Rer1,
        'averag_11r': su17r,
        'averag_12r': su27r,
        'averag_13r': su37r,
        'averag_14r': su47r,
         #--------------ti_6
        'averag_1s': Res17,
        'averag_2s': Res27,
        'averag_3s': Res37,
        'averag_4s': Res47,
        'sms2': Res2,
        'sms1': Res1,
        'averag_11s': su17s,
        'averag_12s': su27s,
        'averag_13s': su37s,
        'averag_14s': su47s,
        #--------------ti_7
        'averag_1t': Ret17,
        'averag_2t': Ret27,
        'averag_3t': Ret37,
        'averag_4t': Ret47,
        'smt2': Ret2,
        'smt1': Ret1,
        'averag_11t': su17t,
        'averag_12t': su27t,
        'averag_13t': su37t,
        'averag_14t': su47t,
        'scoreNt':scoreNt,
        'scoreMt':scoreMt,
        'scoreOt':scoreOt,
        'scoreGt':scoreGt,
        'scoreFt':scoreFt,
        'scoreSt':scoreSt,
        'scoreUt':scoreUt,
        'scoreVt':scoreVt,
        'MSENt':MSENt,
        'MSEMt':MSEMt,
        'MSEOt':MSEOt,
        'MSEGt':MSEGt,
        'MSEFt':MSEFt,
        'MSESt':MSESt,
        'MSEUt':MSEUt,
        'MSEVt':MSEVt,
        #----------------------------------
        'scorNt':scorNt,
        'scorMt':scorMt,
        'scorOt':scorOt,
        'scorGt':scorGt,
        'scorFt':scorFt,
        'scorSt':scorSt,
        'scorUt':scorUt,
        'scorVt':scorVt,
        'MSNt':MSNt,
        'MSMt':MSMt,
        'MSOt':MSOt,
        'MSGt':MSGt,
        'MSFt':MSFt,
        'MSSt':MSSt,
        'MSUt':MSUt,
        'MSVt':MSVt,
   }
    return render(request, 'app/about.html', context)
#=======================Prediccion Uso Equipos=============================
def Tres(request):
    warnings.simplefilter('ignore')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT * FROM tmachinefinal ORDER BY añoi,numsem ASC"
    df = pd.read_sql(query, conn)
    Data=df
    frames = [Data]
    datos = pd.concat(frames)
    datos.head()
       
    #____________________________________________________________________________________________________________
    X = datos.drop(['tipo_0'], axis=1).copy()
    y = datos[['tipo_0']]
    X_train, X_test, y_train, m_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    m_mlpPredict = mlp.predict(X_test)
    m_mlpPredict = m_mlpPredict.round()
        
    #__________________________________________________________________________________________________________________________
    X = datos.drop(['tipo_1'], axis=1).copy()
    y = datos[['tipo_1']]
    X_train, X_test, y_train, n_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    n_mlpPredict = mlp.predict(X_test)
    n_mlpPredict = n_mlpPredict.round()
     
    
    #_______________________________________________________________________________________________
    X = datos.drop(['tipo_2'], axis=1).copy()
    y = datos[['tipo_2']]
    X_train, X_test, y_train, o_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    o_mlpPredict = mlp.predict(X_test)
    o_mlpPredict = o_mlpPredict.round()
     
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_3'], axis=1).copy()
    y = datos[['tipo_3']]
    X_train, X_test, y_train, p_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    p_mlpPredict = mlp.predict(X_test)
    p_mlpPredict = p_mlpPredict.round()
  
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_4'], axis=1).copy()
    y = datos[['tipo_4']]
    X_train, X_test, y_train, q_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    q_mlpPredict = mlp.predict(X_test)
    q_mlpPredict = q_mlpPredict.round()
   
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_5'], axis=1).copy()
    y = datos[['tipo_5']]
    X_train, X_test, y_train, r_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    r_mlpPredict = mlp.predict(X_test)
    r_mlpPredict = r_mlpPredict.round()
 
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_6'], axis=1).copy()
    y = datos[['tipo_6']]
    X_train, X_test, y_train, s_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    s_mlpPredict = mlp.predict(X_test)
    s_mlpPredict = s_mlpPredict.round()
    #____________________________________________________________________________________________
    X = datos.drop(['tipo_7'], axis=1).copy()
    y = datos[['tipo_7']]
    X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=1000,
    learning_rate='adaptive', learning_rate_init=0.001)
    mlp.fit(X_train, y_train)
    t_mlpPredict = mlp.predict(X_test)
    t_mlpPredict = t_mlpPredict.round()
  
    #____________________________________________________________________________________________
   #___tipo_0
    Rem17=    int(m_mlpPredict[1:2])
    Rem27=    int(m_mlpPredict[2:3])
    Rem37=    int(m_mlpPredict[3:4])
    Rem47=    int(m_mlpPredict[4:5])
    Rem2=    int(m_test.iloc[1]['tipo_0'])
    Rem1=    int(m_test.iloc[2]['tipo_0'])
    
    #___tipo_1
    Ren17=    int(n_mlpPredict[1:2])
    Ren27=    int(n_mlpPredict[2:3])
    Ren37=    int(n_mlpPredict[3:4])
    Ren47=    int(n_mlpPredict[4:5])
    Ren2=    int(n_test.iloc[1]['tipo_1'])
    Ren1=    int(n_test.iloc[2]['tipo_1'])
    
    #___tipo_2
    Reo17=    int(o_mlpPredict[1:2])
    Reo27=    int(o_mlpPredict[2:3])
    Reo37=    int(o_mlpPredict[3:4])
    Reo47=    int(o_mlpPredict[4:5])
    Reo2=    int(o_test.iloc[1]['tipo_2'])
    Reo1=    int(o_test.iloc[2]['tipo_2'])
   
    #___tipo_3
    Rep17=    int(p_mlpPredict[1:2])
    Rep27=    int(p_mlpPredict[2:3])
    Rep37=    int(p_mlpPredict[3:4])
    Rep47=    int(p_mlpPredict[4:5])
    Rep2=    int(p_test.iloc[1]['tipo_3'])
    Rep1=    int(p_test.iloc[2]['tipo_3'])
    
     #___tipo_4
    Req17=    int(q_mlpPredict[1:2])
    Req27=    int(q_mlpPredict[2:3])
    Req37=    int(q_mlpPredict[3:4])
    Req47=    int(q_mlpPredict[4:5])
    Req2=    int(q_test.iloc[1]['tipo_4'])
    Req1=    int(q_test.iloc[2]['tipo_4'])
    
     #___tipo_5
    Rer17=    int(r_mlpPredict[1:2])
    Rer27=    int(r_mlpPredict[2:3])
    Rer37=    int(r_mlpPredict[3:4])
    Rer47=    int(r_mlpPredict[4:5])
    Rer2=    int(r_test.iloc[1]['tipo_5'])
    Rer1=    int(r_test.iloc[2]['tipo_5'])
    
    #___tipo_6
    Res17=    int(s_mlpPredict[1:2])
    Res27=    int(s_mlpPredict[2:3])
    Res37=    int(s_mlpPredict[3:4])
    Res47=    int(s_mlpPredict[4:5])
    Res2=    int(s_test.iloc[1]['tipo_6'])
    Res1=    int(s_test.iloc[2]['tipo_6'])
    
    #___tipo_7
    Ret17=   int(t_mlpPredict[1:2])
    Ret27=   int(t_mlpPredict[2:3])
    Ret37=   int(t_mlpPredict[3:4])
    Ret47=   int(t_mlpPredict[4:5])
    Ret2=    int(t_test.iloc[1]['tipo_7'])
    Ret1=    int(t_test.iloc[2]['tipo_7'])
   
    form = AverageForm()
    context = {
        'form': form,
        
        #--------------ti_0
        'averag_1m': Rem17,
        'averag_2m': Rem27,
        'averag_3m': Rem37,
        'averag_4m': Rem47,
        'smm2': Rem2,
        'smm1': Rem1,
        
        #--------------ti_1
        'averag_1n': Ren17,
        'averag_2n': Ren27,
        'averag_3n': Ren37,
        'averag_4n': Ren47,
        'smn2': Ren2,
        'smn1': Ren1,
        
         #--------------ti_2
        'averag_1o': Reo17,
        'averag_2o': Reo27,
        'averag_3o': Reo37,
        'averag_4o': Reo47,
        'smo2': Reo2,
        'smo1': Reo1,
        
        #--------------ti_3
        'averag_1p': Rep17,
        'averag_2p': Rep27,
        'averag_3p': Rep37,
        'averag_4p': Rep47,
        'smp2': Rep2,
        'smp1': Rep1,
        
        #--------------ti_4
        'averag_1q': Req17,
        'averag_2q': Req27,
        'averag_3q': Req37,
        'averag_4q': Req47,
        'smq2': Req2,
        'smq1': Req1,
        
        #--------------ti_5
        'averag_1r': Rer17,
        'averag_2r': Rer27,
        'averag_3r': Rer37,
        'averag_4r': Rer47,
        'smr2': Rer2,
        'smr1': Rer1,
        
         #--------------ti_6
        'averag_1s': Res17,
        'averag_2s': Res27,
        'averag_3s': Res37,
        'averag_4s': Res47,
        'sms2': Res2,
        'sms1': Res1,
         
        #--------------ti_7
        'averag_1t': Ret17,
        'averag_2t': Ret27,
        'averag_3t': Ret37,
        'averag_4t': Ret47,
        'smt2': Ret2,
        'smt1': Ret1,
        
    }
    return render(request, 'app/contact.html', context)
#================= Equipos m2
def Cuat(request):
    warnings.simplefilter('ignore')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT * FROM tmachinefinal ORDER BY añoi,numsem ASC"
    df = pd.read_sql(query, conn)
    Data=df
    frames = [Data]
    datos = pd.concat(frames)
    datos.head()
       
    #____________________________________________________________________________________________________________
    X = datos.drop(['tipo_0'], axis=1).copy()
    y = datos[['tipo_0']]
    X_train, X_test, y_train, m_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    m_ranfor = RanFor.predict(X_test)
    m_ranfor = m_ranfor.round()
    
    
 
    #__________________________________________________________________________________________________________________________
    X = datos.drop(['tipo_1'], axis=1).copy()
    y = datos[['tipo_1']]
    X_train, X_test, y_train, n_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
   
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    n_ranfor = RanFor.predict(X_test)
    n_ranfor = n_ranfor.round()
   
    
    
    #_______________________________________________________________________________________________
    X = datos.drop(['tipo_2'], axis=1).copy()
    y = datos[['tipo_2']]
    X_train, X_test, y_train, o_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    o_ranfor = RanFor.predict(X_test)
    o_ranfor = o_ranfor.round()
    
   
    
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_3'], axis=1).copy()
    y = datos[['tipo_3']]
    X_train, X_test, y_train, p_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
   
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    p_ranfor = RanFor.predict(X_test)
    p_ranfor = p_ranfor.round()
    
       
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_4'], axis=1).copy()
    y = datos[['tipo_4']]
    X_train, X_test, y_train, q_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    q_ranfor = RanFor.predict(X_test)
    q_ranfor = q_ranfor.round()
   
    
   
    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_5'], axis=1).copy()
    y = datos[['tipo_5']]
    X_train, X_test, y_train, r_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    r_ranfor = RanFor.predict(X_test)
    r_ranfor = r_ranfor.round()
    
   

    #_________________________________________________________________________________________________
    X = datos.drop(['tipo_6'], axis=1).copy()
    y = datos[['tipo_6']]
    X_train, X_test, y_train, s_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    s_ranfor = RanFor.predict(X_test)
    s_ranfor = s_ranfor.round()
   
   #____________________________________________________________________________________________
    X = datos.drop(['tipo_7'], axis=1).copy()
    y = datos[['tipo_7']]
    X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=45)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    RanFor = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
    RanFor.fit(X_train, y_train)
    t_ranfor = RanFor.predict(X_test)
    t_ranfor = t_ranfor.round()
   
    
   
    #____________________________________________________________________________________________
   #___tipo_0
    
    Rem2=    int(m_test.iloc[1]['tipo_0'])
    Rem1=    int(m_test.iloc[2]['tipo_0'])
    su17=   int(m_ranfor[1:2])
    su27=   int(m_ranfor[2:3])
    su37=   int(m_ranfor[3:4])
    su47=   int(m_ranfor[4:5])
    #___tipo_1
   
    Ren2=    int(n_test.iloc[1]['tipo_1'])
    Ren1=    int(n_test.iloc[2]['tipo_1'])
    su17n=   int(n_ranfor[1:2])
    su27n=   int(n_ranfor[2:3])
    su37n=   int(n_ranfor[3:4])
    su47n=   int(n_ranfor[4:5])
    #___tipo_2
    
    Reo2=    int(o_test.iloc[1]['tipo_2'])
    Reo1=    int(o_test.iloc[2]['tipo_2'])
    su17o=   int(o_ranfor[1:2])
    su27o=   int(o_ranfor[2:3])
    su37o=   int(o_ranfor[3:4])
    su47o=   int(o_ranfor[4:5])
    #___tipo_3
    
    Rep2=    int(p_test.iloc[1]['tipo_3'])
    Rep1=    int(p_test.iloc[2]['tipo_3'])
    su17p=   int(p_ranfor[1:2])
    su27p=   int(p_ranfor[2:3])
    su37p=   int(p_ranfor[3:4])
    su47p=   int(p_ranfor[4:5])
     #___tipo_4
    
    Req2=    int(q_test.iloc[1]['tipo_4'])
    Req1=    int(q_test.iloc[2]['tipo_4'])
    su17q=   int(q_ranfor[1:2])
    su27q=   int(q_ranfor[2:3])
    su37q=   int(q_ranfor[3:4])
    su47q=   int(q_ranfor[4:5])
     #___tipo_5
   
    Rer2=    int(r_test.iloc[1]['tipo_5'])
    Rer1=    int(r_test.iloc[2]['tipo_5'])
    su17r=   int(r_ranfor[1:2])
    su27r=   int(r_ranfor[2:3])
    su37r=   int(r_ranfor[3:4])
    su47r=   int(r_ranfor[4:5])
    #___tipo_6
   
    Res2=    int(s_test.iloc[1]['tipo_6'])
    Res1=    int(s_test.iloc[2]['tipo_6'])
    su17s=   int(s_ranfor[1:2])
    su27s=   int(s_ranfor[2:3])
    su37s=   int(s_ranfor[3:4])
    su47s=   int(s_ranfor[4:5])
    #___tipo_7
    
    Ret2=    int(t_test.iloc[1]['tipo_7'])
    Ret1=    int(t_test.iloc[2]['tipo_7'])
    su17t=   int(t_ranfor[1:2])
    su27t=   int(t_ranfor[2:3])
    su37t=   int(t_ranfor[3:4])
    su47t=   int(t_ranfor[4:5])
    form = AverageForm()
    context = {
        'form': form,
        
        #--------------ti_0
        
        'smm2': Rem2,
        'smm1': Rem1,
        'averag_11m': su17,
        'averag_12m': su27,
        'averag_13m': su37,
        'averag_14m': su47,
        #--------------ti_1
        
        'smn2': Ren2,
        'smn1': Ren1,
        'averag_11n': su17n,
        'averag_12n': su27n,
        'averag_13n': su37n,
        'averag_14n': su47n,
         #--------------ti_2
        
        'smo2': Reo2,
        'smo1': Reo1,
        'averag_11o': su17o,
        'averag_12o': su27o,
        'averag_13o': su37o,
        'averag_14o': su47o,
        #--------------ti_3
        
        'smp2': Rep2,
        'smp1': Rep1,
        'averag_11p': su17p,
        'averag_12p': su27p,
        'averag_13p': su37p,
        'averag_14p': su47p,
        #--------------ti_4
        
        'smq2': Req2,
        'smq1': Req1,
        'averag_11q': su17q,
        'averag_12q': su27q,
        'averag_13q': su37q,
        'averag_14q': su47q,
        #--------------ti_5
        
        'smr2': Rer2,
        'smr1': Rer1,
        'averag_11r': su17r,
        'averag_12r': su27r,
        'averag_13r': su37r,
        'averag_14r': su47r,
         #--------------ti_6
        
        'sms2': Res2,
        'sms1': Res1,
         'averag_11s': su17s,
        'averag_12s': su27s,
        'averag_13s': su37s,
        'averag_14s': su47s,
        #--------------ti_7
        
        'smt2': Ret2,
        'smt1': Ret1,
        'averag_11t': su17t,
        'averag_12t': su27t,
        'averag_13t': su37t,
        'averag_14t': su47t,
    }
    return render(request, 'app/pred2.html', context)
##=======================Tabla ============================================
def Cuatro(request):
    warnings.simplefilter('ignore')
    conn= psycopg2.connect(database ="d96jei46kjtrqh", user="jcurhhlrwaxdbk", password="624d7d69b068ce2ba1437a65918eb6f115698fcfbcde0e7099a955759af0f90f", host="ec2-54-235-89-123.compute-1.amazonaws.com", port="5432")
    query="SELECT año FROM tabla ORDER BY año,mes , uso DESC"
    query1= "SELECT mes FROM tabla ORDER BY año,mes , uso DESC"
    query2= "SELECT uso FROM tabla ORDER BY año,mes , uso DESC"
    query3= "SELECT equipo_nombre FROM tabla ORDER BY año,mes , uso DESC"

    df  = pd.read_sql(query, conn)
    mf  = pd.read_sql(query1, conn)
    uf  = pd.read_sql(query2, conn)
    ef  = pd.read_sql(query3, conn)
    equipos=[0]*10
    for i in range(0, 10):
        var=str(ef[i:i+1]).strip()
        var=var[16:].strip()
        equipos[i]=var

    form = AverageForm()
    context = {
        'form': form,
        'aver' : (str(df[0:1]))[9:19],
        'aver1': (str(df[1:2]))[9:19],
        'aver2': (str(df[2:3]))[9:19],
        'aver3': (str(df[3:4]))[9:19],
        'aver4': (str(df[4:5]))[9:19],
        'aver5': (str(df[5:6]))[9:19],
        'aver6': (str(df[6:7]))[9:19],
        'aver7': (str(df[7:8]))[9:19],
        'aver8': (str(df[8:9]))[9:19],
        'aver9': (str(df[9:10]))[9:19],

        'Nmes' : (str(mf[0:1]))[9:19],
        'Nmes1': (str(mf[1:2]))[9:19],
        'Nmes2': (str(mf[2:3]))[9:19],
        'Nmes3': (str(mf[3:4]))[9:19],
        'Nmes4': (str(mf[4:5]))[9:19],
        'Nmes5': (str(mf[5:6]))[9:19],
        'Nmes6': (str(mf[6:7]))[9:19],
        'Nmes7': (str(mf[7:8]))[9:19],
        'Nmes8': (str(mf[8:9]))[9:19],
        'Nmes9': (str(mf[9:10]))[9:19],

        'Uss' : (str(uf[0:1]))[9:19],
        'Uss1': (str(uf[1:2]))[9:19],
        'Uss2': (str(uf[2:3]))[9:19],
        'Uss3': (str(uf[3:4]))[9:19],
        'Uss4': (str(uf[4:5]))[9:19],
        'Uss5': (str(uf[5:6]))[9:19],
        'Uss6': (str(uf[6:7]))[9:19],
        'Uss7': (str(uf[7:8]))[9:19],
        'Uss8': (str(uf[8:9]))[9:19],
        'Uss9': (str(uf[9:10]))[9:19],

        'Eqn' : (str(equipos[0]))[:-2],
        'Eqn1': (str(equipos[1]))[:-2],
        'Eqn2': (str(equipos[2]))[:-2],
        'Eqn3': (str(equipos[3]))[:-2],
        'Eqn4': (str(equipos[4]))[:-2],
        'Eqn5': (str(equipos[5]))[:-2],
        'Eqn6': (str(equipos[6]))[:-2],
        'Eqn7': (str(equipos[7]))[:-2],
        'Eqn8': (str(equipos[8]))[:-2],
        'Eqn9': (str(equipos[9]))[:-2],
        }

    return render(request, 'app/UsoM.html', context)