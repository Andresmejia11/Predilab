"""
Definition of urls for Predi_Lab.
"""

from datetime import datetime
from django.urls import path
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from app import forms, views


urlpatterns = [
    path('', views.home, name='home'),
    path('informe/', views.informe, name='informe'),
    path('contact/', views.Tres, name='contact'),
    path('about/', views.Dos, name='about'),
    path('prueba/', views.averageProm, name='prueba'),
    path('pag1/', views.UNO, name='pag1'),
    path('UsoM/', views.Cuatro, name='UsoM'),
    path('pred2/', views.Cuat, name='pred2'),
    path('login/',
         LoginView.as_view
         (
             template_name='app/login.html',
             authentication_form=forms.BootstrapAuthenticationForm,
             extra_context=
             {
                 'title': 'Log in',
                 'year' : datetime.now().year,
             }
         ),
         name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('admin/', admin.site.urls),
]
