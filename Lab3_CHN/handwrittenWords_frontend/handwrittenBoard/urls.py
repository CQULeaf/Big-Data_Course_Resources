from django.urls import path
from . import views

urlpatterns = [
    path('handwrittenBoard/', views.handwrittenBoard),  # 修改url为path
]
