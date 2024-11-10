from django.contrib import admin
from django.urls import path, include
from handwrittenWords_frontend.views import handwrittenBoard

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', handwrittenBoard, name='handwrittenBoard'),
    path('', include('handwrittenBoard.urls')),
]