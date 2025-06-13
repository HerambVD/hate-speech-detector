from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_text, name='classify_text'),
    path('health/', views.health_check, name='health_check'),
]
