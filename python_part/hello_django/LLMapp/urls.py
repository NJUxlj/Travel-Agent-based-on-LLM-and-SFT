from django.urls import path
from . import views
    
    
urlpatterns = [
    path("llm",views.llm, name = "llm"),
    path("unity",views.handle_unity_request, name = "handle_unity_request"),
]