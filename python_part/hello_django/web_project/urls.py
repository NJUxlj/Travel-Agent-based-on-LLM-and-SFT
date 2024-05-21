"""
URL configuration for web_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from django.contrib.staticfiles.urls import staticfiles_urlpatterns

'''
这是实际处理 URL 路由的地方
'''

'''
为什么要用include？ 为了将属于hello这个app的路由仅仅存留在该app内部，这么做是出于封装的考虑
'''

urlpatterns = [
    path("", include("hello.urls") ),
    path("admin/", admin.site.urls),
    path("llmapp/", include("LLMapp.urls")),
]


urlpatterns+=staticfiles_urlpatterns()
