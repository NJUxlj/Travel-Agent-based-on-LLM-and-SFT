from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render

import re
from django.utils.timezone import datetime

# Create your views here.
# it contains the functions that define pages in your web app


# 为应用程序的主页创建单个视图
def home(request):
    return HttpResponse("Hello, Django")


def hello_there(request, name):
    print(request.build_absolute_uri())
    
    return render(
        request,
        "hello/hello_there.html",
        {
            "name":name,
            "date":datetime.now()
        }
    )