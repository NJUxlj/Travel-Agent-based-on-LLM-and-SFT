# it is where you specify patterns to route different URLs to their appropriate views.

from django.urls import path
from hello import views




'''
下面的代码包含一条将应用程序的根 URL ("") 映射到您刚刚添加到 hello/views.py 的 views.home 函数的路由：
'''

# 路由列表
urlpatterns = [
    path("hello", views.home, name="home"),
    path("hello/<name>", views.hello_there, name="hello_there"),
]