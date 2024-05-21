from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
# Create your views here.



# Views are python functons

# how can we call the views: through URL


def llm(request):
    # return HttpResponse("This is the place that we are going to handle the LLM response transfer")
    template = loader.get_template("myfirst.html")
    return HttpResponse(template.render())




def handle_unity_request(request):
    pass