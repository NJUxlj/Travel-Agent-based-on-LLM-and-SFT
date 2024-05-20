from django.db import models

# Create your models here.




class Gemma2bLora(models.Model):
    # 每个模型都自动包含一个子增幅的id'字段
    prompt = models.CharField(max_length=3000)
    response = models.TextField()

    def __str__(self):
        return self.title
    



class Gemma2bQlora(models.Model):
    prompt = models.CharField(max_length=3000)
    response = models.TextField()

    def __str__(self):
        return self.title


class Gemma7bLora(models.Model):
    prompt = models.CharField(max_length=3000)
    response = models.TextField()

    def __str__(self):
        return self.title



class Gemmas7bQlora(models.Model):
    prompt = models.CharField(max_length=3000)
    response = models.TextField()

    def __str__(self):
        return self.title
