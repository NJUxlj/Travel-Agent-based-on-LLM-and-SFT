from django.db import models

# Create your models here.

# it contains classes defining your data objects


class Gemma2bLora(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return self.title
    



class Gemma2bQlora(models.Model):
    pass


class Gemma7bLora(models.Model):
    pass



class Gemmas7bQlora(models.Model):
    pass


