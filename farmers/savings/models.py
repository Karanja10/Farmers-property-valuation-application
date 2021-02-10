from django.db import models

class Users(models.Model):
	"""docstring for Users"""
	first_name=models.CharField(max_length==100)
	last_name=models.CharField(max_length==100)
	phone=models.CharField(max_length==100)
	location=models.CharField(max_length==100)
	id_no=models.CharField(max_length==100)
	KRA_pin=models.CharField(max_length==100)
	email=models.CharField(max_length==100)
	password=models.CharField(max_length==100)
	pin=models.CharField(max_length==100)
	Saving=models.OneToOneField
		
class Saving(models.Model):
	
	transaction_history = models.CharField(max_length==100)
	loan_limit=models.CharField(max_length==100)
	usage=models.CharField(max_length==100)
	status=models.CharField(max_length==100)
	period=models.CharField(max_length==100)