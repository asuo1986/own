from django.shortcuts import render
from django.http import HttpResponse
#from django.template import loader
#from django.template import Context
from django.shortcuts import render_to_response

# Create your views here.

class Person(object):
	def __init__(self, name, age, sex):
		self.name = name
		self.age = age
		self.sex = sex
	def say(self):
		return 'this is' + self.name
'''
def index(req):
	t = loader.get_template('index.html')
	c = Context({})
	return HttpResponse(t.render(c))
'''
'''
def index(req):
	return render_to_response('index.html', {})
'''
'''
def index(req):
	return render_to_response('index.html', {'title':'mypage','user':'yanasdsadsazhang'})
'''
def index(req):
	#person = {'name':'tom', 'id':'TOMMM'}
	person = Person('zhangsan',18,'male')

	return render_to_response('index.html', {'title':'mypage','user':person})