import json
from flask import (
					Flask, make_response, redirect,
					render_template, request, url_for, session
				)

app = Flask(__name__)


def get_saved():
	try:
		data = json.loads(request.cookies.get('summation'))
	except TypeError:
		data = {}
	return data
	
@app.route('/save', methods=["POST"])
def save():
	response = make_response( redirect(url_for("save")) )
	data = get_saved()
	data.update(dict(request.form.items()))
	response.set_cookie("summation", json.dumps(data), expires=3)
	return response

@app.route('/')
@app.route('/<name>')
def index(name="Sean Hegarty"):
	return 'Hello from {}'.format(name)

@app.route('/add')
@app.route('/add/<int:num1>/<int:num2>')
@app.route('/add/<float:num1>/<int:num2>')
@app.route('/add/<int:num1>/<float:num2>')
@app.route('/add/<float:num1>/<float:num2>')
def add(num1=1, num2=2):
	data = get_saved()
# 	print data
	try:
		num1 = int(data['num1'])
		num2 = int(data['num2'])
	except:#  ValueError or TypeError or KeyError:
		data = {
				'num1':num1, 
				'num2':num2
				}
	
	context = {
				'num1':num1, 
				'num2':num2
				}
	return render_template("add.html", saves=data)#, **context) 
		

	
app.run(debug=True, port=8000, host='0.0.0.0')