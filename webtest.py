import json
from flask import (
					Flask, make_response, redirect,
					render_template, request, url_for
				)

app = Flask(__name__)

@app.route('/')
@app.route('/<name>')
def index(name="Safsdaf"):
	return 'Hello {}'.format(name)

@app.route('/add')
@app.route('/add/<int:num1>/<int:num2>')
@app.route('/add/<float:num1>/<int:num2>')
@app.route('/add/<int:num1>/<float:num2>')
@app.route('/add/<float:num1>/<float:num2>')
def add(num1=1, num2=2):
	data = get_saved()
	print data
	context = {
				'num1':num1, 
				'num2':num2
				}
	return render_template("add.html", saves=data, **context) 

def get_saved():
	try:
		data = json.loads(request.cookies.get('summation'))
	except TypeError:
		data = {}
	return data
		
@app.route('/save', methods=["POST"])
def save():
	response = make_response( redirect(url_for("add")) )
	data = get_saved()
	data.update(dict(request.form.items()))
	response.set_cookie("summation", json.dumps(data))
	return response
	
app.run(debug=True, port=8000, host='0.0.0.0')