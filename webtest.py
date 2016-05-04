import json
from flask import (
					Flask, 
					redirect, 
					render_template, 
					request, url_for
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
	context = {
				'num1':num1, 
				'num2':num2
				}
	return render_template("add.html", **context)

@app.route('/save', methods=["POST"])
def save():
	return redirect(url_for("add"))
	
app.run(debug=True, port=8000, host='0.0.0.0')