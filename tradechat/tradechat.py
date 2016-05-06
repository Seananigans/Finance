import os
import datetime as dt
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
					abort, render_template, flash
					
app = Flask(__name__)

app.config.update(
	dict(
		DATABASE=os.path.join(app.root_path, 'tradechat.db'),
		DEBUG=True,
		SECRET_KEY = 'secret_key'
		)
	)
	
app.config.from_envvar('TC_SETTINGS', silent=True)

def connect_db():
	#connects to the TC database
	rv = sqlite3.connect(app.config['DATABASE'])
	rv.row_factory = sqlite3.Row
	return rv
	
def get_db():
	if not hasattr(g, 'sqlite_db'):
		#open only if none exists yet
		g.sqlite_db = connect_db()
	return g.sqlite_db
	
def init_db():
	with app.app_context():
		db = get_db()
		with app.open_resource('tables.sql', 'r') as f:
			db.cursor().executescript(f.read())
		db.commit()
		
@app.teardown_appcontext
def close_db():
	''' Closes the TC database at the end of the request. '''
	if hasattr(g, 'sqlite_db'):
		g.sqlite_db.close()
		
		
@app.route('/')
def show_entries():
	db = get_db()
	query = 'select comment, user, time from comments order by id desc'
	cursor = db.execute(query)
	comments = cursor.fetchall()
	return render_template('show_entries.html', comments = comments)