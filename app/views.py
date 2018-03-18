from app import app
from flask import render_template
import time
#from app import MainFunction
#from tasks import tasknames
#import redis # if using Redis as a database
import pandas as pd

@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')