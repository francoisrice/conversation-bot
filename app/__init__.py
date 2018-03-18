from flask import Flask

app = Flask(__name__)
#app.config.broker_url = 'redis://localhost:6379/0'     # Required if using Redis database
#app.config.result_backend = 'redis://localhost:6379/0'

app.config.update(TEMPLATES_AUTO_RELOAD=True)

# from tasks import taskname # Add the correct task name for each task run through celery
from app import views