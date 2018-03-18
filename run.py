from app import app
from app import tasks

if __name__ == '__main__':
	#tasks.refresh.delay() #tasks that are run on the Scheduler
	app.run(host='0.0.0.0',port=505,debug=True)
