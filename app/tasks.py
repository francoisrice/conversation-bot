from app import celery
# from app import Nirvana as nv # This space is for the main application or other applications that 
#need to run asynchronosly
import json, time
import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler # Scheduler to run autonomous functions
from apscheduler.triggers.interval import IntervalTrigger

# RedisClient = redis.StrictRedis(host='localhost', port=6379, db=0) # Use this if using a Redis database

@celery.task(name='importFunction.taskname')
def taskname(args):
	return True

@celery.task(name='importFunction.taskname2')
def taskname2(): # This is an example for the set up of a autonomous/prescheduled function
	def wrapper(): 
		return True
	scheduler = BackgroundScheduler()
	scheduler.start()
	scheduler.add_job(
		func= wrapper,
		trigger= IntervalTrigger(seconds=5*3600)
		)
	return True