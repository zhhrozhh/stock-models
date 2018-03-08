import numpy as np
import pandas as pd
import math as mth
import sys
import os

class DataLoader:
	def __init__(self,**args):
		self.fl_name = args['data_name']+'/'
		ds = []
		for c in ["op_mean","cl_mean","hi_max","lo_min","vl_mean","vl2_mean","wr","rsi","VIP","VIN",r"%b","bb","risk","skew","kurt","dma","%k",r"%d"]:
			ds.append(pd.DataFrame.from_csv(self.fl_name+c+".csv").values)
		self.raw_data = np.stack(ds,axis  = 2)
		self.mode = args['mode']
		assert(self.mode in ['reg','cla',"ud"])
		self.p_day = args['p_day']
		self.train_prob = args['train_prob']

	def one_hot(self,rubric,inp):
		res = np.zeros(len(rubric)+1)
		if inp < rubric[0]:
			res[0] = 1
			return res
		if inp > rubric[-1]:
			res[-1] = 1
			return res
		for i in range(1,len(rubric)):
			if inp >= rubric[i-1] and inp < rubric[i]:
				res[i] = 1
				return res
		return None 

	def __call__(self,**args):
		if self.mode == 'cla':
			rubric = args['rubric']
		p_type = args["p_type"]
		if p_type == 'close':
			p_type = 1
		elif p_type == 'high':
			p_type = 2
		elif p_type == 'low':
			p_type = 3
		elif p_type == "all":
			p_type = range(1,4)
		elif p_type == "hl":
			p_type = np.array([2,3])
		else:
			pass

		X_train = []
		Y_train = []
		X_test = []
		Y_test = []
		L = self.raw_data.shape[0]
		for i in range(L-self.p_day-1):
			if np.random.rand() <= self.train_prob:
				X_train.append(self.raw_data[i:i+self.p_day,:,:])
				if self.mode == 'reg':
					Y_train.append(self.raw_data[i+self.p_day,0,p_type])
				elif self.mode == 'cla':
					Y_train.append(self.one_hot(rubric,self.raw_data[i+self.p_day,0,p_type]))
				elif self.mode == 'ud':
					if self.raw_data[i+self.p_day,0,1] > 0.01:
						Y_train.append(np.array([1.0,0.0,0.0]))
					elif self.raw_data[i+self.p_day,0,1] < -0.01:
						Y_train.append(np.array([0.0,0.0,1.0]))
					else:
						Y_train.append(np.array([0.0,1.0,0.0]))
			else:
				X_test.append(self.raw_data[i:i+self.p_day,:,:])
				if self.mode == 'reg':
					Y_test.append(self.raw_data[i+self.p_day,0,p_type])
				elif self.mode == 'cla':
					Y_test.append(self.one_hot(rubric,self.raw_data[i+self.p_day,0,p_type]))
				elif self.mode == 'ud':
					if self.raw_data[i+self.p_day,0,1] > 0.01:
						Y_test.append(np.array([1.0,0.0,0.0]))
					elif self.raw_data[i+self.p_day,0,1] < -0.01:
						Y_test.append(np.array([0.0,0.0,1.0]))
					else:
						Y_test.append(np.array([0.0,1.0,0.0]))
		try:
			X_train = np.stack(X_train)
		except:
			pass
		try:
			Y_train = np.stack(Y_train)
		except:
			pass
		try:
			X_test = np.stack(X_test)
		except:
			pass
		try:
			Y_test = np.stack(Y_test)
		except:
			pass
		return {"X_tr":X_train,"Y_tr":Y_train,"X_te":X_test,"Y_te":Y_test}