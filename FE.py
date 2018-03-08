#!/home/hanghang/anaconda3/bin/python3
import pandas as pd
import quandl
import sys
import os
import numpy as np
import math as mth
quandl.ApiConfig.api_key = ""

class FE:
	def __init__(self,**args):
		self.scode = args["scode"].upper()
		self.par_max = args["par_max"]+1

		raw_data = quandl.get("EOD/"+self.scode)
		print("data downloaded")
		self.cl = raw_data["Close"]
		self.op = raw_data["Open"]
		self.vol_o = raw_data["Volume"]
		self.vol = (raw_data["Volume"]+1).apply(mth.log)
		self.hi = raw_data["High"]
		self.lo = raw_data["Low"]

		c_1 = self.cl.shift(1)

		c_d = self.cl - c_1
		self.cchange = (self.cl - c_1)/c_1
		self.hchange = (self.hi - c_1)/c_1
		self.lchange = (self.lo - c_1)/c_1
		self.ochange = (self.op - c_1)/c_1
		self.vchange = self.vol - self.vol.shift(1)

		self.channels = {}

		self.U = (self.cl>c_1)*c_d
		self.D = (self.cl<c_1)*(-c_d)

		self.c_1 = c_1
		self.c_d = c_d

		try:
			self.strides = args["strides"]
		except:
			self.strides = 1

	def rstd(self,i):
		return self.cl.rolling(window = i,center = False).std()

	def rsma(self,i):
		return self.cl.rolling(window = i,center = False).mean()

	def rema(self,i):
		return self.cl.ewm(alpha = 1.0/float(i),min_periods = i).mean()

	def rhmax(self,i):
		return self.hi.rolling(window = i,center = False).max()

	def rmmin(self,i):
		return self.lo.rolling(window = i,center = False).min()



	def op_mean(self):
		self.channels["op_mean"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["op_mean"][i] = self.ochange.rolling(window = i,center = False).mean()
		
	def cl_mean(self):
		self.channels["cl_mean"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["cl_mean"][i] = self.cchange.rolling(window = i,center = False).mean()

	def hi_max(self):
		self.channels["hi_max"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["hi_max"][i] = self.hchange.rolling(window = i,center = False).max()

	def lo_min(self):
		self.channels["lo_min"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["lo_min"][i] = self.lchange.rolling(window = i,center = False).min()

	def vl_mean(self):
		self.channels["vl_mean"] = pd.DataFrame()
		self.channels["vl2_mean"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["vl_mean"][i] = (self.vol_o.rolling(window = i,center = False).mean()+1).apply(mth.log)
			self.channels["vl2_mean"][i] = self.vchange.rolling(window = i,center = False).mean()

	def kd(self):
		self.channels["%k"] = pd.DataFrame()
		self.channels[r"%d"] = pd.DataFrame()
		for i in range(1+3,self.par_max+3,self.strides):
			rsv = ((self.cl - self.rmmin(i))/(self.rhmax(i) - self.rmmin(i))).fillna(0).replace(np.inf,0)
			k = rsv.ewm(alpha = 3/float(i),min_periods = i//3).mean()
			d = k.ewm(alpha = 3/float(i),min_periods = i//3).mean()
			self.channels["%k"][i] = k
			self.channels[r"%d"][i] = d


	def dma(self):
		self.channels["dma"] = pd.DataFrame()
		for i in range(1,self.par_max,self.strides):
			self.channels["dma"][i] = (self.rsma(i) - self.cl)/self.cl

	def wr(self):
		self.channels["wr"] = pd.DataFrame()
		for i in range(1+1,self.par_max+1,self.strides):
			self.channels["wr"][i] = ((self.rhmax(i)-self.cl)/(self.rhmax(i)-self.rmmin(i))).fillna(0.5)

	def rsi(self):
		self.channels["rsi"] = pd.DataFrame()
		for i in range(1+2,self.par_max+2,self.strides):
			smau = self.U.ewm(alpha = 1.0/float(i),min_periods = i).mean()
			smad = self.D.ewm(alpha = 1.0/float(i),min_periods = i).mean()
			self.channels["rsi"][i] = 1-1/(1+smau/smad)

	def vortex(self):
		self.channels["VIP"] = pd.DataFrame()
		self.channels["VIN"] = pd.DataFrame()
		TR = pd.DataFrame({"hl":abs(self.hi-self.lo),"lc":abs(self.lo - self.c_1),"hc":abs(self.hi-self.c_1)}).max(axis = 1)
		VMP = abs(self.hi - self.lo.shift(1))
		VMN = abs(self.lo - self.hi.shift(1))

		for i in range(1+2,self.par_max+2,self.strides):
			STR = TR.rolling(window = i,center = False).sum()
			SVMP = VMP.rolling(window = i,center = False).sum()
			SVMN = VMN.rolling(window = i,center = False).sum()
			self.channels["VIP"][i] = (SVMP/STR).replace(np.inf,1).fillna(1)
			self.channels["VIN"][i] = (SVMN/STR).replace(np.inf,1).fillna(1)


	def boll(self):
		self.channels[r"%b"] = pd.DataFrame()
		self.channels["bb"] = pd.DataFrame()
		for i in range(1+2,self.par_max+2,self.strides):
			ubb = self.rsma(i)+1.5*self.rstd(i)
			lbb = self.rsma(i)-1.5*self.rstd(i)
			self.channels[r"%b"][i] = (self.cl-lbb)/(ubb-lbb+0.01)
			self.channels["bb"][i] = (ubb-lbb)/(self.rsma(i)+0.01)


	def risk(self):
		self.channels["risk"] = pd.DataFrame()
		for i in range(1+2,self.par_max+2,self.strides):
			self.channels["risk"][i] = self.cchange.rolling(window = i,center = False).std()


	def skew(self):
		self.channels["skew"] = pd.DataFrame()
		for i in range(1+3,self.par_max+3,self.strides):
			self.channels["skew"][i] = (self.cchange.rolling(window = i,center = False).skew()).apply(lambda x:max(min(10,x),-10))


	def kurt(self):
		self.channels["kurt"] = pd.DataFrame()
		for i in range(1+4,self.par_max+4,self.strides):
			self.channels["kurt"][i] = (self.cchange.rolling(window = i,center = False).kurt()).apply(lambda x:max(min(10,x),-10))


	def generate(self):
		print("op mean...",end = " ")
		self.op_mean()
		print("done")
		print("cl mean...",end = " ")
		self.cl_mean()
		print("done")
		print("hi max...",end = " ")
		self.hi_max()
		print("done")
		print("lo min...",end = " ")
		self.lo_min()
		print("done")
		print("vl mean...",end = " ")
		self.vl_mean()
		print("done")
		print("wr...",end = " ")
		self.wr()
		print("done")
		print("rsi...",end = " ")
		self.rsi()
		print("done")
		print("vortex...",end = " ")
		self.vortex()
		print("done")
		print("boll...",end = " ")
		self.boll()
		print("done")
		print("risk...",end = " ")
		self.risk()
		print("done")
		print("skew...",end = " ")
		self.skew()
		print("done")
		print("kurt...",end = " ")
		self.kurt()
		print("done")
		print("dma...",end = " ")
		self.dma()
		print("done")
		print("kd...",end = " ")
		self.kd()
		print("done")

	def save(self):
		fl_name = self.scode+str(self.par_max-1)+"_"+str(self.strides)+"_"+"/"
		if not os.path.exists(fl_name):
			os.mkdir(fl_name)

		for c in self.channels:
			self.channels[c].iloc[self.par_max+5:].to_csv(fl_name+c+".csv")

	
