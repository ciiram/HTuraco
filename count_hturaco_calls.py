import numpy as np
import scipy as sp
import pylab as pb
from scipy import signal
import scipy.io.wavfile
import time
import sys
import os
import bob
import spectral_features as sf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances,accuracy_score
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import roc_curve,auc
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def moving_avg(x,L):
	mv_avg=np.zeros(len(x))
	for i in range(len(x)):
		if i<L:
			mv_avg[i]=np.mean(x[:i+1])
		else:
			mv_avg[i]=np.sum(x[i-L:i+1])/L
	return mv_avg	


#load models
myh5_file = bob.io.HDF5File('Models/MfccTuracoModel.hdf5')
mfcc_gmm_turaco = bob.machine.GMMMachine(myh5_file)
myh5_file = bob.io.HDF5File('Models/MfccNoTuracoModel.hdf5')
mfcc_gmm_noturaco = bob.machine.GMMMachine(myh5_file)


#MFCC parameters
win_length_ms = 32 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 16 # The window shift of the cepstral analysis in milliseconds
n_filters = 41 # The number of filter bands
n_ceps = 19 # The number of cepstral coefficients
f_min = 350. # The minimal frequency of the filter bank
f_max = 6000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale

start_time=time.time()
blk_win=300 #300 blocks is approximately 5 seconds
rec_len=60#60 second recordings
num_files=0
#evaluate performance on test data

file1=open('Location4_Annotation.csv',"r")#Filenames of wav files

wav_dir='Data/Location-4/'


llr_turaco=np.array([])
num_calls=[]
times_dt=[] #date and time

MfccThreshold=0.98
Thresh_DetectFunc=2.6
line1=file1.readline()#header



print 'Screening files ...'
while file1:

	
	
	line1=file1.readline()
	s1=line1.split(',')
	
	if len(line1)==0:
		break
	
	
	#get audio file
	try:
		x= scipy.io.wavfile.read(wav_dir+s1[0]+'.wav')
	except IOError:
		print 'File',s1[0],'not found'
		continue
	fs=float(x[0])
	x2=x[1]/2.**15
	x2=x2/np.max(np.abs(x2)) 

	c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	mfcc = c(x2*2**15)#normalize to integer

	llr_blk_mfcc=np.zeros(mfcc.shape[0])
	for i in range(mfcc.shape[0]):
		llt=mfcc_gmm_turaco.log_likelihood(mfcc[i,:])
		llnt=mfcc_gmm_noturaco.log_likelihood(mfcc[i,:])
		llr_blk_mfcc[i]=llt-llnt

	y_mfcc=moving_avg(llr_blk_mfcc,blk_win)
	
	if np.mean(y_mfcc)>MfccThreshold:
		CallDetect=(y_mfcc>Thresh_DetectFunc)*Thresh_DetectFunc
		CallDetect=CallDetect.astype('int')
		num_calls.append(sum(np.diff(CallDetect)>0))
		print s1[0],sum(np.diff(CallDetect)>0)
	else:
		num_calls.append(0)
	
	#get time 	
	t2=datetime.datetime(int(s1[0].split('-')[1]),int(s1[0].split('-')[2]),int(s1[0].split('-')[3]),int(s1[0].split('-')[4]),int(s1[0].split('-')[5]),int(s1[0].split('-')[6]))
	times_dt.append(t2)	
		
	
	


file1.close()

num_calls=np.array(num_calls)
plt.figure()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H:%M'))
plt.plot(times_dt,num_calls,'-bs',linewidth=2)
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('Number of calls per minute')
plt.ylim([0,np.max(num_calls)+2])
plt.savefig('figures/htcalls-permin.png',dpi=300)
plt.show()


