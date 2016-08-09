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
myh5_file = bob.io.HDF5File('Models/SpecTuracoModel.hdf5')
spec_gmm_turaco = bob.machine.GMMMachine(myh5_file)
myh5_file = bob.io.HDF5File('Models/SpecNoTuracoModel.hdf5')
spec_gmm_noturaco = bob.machine.GMMMachine(myh5_file)
myh5_file = bob.io.HDF5File('Models/MfccTuracoModel.hdf5')
mfcc_gmm_turaco = bob.machine.GMMMachine(myh5_file)
myh5_file = bob.io.HDF5File('Models/MfccNoTuracoModel.hdf5')
mfcc_gmm_noturaco = bob.machine.GMMMachine(myh5_file)

#spectral feature parameters
N=512
alpha=0.5
upper_lim=6000
lower_lim=350
fs=16e3
Freqs=(np.arange(N/2+1)/(N/2.))*(fs/2)
win_size=32e-3 # window size in seconds
num_bands=6
rolloff_pct=.85

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
num_calls=np.array([])
times_dt=[] #date and time

SpecThreshold=0.747
MfccThreshold=0.98
neq=0
htcall_eq=0
htcall_neq=0
Thresh_DetectFunc=2

File_HTcall=[]
File_NoHTcall=[]
File_HTcallNeq=[]

#read header
line1=file1.readline()
file2=open('Data/Location4_Screening.txt','w')
file3=open('Data/Location4_Screening_HT.txt','w')
file4=open('Data/Location4_Screening_Neq.txt','w')
file2.write('File\tTuraco_Call_Spec(Yes/No)\tTuraco_Call_Mfcc(Yes/No)\n')
file3.write('File\tTuraco_Call_Spec(Yes/No)\tTuraco_Call_Mfcc(Yes/No)\n')
file4.write('File\tTuraco_Call_Spec(Yes/No)\tTuraco_Call_Mfcc(Yes/No)\n')
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
	

	#compute features
	spec_features=sf.all_spec_features(x2,fs,N,alpha,lower_lim,upper_lim,rolloff_pct,num_bands)
	c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	mfcc = c(x2*2**15)#normalize to integer

	
	llr_blk_spec=np.zeros(spec_features.shape[0])
	for i in range(spec_features.shape[0]):
		llt=spec_gmm_turaco.log_likelihood(spec_features[i,:])
		llnt=spec_gmm_noturaco.log_likelihood(spec_features[i,:])
		llr_blk_spec[i]=llt-llnt
	llr_blk_mfcc=np.zeros(mfcc.shape[0])
	for i in range(mfcc.shape[0]):
		llt=mfcc_gmm_turaco.log_likelihood(mfcc[i,:])
		llnt=mfcc_gmm_noturaco.log_likelihood(mfcc[i,:])
		llr_blk_mfcc[i]=llt-llnt

	y_spec=moving_avg(llr_blk_spec,blk_win)
	y_mfcc=moving_avg(llr_blk_mfcc,blk_win)
	
	if np.mean(y_spec)>SpecThreshold:
		SpecRes='Yes'
				
	else:
		SpecRes='No'

	if np.mean(y_mfcc)>MfccThreshold:
		MfccRes='Yes'
				
	else:
		MfccRes='No'
	
	if SpecRes!=MfccRes:
		File_HTcallNeq.append(s1[0])
		file4.write(s1[0])
		file4.write('\t')
		file4.write(SpecRes)
		file4.write('\t')
		file4.write(MfccRes)
		file4.write('\n')

		neq+=1

	if SpecRes==MfccRes and SpecRes=='Yes':
		File_HTcall.append(s1[0])
		#np.sum(np.diff((y_spec>Thresh_DetectFunc)*Thresh_DetectFunc)>0),np.sum(np.diff((y_mfcc>Thresh_DetectFunc)*Thresh_DetectFunc)>0)
		file3.write(s1[0])
		file3.write('\t')
		file3.write(SpecRes)
		file3.write('\t')
		file3.write(MfccRes)
		file3.write('\n')
		htcall_eq+=1

	if SpecRes==MfccRes and SpecRes=='No':
		File_NoHTcall.append(s1[0])

	
	file2.write(s1[0])
	file2.write('\t')
	file2.write(SpecRes)
	file2.write('\t')
	file2.write(MfccRes)
	file2.write('\n')
	
	

	
	num_files+=1

file1.close()
file2.close()
file3.close()
file4.close()

print 'Screening in ',(time.time()-start_time),' seconds'
print 'Screening in ',(time.time()-start_time)/(60.0),' minutes'
print 'Screening in ',(time.time()-start_time)/(60.0*num_files),' minutes per file'


