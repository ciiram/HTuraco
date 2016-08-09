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
from matplotlib import gridspec
from sklearn import linear_model


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
gmm_turaco = bob.machine.GMMMachine(myh5_file)
myh5_file = bob.io.HDF5File('Models/MfccNoTuracoModel.hdf5')
gmm_noturaco = bob.machine.GMMMachine(myh5_file)

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

N=512
alpha=.5

#file used in manuscript
x=scipy.io.wavfile.read('Data/Location-4/4-2016-01-05-17-00-01.wav')

fs=float(x[0])
x2=x[1]/2.**15
x2=x2/np.max(np.abs(x2)) 

plt.figure()
Pxx, freqs, tt, plot = plt.specgram(
    x2,
    NFFT=N, 
    Fs=fs, 
    noverlap=int(N * alpha))

plt.xlim([0,len(x2)/fs])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)') 


#compute features
c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
mfcc = c(x2*2**15)#normalize to integer

blk_win=300
porder=16
Thresh_DetectFunc=2.6

llr_blk=np.zeros(mfcc.shape[0])
for i in range(mfcc.shape[0]):
	llt=gmm_turaco.log_likelihood(mfcc[i,:])
	llnt=gmm_noturaco.log_likelihood(mfcc[i,:])
	llr_blk[i]=llt-llnt

y=moving_avg(llr_blk,blk_win)
x=tt[:len(llr_blk)]

CallDetect=(y>Thresh_DetectFunc)*Thresh_DetectFunc
CallDetect=CallDetect.astype('int')
NumCallDetect=sum(np.diff(CallDetect)>0)


fig=plt.figure()
gs = gridspec.GridSpec(3, 1, height_ratios=[2,1, 1])
ax0 = plt.subplot(gs[0]) 

Pxx, freqs, tt, ax0 = plt.specgram(
    x2,
    NFFT=N, 
    Fs=fs, 
    noverlap=int(N * alpha))


plt.xticks([])
plt.ylabel('Frequency (Hz)') 
ax1 = plt.subplot(gs[1]) 
ax1.plot(x,llr_blk)
ax1.plot(x,y,'-r',linewidth=2)
plt.ylabel('LLR')
plt.yticks([])
plt.xticks([])
ax2 = plt.subplot(gs[2]) 
ax2.plot(x,CallDetect,linewidth=2)
plt.ylim([0,3])
plt.yticks([])
plt.xlabel('Time (s)')
plt.ylabel('Call Location')
plt.xlim([0,len(x2)/fs])
plt.savefig('figures/call_detect_1.png',dpi=300)



plt.show()
	


