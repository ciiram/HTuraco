import numpy as np
import scipy as sp
from scipy import signal
import scipy.io.wavfile
import time
import sys
import os
import bob
import spectral_features as sf
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn import mixture
from matplotlib import gridspec
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import pairwise_distances,accuracy_score
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import roc_curve,auc



np.random.seed(123)
file1=open('Data/AllFilesAnnotation.csv',"r")#  Data annotation file 

#read header 
file1.readline()
file1.readline()
file1.readline()


Filename=[]
Label=[]
while file1:
	
	line1=file1.readline()
	s1=line1.split(',')
	
	if len(line1)==0:
		break

	Filename.append(s1[0])
	Label.append(int(s1[1]))

file1.close()




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

#Get the training data
X_train=np.array([])
train_labels=[]

for i in range(len(Filename)):
	x = scipy.io.wavfile.read('Data/HT/'+Filename[i])
	fs = float(x[0])
	x2 = x[1] / 2.**15
	x2 = x2 / np.max(np.abs(x2))
	c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	mfcc = c(x2*2**15)#normalize to integer
	X_train= np.vstack([X_train, mfcc]) if X_train.size else mfcc

	train_labels=np.concatenate((train_labels,np.ones(mfcc.shape[0])*Label[i]))




#Train the models

num_clust=32
max_iterations=20
kmeans_turaco = bob.machine.KMeansMachine(num_clust, mfcc.shape[1])
kmeans_noturaco = bob.machine.KMeansMachine(num_clust, mfcc.shape[1])
gmm_turaco = bob.machine.GMMMachine(num_clust, mfcc.shape[1])
gmm_noturaco = bob.machine.GMMMachine(num_clust, mfcc.shape[1])
kmeans_trainer = bob.trainer.KMeansTrainer()
kmeans_trainer.convergence_threshold = 0.0005
kmeans_trainer.max_iterations = max_iterations
kmeans_trainer.check_no_duplicate = True

# Trains using the KMeansTrainer
print 'Running Kmeans...'
start_time=time.time()
kmeans_trainer.train(kmeans_turaco, X_train[train_labels==1,:])
[variances_turaco, weights_turaco] = kmeans_turaco.get_variances_and_weights_for_each_cluster(X_train[train_labels==1,:])
means_turaco = kmeans_turaco.means

kmeans_trainer.train(kmeans_noturaco, X_train[train_labels==0,:])
[variances_noturaco, weights_noturaco] = kmeans_noturaco.get_variances_and_weights_for_each_cluster(X_train[train_labels==0,:])
means_noturaco = kmeans_noturaco.means

print 'Run Kmeans in ',(time.time()-start_time)/(60.0),' minutes'

#train the gmm
# Initializes the GMM
gmm_turaco.means = means_turaco
gmm_turaco.variances = variances_turaco
gmm_turaco.weights = weights_turaco
gmm_turaco.set_variance_thresholds(0.0005)

gmm_noturaco.means = means_noturaco
gmm_noturaco.variances = variances_noturaco
gmm_noturaco.weights = weights_noturaco
gmm_noturaco.set_variance_thresholds(0.0005)

trainer = bob.trainer.ML_GMMTrainer(True, True, True)
trainer.convergence_threshold = 0.0005
trainer.max_iterations = 25

print 'Training GMMs...'
start_time=time.time()
trainer.train(gmm_turaco, X_train[train_labels==1,:])
trainer.train(gmm_noturaco, X_train[train_labels==0,:])
print 'Trained GMMs in ',(time.time()-start_time)/(60.0),' minutes'


#evaluate performance on training data
start_time=time.time()
llr=np.zeros(len(Filename))
for i in range(len(Filename)):
	x = scipy.io.wavfile.read('Data/HT/'+Filename[i])
	fs = float(x[0])
	x2 = x[1] / 2.**15
	x2 = x2 / np.max(np.abs(x2))
	c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
	mfcc = c(x2*2**15)#normalize to integer
	

	ll_turaco=0
	ll_noturaco=0
	for j in range(mfcc.shape[0]):
		ll_turaco+=gmm_turaco.log_likelihood(mfcc[j,:])
		ll_noturaco+=gmm_noturaco.log_likelihood(mfcc[j,:])

	
	llr[i]=(ll_turaco-ll_noturaco)/mfcc.shape[0]

print 'Testing GMMs in ',(time.time()-start_time)/(60.0),' minutes'


fpr,tpr,thresh=roc_curve(Label,llr)
AUC=auc(fpr,tpr)
xx=tpr+fpr
print "True Positive Rate",tpr[np.argmin((np.abs(1-xx)))]
print "False Positive Rate",fpr[np.argmin((np.abs(1-xx)))]
print "Threshold",thresh[np.argmin((np.abs(1-xx)))]


#save models
myh5_file = bob.io.HDF5File('Models/MfccTuracoModel.hdf5', 'w')
gmm_turaco.save(myh5_file)
del myh5_file #close
myh5_file = bob.io.HDF5File('Models/MfccNoTuracoModel.hdf5', 'w')
gmm_noturaco.save(myh5_file)
del myh5_file #close


plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % AUC,linewidth=2)
plt.plot([0, 1], [0, 1], 'k--',linewidth=2)
plt.plot(fpr[np.argmin((np.abs(1-xx)))],tpr[np.argmin((np.abs(1-xx)))],'bo',markersize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=14)
plt.legend(loc="lower right")
plt.savefig('figures/mfcc_hturaco_roc.png')
plt.show()


