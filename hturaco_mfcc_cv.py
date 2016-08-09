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
from tabulate import tabulate



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


clust=[2,4,8,16,32,64,128]
num_fold=5

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

AUC=np.zeros((len(clust),num_fold))
max_iterations=20
clust_indx=0
kf = KFold(len(Filename), n_folds=num_fold,shuffle=True)
start_time=time.time()
for num_clust in clust:
	fold=0
	for train_index, test_index in kf:
		#Train Test split
		File_train=[Filename[i] for i in train_index]
		File_test=[Filename[i] for i in test_index]
		file_label_train=[Label[i] for i in train_index]
		file_label_test =[Label[i] for i in test_index] 

		#Get the training data
		X_train=np.array([])
		train_labels=[]
		for i in range(len(file_label_train)):
			x = scipy.io.wavfile.read('Data/HT/'+File_train[i])
			fs = float(x[0])
			x2 = x[1] / 2.**15
			x2 = x2 / np.max(np.abs(x2))
			c = bob.ap.Ceps(int(fs), win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
			mfcc = c(x2*2**15)#normalize to integer
			X_train= np.vstack([X_train, mfcc]) if X_train.size else mfcc

			train_labels=np.concatenate((train_labels,np.ones(mfcc.shape[0])*file_label_train[i]))


		kmeans_turaco = bob.machine.KMeansMachine(num_clust, mfcc.shape[1])
		kmeans_noturaco = bob.machine.KMeansMachine(num_clust, mfcc.shape[1])
		gmm_turaco = bob.machine.GMMMachine(num_clust, mfcc.shape[1])
		gmm_noturaco = bob.machine.GMMMachine(num_clust, mfcc.shape[1])
		kmeans_trainer = bob.trainer.KMeansTrainer()
		kmeans_trainer.convergence_threshold = 0.0005
		kmeans_trainer.max_iterations = max_iterations
		kmeans_trainer.check_no_duplicate = True

		# Trains using the KMeansTrainer
		print 'Running Kmeans for %d components, Trial %d'%(num_clust,fold)
		kmeans_trainer.train(kmeans_turaco, X_train[train_labels==1,:])
		[variances_turaco, weights_turaco] = kmeans_turaco.get_variances_and_weights_for_each_cluster(X_train[train_labels==1,:])
		means_turaco = kmeans_turaco.means

		kmeans_trainer.train(kmeans_noturaco, X_train[train_labels==0,:])
		[variances_noturaco, weights_noturaco] = kmeans_noturaco.get_variances_and_weights_for_each_cluster(X_train[train_labels==0,:])
		means_noturaco = kmeans_noturaco.means

		

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

		print 'Training GMMs for %d components, Trial %d'%(num_clust,fold)
		trainer.train(gmm_turaco, X_train[train_labels==1,:])
		trainer.train(gmm_noturaco, X_train[train_labels==0,:])
		


		#evaluate performance on test data
		llr=np.zeros(len(file_label_test))
		for i in range(len(file_label_test)):
			x = scipy.io.wavfile.read('Data/HT/'+File_test[i])
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

		fpr,tpr,thresh=roc_curve(file_label_test,llr)
		AUC[clust_indx,fold]=auc(fpr,tpr)

		print num_clust,fold,AUC[clust_indx,fold]
		fold+=1

	clust_indx+=1


MeanAUC=np.mean(AUC,1)
table=[[clust[i],MeanAUC[i]]  for i in range(len(clust))]
headers=["Number of Mixtures","AUC"]
print tabulate(table, headers)

print 'Cross-validation completed in ',(time.time()-start_time)/(60.0),' minutes'

f = open('mfcc_cv_table.txt', 'w')
f.write(tabulate(table, headers))
f.close()
	
