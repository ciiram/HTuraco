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
			spec_features=sf.all_spec_features(x2,fs,N,alpha,lower_lim,upper_lim,rolloff_pct,num_bands)
			X_train= np.vstack([X_train, spec_features]) if X_train.size else spec_features

			train_labels=np.concatenate((train_labels,np.ones(spec_features.shape[0])*file_label_train[i]))


		kmeans_turaco = bob.machine.KMeansMachine(num_clust, spec_features.shape[1])
		kmeans_noturaco = bob.machine.KMeansMachine(num_clust, spec_features.shape[1])
		gmm_turaco = bob.machine.GMMMachine(num_clust, spec_features.shape[1])
		gmm_noturaco = bob.machine.GMMMachine(num_clust, spec_features.shape[1])
		kmeans_trainer = bob.trainer.KMeansTrainer()
		kmeans_trainer.convergence_threshold = 0.0005
		kmeans_trainer.max_iterations = max_iterations
		kmeans_trainer.check_no_duplicate = True

		# Trains using the KMeansTrainer
		print 'Running Kmeans for %d components, Fold %d'%(num_clust,fold)
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

		print 'Training GMMs for %d components, Fold %d'%(num_clust,fold)
		trainer.train(gmm_turaco, X_train[train_labels==1,:])
		trainer.train(gmm_noturaco, X_train[train_labels==0,:])
		


		#evaluate performance on test data
		llr=np.zeros(len(file_label_test))
		for i in range(len(file_label_test)):
			x = scipy.io.wavfile.read('Data/HT/'+File_test[i])
			fs = float(x[0])
			x2 = x[1] / 2.**15
			x2 = x2 / np.max(np.abs(x2))
			spec_features=sf.all_spec_features(x2,fs,N,alpha,lower_lim,upper_lim,rolloff_pct,num_bands)

			ll_turaco=0
			ll_noturaco=0
			for j in range(spec_features.shape[0]):
				ll_turaco+=gmm_turaco.log_likelihood(spec_features[j,:])
				ll_noturaco+=gmm_noturaco.log_likelihood(spec_features[j,:])

	
			llr[i]=(ll_turaco-ll_noturaco)/spec_features.shape[0]

		fpr,tpr,thresh=roc_curve(file_label_test,llr)
		AUC[clust_indx,fold]=auc(fpr,tpr)

		print 'AUC %0.3f'%AUC[clust_indx,fold]
		fold+=1

	clust_indx+=1

MeanAUC=np.mean(AUC,1)
table=[[clust[i],MeanAUC[i]]  for i in range(len(clust))]
headers=["Number of Mixtures","AUC"]
print tabulate(table, headers)

print 'Cross-validation completed in ',(time.time()-start_time)/(60.0),' minutes'

f = open('spec_cv_table.txt', 'w')
f.write(tabulate(table, headers))
f.close()
	
