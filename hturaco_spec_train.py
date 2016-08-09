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

#Get the training data
X_train=np.array([])
train_labels=[]

for i in range(len(Filename)):
	x = scipy.io.wavfile.read('Data/HT/'+Filename[i])
	fs = float(x[0])
	x2 = x[1] / 2.**15
	x2 = x2 / np.max(np.abs(x2))
	spec_features=sf.all_spec_features(x2,fs,N,alpha,lower_lim,upper_lim,rolloff_pct,num_bands)
	X_train= np.vstack([X_train, spec_features]) if X_train.size else spec_features

	train_labels=np.concatenate((train_labels,np.ones(spec_features.shape[0])*Label[i]))




#Train the models

num_clust=64
max_iterations=20
kmeans_turaco = bob.machine.KMeansMachine(num_clust, spec_features.shape[1])
kmeans_noturaco = bob.machine.KMeansMachine(num_clust, spec_features.shape[1])
gmm_turaco = bob.machine.GMMMachine(num_clust, spec_features.shape[1])
gmm_noturaco = bob.machine.GMMMachine(num_clust, spec_features.shape[1])
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
	spec_features=sf.all_spec_features(x2,fs,N,alpha,lower_lim,upper_lim,rolloff_pct,num_bands)

	ll_turaco=0
	ll_noturaco=0
	for j in range(spec_features.shape[0]):
		ll_turaco+=gmm_turaco.log_likelihood(spec_features[j,:])
		ll_noturaco+=gmm_noturaco.log_likelihood(spec_features[j,:])

	
	llr[i]=(ll_turaco-ll_noturaco)/spec_features.shape[0]

print 'Testing GMMs in ',(time.time()-start_time)/(60.0),' minutes'


fpr,tpr,thresh=roc_curve(Label,llr)
AUC=auc(fpr,tpr)
xx=tpr+fpr
print "True Positive Rate",tpr[np.argmin((np.abs(1-xx)))]
print "False Positive Rate",fpr[np.argmin((np.abs(1-xx)))]
print "Threshold",thresh[np.argmin((np.abs(1-xx)))]

#save models
myh5_file = bob.io.HDF5File('Models/SpecTuracoModel.hdf5', 'w')
gmm_turaco.save(myh5_file)
del myh5_file #close
myh5_file = bob.io.HDF5File('Models/SpecNoTuracoModel.hdf5', 'w')
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
plt.savefig('figures/spec_hturaco_roc.png')
plt.show()


