import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile
import time
import sys
import os
import datetime

def spec(x,N,alpha=0):

	'''
	This function computes the magnitude spectrum of the file
	The inputs are:
		x: The sampled audio
		N: Block length in samples
		alpha: overlap, default is 0 (no overlap)
	The function returns the magnitude spectrum
	'''
	Na=int((1-alpha)*N)
	num_block=int((float(len(x))/N-1)/(1-alpha))
	spec=np.zeros((N/2+1,num_block), dtype=complex)
	

	for i in range(num_block):
		spec[:,i]=np.fft.fft(x[i*Na:i*Na+N]*np.hanning(N))[0:N/2+1]

	Sabs=np.abs(spec)
	
	return Sabs

def centroid(Sxx,N,fs,fmin,fmax):
	
	num_frames=Sxx.shape[1]
	centroids=np.zeros(num_frames)
	band_width=np.zeros(num_frames)
	freqs=np.arange(Sxx.shape[0])*fs/float(N)
	Sxx=Sxx[(freqs>fmin)&(freqs<fmax),:]
	freqs=freqs[(freqs>fmin)&(freqs<fmax)]
	Sum=np.sum(Sxx,0)
	centroids=np.sum(freqs[:,None]*Sxx,0)/Sum
	F=freqs[:,None]*np.ones(Sxx.shape)
	band_width=np.sum(np.abs(F-centroids)*Sxx,0)/Sum
	
	return centroids,band_width


def spectral_rolloff(Sxx,N,fs,percent):
	num_frames=Sxx.shape[1]
	rolloff=np.zeros(num_frames)
	S=Sxx/np.sum(Sxx,axis=0)
	freqs=np.arange(Sxx.shape[0])*fs/float(N)
	#compute cumulative sum
	CummulativeSum=np.cumsum(S,0)
	freq_indx=np.sum(CummulativeSum<percent,0)+1
	for i in range(num_frames):
		rolloff[i]=freqs[freq_indx[i]]

	return rolloff

def band_energy_ratio(Sxx,N,fs,num_bands):
	num_frames=Sxx.shape[1]
	BER=np.zeros((num_bands,num_frames))
	band_lim=.5**np.arange(num_bands)*N*.5
	band_lim=np.concatenate((np.array([0]),band_lim[::-1]))
	band_lim=band_lim.astype(int)
	S2=Sxx**2
	E=sum(S2,0)

	for i in range(num_bands):
		BER[i,:]=np.sum(S2[band_lim[i]:band_lim[i+1],:],0)/E


	return BER


def spectral_flux(Sxx):
	num_frames=Sxx.shape[1]
	S=Sxx/np.sum(Sxx,axis=0)
	spectral_flux=np.sum(np.diff(S,1,1)**2,0)
	
	

	return spectral_flux

def spectral_flux_bands(Sxx,N,num_bands):
	num_frames=Sxx.shape[1]
	spectral_flux=np.zeros((num_bands,num_frames-1))
	band_lim=.5**np.arange(num_bands)*N*.5
	band_lim=np.concatenate((np.array([0]),band_lim[::-1]))
	band_lim=band_lim.astype(int)
	S=Sxx/np.sum(Sxx,axis=0)
	S2=np.diff(S,1,1)**2
	spectral_flux=np.zeros((num_bands,num_frames-1))
	for i in range(num_bands):
		spectral_flux[i,:]=np.sum(S2[band_lim[i]:band_lim[i+1],:],0)
			

	return spectral_flux

def all_spec_features(x,fs,N,alpha,fmin,fmax,rolloff_pct,num_bands):

	'''
	This function computes all the spectral features of a file 

	'''
	file_feature=[]
	#Sxx=spec(x,N,alpha)
	Sxx=sp.signal.spectrogram(x,fs,np.hanning(N),N,int(N*alpha),detrend=None,scaling='spectrum',mode='magnitude')[2]
	centroids,band_width=centroid(Sxx,N,fs,fmin,fmax)
	file_feature.append(centroids[1:])
	file_feature.append(band_width[1:])
	rolloff=spectral_rolloff(Sxx,N,fs,rolloff_pct)
	file_feature.append(rolloff[1:])
	BER=band_energy_ratio(Sxx,N,fs,num_bands)
	for i in range(BER.shape[0]):
		file_feature.append(BER[i,1:])
	spec_flux=spectral_flux_bands(Sxx,N,num_bands)
	for i in range(spec_flux.shape[0]):
		file_feature.append(spec_flux[i,:])
	

	return np.array(file_feature).T
	
		
	
