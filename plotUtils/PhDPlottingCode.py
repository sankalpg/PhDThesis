# This script contains all the addition plotting code which is not covered in individual papers
import os, sys
import numpy as np
import json, pickle
import essentia.standard as ess
import matplotlib.pyplot as plt
from scipy.signal import hamming
import copy



sys.path.append(os.path.join('/home/sankalp/Work/Work_PhD/TA_Courses/sms-tools/software/models'))
import stft as STFT


def plotPredominantPitchExample(audio_file, pitch_file, output_file, time_start, time_end):

	frameSize = 4096
	hopSize = 512
	NFFT = 4096
	w = np.hamming(frameSize)

	audio = ess.MonoLoader(filename = audio_file)()
	sampleRate = float(ess.MetadataReader(filename=audio_file)()[10])

	time_pitch = np.loadtxt(pitch_file)

	sample_start = sampleRate*time_start
	sample_end = sampleRate*time_end

	audio = audio[sample_start: sample_end]

	ind_start = np.argmin(abs(time_pitch[:,0]-time_start))
	ind_end = np.argmin(abs(time_pitch[:,0]-time_end))

	pitch = copy.deepcopy(time_pitch[ind_start:ind_end, 1])
	time = copy.deepcopy(time_pitch[ind_start:ind_end, 0])-time_pitch[ind_start,0]

	mX, pX = STFT.stftAnal(audio, w, NFFT, hopSize)

	fig = plt.figure() 
	ax = fig.add_subplot(111)
	plt.hold(True)
	fsize = 14
	fsize2 = 14
	font="Times New Roman"


	maxplotfreq = 1000.0
	numFrames = int(mX[:,0].size)
	frmTime = hopSize*np.arange(numFrames)/float(sampleRate)                             
	binFreq = sampleRate*np.arange(NFFT*maxplotfreq/sampleRate)/NFFT                       
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:NFFT*maxplotfreq/sampleRate+1]))
	plt.hold(True)
	p, = plt.plot(time, pitch, color = 'k')


	xLim = ax.get_xlim()
	yLim = ax.get_ylim()    
	ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
	plt.autoscale(tight=True)

	plt.legend([p], ['Predominant pitch'])
	plt.xlabel("Time (s)", fontsize = fsize, fontname=font)
	plt.ylabel("Frequency (Hz)", fontsize = fsize, fontname=font)


	plt.tight_layout()
	plt.savefig(output_file, dpi = 600)


def plotPitchSmootheningExample():
	pass


def plotPitchJumpCorrectionExample(pitch_file, corrected_pitch_file, time_start, time_end):
	
	time_pitch = np.loadtxt(pitch_file)
	ind_start = np.argmin(abs(time_pitch[:,0]-time_start))
	ind_end = np.argmin(abs(time_pitch[:,0]-time_end))
	pitch = copy.deepcopy(time_pitch[ind_start:ind_end, 1])
	time = copy.deepcopy(time_pitch[ind_start:ind_end, 0])-time_pitch[ind_start,0]

	fig = plt.figure() 
	ax = fig.add_subplot(111)
	plt.hold(True)
	fsize = 14
	fsize2 = 14
	font="Times New Roman"

	p1, = plt.plot(time, pitch, color = 'k')

	time_pitch = np.loadtxt(corrected_pitch_file)
	ind_start = np.argmin(abs(time_pitch[:,0]-time_start))
	ind_end = np.argmin(abs(time_pitch[:,0]-time_end))
	pitch = copy.deepcopy(time_pitch[ind_start:ind_end, 1])
	time = copy.deepcopy(time_pitch[ind_start:ind_end, 0])-time_pitch[ind_start,0]

	p2, = plt.plot(time, pitch, color = 'k')

	xLim = ax.get_xlim()
	yLim = ax.get_ylim()    
	ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
	plt.autoscale(tight=True)

	plt.legend([p1, p2], ['Original Pitch', 'Corrected Pitch'])
	plt.xlabel("Time (s)", fontsize = fsize, fontname=font)
	plt.ylabel("Frequency (Hz)", fontsize = fsize, fontname=font)


	plt.tight_layout()
	plt.savefig(output_file, dpi = 600)










