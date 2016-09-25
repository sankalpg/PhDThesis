# This script contains all the addition plotting code which is not covered in individual papers
import os, sys
import numpy as np
import json, pickle
import essentia.standard as ess
import matplotlib.pyplot as plt
from scipy.signal import hamming
import copy

import matplotlib as mpl

#mpl.rc('font',family='Times New Roman')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rc('font',family='Times New Roman')



sys.path.append(os.path.join('/home/sankalp/Work/Work_PhD/TA_Courses/sms-tools/software/models'))
import stft as STFT


def plotPitch(audio_file, pitch_file, output_file, time_start, time_end):
	"""
	Example:
	PredominantPitchExample: plt.plotPitch('/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/a99e07d5-20a0-467b-8dcd-aa5a095177fd/Rashid_Khan/Evergreen/Raga_Lalit_783aa4b0-26f3-4e18-844c-b787be6d9849.mp3', '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/a99e07d5-20a0-467b-8dcd-aa5a095177fd/Rashid_Khan/Evergreen/Raga_Lalit_783aa4b0-26f3-4e18-844c-b787be6d9849.pitch', '/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/plotUtils/ch05_preProcessing/predominantMelodyExample.png', 22*60 + 15,  22*60 + 45)
	octaveErrorIllustration1: plt.plotPitch('/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/a99e07d5-20a0-467b-8dcd-aa5a095177fd/Rashid_Khan/Evergreen/Raga_Lalit_783aa4b0-26f3-4e18-844c-b787be6d9849.mp3', '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/a99e07d5-20a0-467b-8dcd-aa5a095177fd/Rashid_Khan/Evergreen/Raga_Lalit_783aa4b0-26f3-4e18-844c-b787be6d9849.pitch', '/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/plotUtils/ch05_preProcessing/octaveErrorIllustration.png', 22*60 + 20,  22*60 + 25)
	octaveErrorIllustration2:plt.plotPitch('/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/46997b02-f09c-4969-8138-4e1861f61967/Kaustuv_Kanti_Ganguli/Raag_Shree/Raag_Shree_928a430e-813e-48b0-8a23-566e74aa8dc9.mp3', '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/46997b02-f09c-4969-8138-4e1861f61967/Kaustuv_Kanti_Ganguli/Raag_Shree/Raag_Shree_928a430e-813e-48b0-8a23-566e74aa8dc9.pitch', '/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/plotUtils/ch05_preProcessing/octaveErrorIllustration.png', 59*60 + 47,  59*60 + 50)
	octaveErrorIllustration3: plt.plotPitch('/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/64e5fb9e-5569-4e80-8e6c-f543af9469c7/Prabha_Atre/Maalkauns/Jaako_Mana_Raam_980b4a00-6e7c-41c1-81ee-6b021d237343.mp3', '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/64e5fb9e-5569-4e80-8e6c-f543af9469c7/Prabha_Atre/Maalkauns/Jaako_Mana_Raam_980b4a00-6e7c-41c1-81ee-6b021d237343.pitch', '/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/plotUtils/ch05_preProcessing/octaveErrorIllustration.png', 25*60 + 9,  25*60 + 11)




	
	"""
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
	#font="Times New Roman"


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
	plt.xlabel("Time (s)", fontsize = fsize)
	plt.ylabel("Frequency (Hz)", fontsize = fsize)


	plt.tight_layout()
	plt.savefig(output_file, dpi = 600)


# def plotPitchSmootheningExample():
# 	pass


def plotPitchJumpCorrectionExample(pitch_file, corrected_pitch_file, output_file, time_start, time_end):

	#Example : plt.plotPitchJumpCorrectionExample('/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/1b05a564-059f-445b-b325-cf26318367e3/Kumar_Gandharv/The_Genius_Of_Pt__Kumar_Gandharv/Raga_Miyan_Malhar_6c573272-1f4a-4504-b7e1-996f877e1e15.pitch', '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/1b05a564-059f-445b-b325-cf26318367e3/Kumar_Gandharv/The_Genius_Of_Pt__Kumar_Gandharv/Raga_Miyan_Malhar_6c573272-1f4a-4504-b7e1-996f877e1e15.pitchSilIntrpPP', '/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/plotUtils/ch05_preProcessing/smootheningExample.png', 15*60 + 35 , 15*60+40)
	
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

	p1, = plt.plot(time, pitch, color = 'b')

	time_pitch = np.loadtxt(corrected_pitch_file)
	ind_start = np.argmin(abs(time_pitch[:,0]-time_start))
	ind_end = np.argmin(abs(time_pitch[:,0]-time_end))
	pitch = copy.deepcopy(time_pitch[ind_start:ind_end, 1])
	time = 0.005 + copy.deepcopy(time_pitch[ind_start:ind_end, 0])-time_pitch[ind_start,0]

	p2, = plt.plot(time, pitch, color = 'r')

	xLim = ax.get_xlim()
	yLim = ax.get_ylim()    
	ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
	plt.autoscale(tight=True)

	plt.legend([p1, p2], ['Original Pitch', 'Post-Processed Pitch'])
	plt.xlabel("Time (s)", fontsize = fsize, fontname=font)
	plt.ylabel("Frequency (Hz)", fontsize = fsize, fontname=font)


	plt.tight_layout()
	plt.savefig(output_file, dpi = 600)



def plotPairs(pattern1, pattern2, plotName=-1):
    
    
    colors = ['g', 'r']
#   linewidths = [3,0.1 ,0.1 , 3]

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pLeg = []
    
    p, = plt.plot((196/44100.0)*np.arange(pattern1.size), pattern1, 'r', linewidth=2, markersize=4.5)
    pLeg.append(p)
    
    p, = plt.plot((196/44100.0)*np.arange(pattern2.size), pattern2, 'k', linewidth=2, markersize=4.5)
    pLeg.append(p)

    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("time (s)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency (Hz)", fontsize = fsize, fontname=font, labelpad=fsize2)

    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1 



def plotMotifPairs(motifFile):

	#example: plt.plotMotifPairs('/home/sankalp/Work/Work_PhD/publications/2016_PhDThesis/MISC/TaniPattern/data/Ninni_Vinaga.disPatt_2s_config0')

    pitchExt = '.pitch'
    
    motifs = np.loadtxt(motifFile)
    
    fname, ext = os.path.splitext(motifFile)
    pitchFile = fname + pitchExt
    
    #load pitch data
    print("Loading pitch file, it might take a while, please be patient!!!")
    pitchData = np.loadtxt(pitchFile)
    hop = pitchData[1,0]-pitchData[0,0]
    
    for ii, motif in enumerate(motifs):
        str1 = motif[0]
        end1 = motif[1]
        str2 = motif[2]
        end2 = motif[3]
        dist = motif[4]
        
        start_ind1 = np.argmin(abs(pitchData[:,0]-str1))
        end_ind1 = np.argmin(abs(pitchData[:,0]-end1))
        start_ind2 = np.argmin(abs(pitchData[:,0]-str2))
        end_ind2 = np.argmin(abs(pitchData[:,0]-end2))
        
        plotPairs(pitchData[start_ind1:end_ind1,1],  pitchData[start_ind2:end_ind2,1])


def plotGamakaExample(pitchFile, plotName=-1, tonic = 146.8325, time_range= []):

	#Hindustani
	#Example 1: plt.plotGamakaExample('GamakaExamples/hindustani/Bhimpalasi_S-m-g_Stereo.pitch', plotName='GamakaExamples/hindustani/Bhimpalasi_S-m-g_Stereo.pdf',  tonic = 146.8325)
	#Example 2: plt.plotGamakaExample('GamakaExamples/hindustani/Bageshri_S-m-g_Stereo.pitch', plotName='GamakaExamples/hindustani/Bageshri_S-m-g_Stereo.pdf',  tonic = 146.8325)
	#Example 3: plt.plotGamakaExample('GamakaExamples/hindustani/DemoAlankar.pitch', plotName='GamakaExamples/hindustani/DemoAlankar.pdf',  tonic = 146.8325)
	#Example 4: plt.plotGamakaExample('GamakaExamples/hindustani/DemoAlankar.pitch', plotName='GamakaExamples/hindustani/DemoAlankar_2.7__11.6.pdf',  tonic = 146.8325, time_range=[2.7, 11.6])
	#Example 5: plt.plotGamakaExample('GamakaExamples/hindustani/DemoAlankar.pitch', plotName='GamakaExamples/hindustani/DemoAlankar_13.5__18.pdf',  tonic = 146.8325, time_range=[13.5, 18])

	#Carnatic1: 
	#Example 1: plt.plotGamakaExample('GamakaExamples/carnatic/Sphuritam_on_M1_Raga_Arabhi.pitch', plotName='GamakaExamples/carnatic/Sphuritam_on_M1_Raga_Arabhi.pdf', tonic = 146.8325)
	#Example 2: plt.plotGamakaExample('GamakaExamples/carnatic/Kampitam_on_N2_Todi.pitch', plotName='GamakaExamples/carnatic/Kampitam_on_N2_Todi.pdf', tonic = 146.8325)
	#Example 3: plt.plotGamakaExample('GamakaExamples/carnatic/Kampitam_on_N2_Rag_Ahiri.pitch', plotName='GamakaExamples/carnatic/Kampitam_on_N2_Rag_Ahiri.pdf', tonic = 146.8325)
	#Example 4: plt.plotGamakaExample('GamakaExamples/carnatic/Odukkal_on_D1.pitch', plotName='GamakaExamples/carnatic/Odukkal_on_D1.pdf', tonic = 146.8325)

	pitchData = np.loadtxt(pitchFile)
	ind_non_zero = np.where(pitchData[:,1]>60)[0]
	ind_zero = np.where(pitchData[:,1]<=60)[0]
	pitchData[ind_non_zero,1] = 1200*np.log2(pitchData[ind_non_zero,1]/tonic)
	pitchData[ind_zero,1] = -10000
	hop = pitchData[1,0]-pitchData[0,0]

	if len(time_range) == 2:
		start_ind1 = np.argmin(abs(pitchData[:,0]-time_range[0]))	
		end_ind1 = np.argmin(abs(pitchData[:,0]-time_range[1]))
	else:
		start_ind1 = 0
		end_ind1 = pitchData.shape[0]

	ind_non_zero = np.where(pitchData[start_ind1:end_ind1,1]>-10000)[0]
	max_pitch_range = np.max(pitchData[start_ind1:end_ind1,1][ind_non_zero])+100
	min_pitch_range = np.min(pitchData[start_ind1:end_ind1,1][ind_non_zero])-100

	fsize = 18
	fsize2 = 16

	fig = plt.figure() 
	ax = fig.add_subplot(111)    

	plt.plot(hop*np.arange(pitchData.shape[0])[:end_ind1-start_ind1], pitchData[start_ind1:end_ind1,1], 'k', linewidth=1)

	plt.ylim([min_pitch_range, max_pitch_range])
	plt.xlabel("Time (s)", fontsize = fsize)
	plt.ylabel("Frequency (Cents)", fontsize = fsize, labelpad=fsize2)

	xLim = ax.get_xlim()
	yLim = ax.get_ylim()
	print yLim

	ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
	plt.tick_params(axis='both', which='major', labelsize=fsize2)

	if isinstance(plotName, int):
		plt.show()				
	elif isinstance(plotName, str):
		fig.savefig(plotName, bbox_inches='tight')

	return 1 		