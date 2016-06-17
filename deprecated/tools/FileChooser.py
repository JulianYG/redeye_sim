import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import Tkinter as tk
import tkFileDialog as tkfd
import tkMessageBox as tkmb
import os

PATH = 1
FILE = 0

class FileChooser():
	"""
	A simple GUI interface to pick file names and paths
	"""
	
	def __init__(self):
		
		self.__initPathNames__()
		root_window = tk.Tk()
		root_window.withdraw()

	def getFile(self, filetype, isPath):
		if filetype not in self._pathDic.keys():
			print "Error: This type of file is not defined: " + filetype
			return None
		if isPath:
			pn = tkfd.askdirectory(title='Choose your ' + filetype) + '/'
		elif 'Save' in filetype:
			pn = tkfd.asksaveasfilename(title='Choose your ' + filetype)
		else:
			pn = tkfd.askopenfilename(title='Choose your ' + filetype)
		self._pathDic[filetype] = pn
		return pn

	def load_from_file(self, filename):
		with open(filename, 'r') as fp:
			for line in fp:
				items = line.split('\t')
				if items[0] in self._pathDic.keys():
					self._pathDic[items[0]] = items[1].rstrip('\n')

	def export(self, filename):
		with open(filename, 'w') as fp:
			for ftype, path in self._pathDic.items():
				if '/' in path:
					fp.write(ftype + '\t' + path + '\n')

	def listPath(self):
		print self._pathDic

	def getPath(self):
		return self._pathDic.keys()

	def getLoc(self, varname):
		return self._pathDic[varname]

	def __initPathNames__(self):
		self._pathDic = dirs = {}
		dirs['caffeRoot'] = ''
		dirs['caffeExpRoot'] = ''
		dirs['vecSavePath'] = ''

		dirs['pretrainedModel'] = ''
		dirs['noiseTrainedModel'] = ''
		dirs['noiseTrainedModelPath'] = ''
		dirs['fineTuneModelSave'] = ''

		dirs['googleNetSolverProto'] = ''
		dirs['tabSolverProto'] = ''
		dirs['tabTrainValProto'] = ''
		dirs['deployProto'] = ''
		
		dirs['alexTabPath'] = ''
		dirs['googTabPath'] = ''

		dirs['trainDataPath'] = ''
		dirs['valDataPath'] = ''
		dirs['testDataPath'] = ''
		# Make sure val.txt and train.txt are under same directory
		dirs['valTrainTxtPath'] = ''

		dirs['LMDBPath'] = ''
		dirs['trainLMDBFileSave'] = ''
		dirs['valLMDBFileSave'] = ''

		dirs['createNet'] = ''
		dirs['makeMean'] = ''
		dirs['trainNet'] = ''

class Record():
	"""
	Call FileChooser to generate a file that record all necessary paths.
	"""
	def __init__(self, exportRoute, fc=FileChooser()):

		self._route = exportRoute
		self.fc = fc
		self.__extract__()

	def __extract__(self):
		self.path = self.fc.getPath()
		for f in self.path:
			if 'Path' in f or 'Root' in f:
				self.fc.getFile(f, 1)
			else:
				self.fc.getFile(f, 0)
		self.fc.export(self._route)

	def load(filetypeStr):
		# Deprecated
		if filetype not in self.path:
			print "Error: File type not matched"
			return None
		else:
			return self.fc.getLoc(filetypeStr)
