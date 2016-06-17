import os
import linecache as lc

class Dial():
	"""
	A simple class to parse text file and replace certain strings.
	"""
	def __init__(self, filename, overwrite=1):
		assert (os.path.isfile(filename)), "Error: file does not exist"
		self.changeFrom = filename
		self.fn = os.path.basename(filename).split('.')
		if overwrite:
			self.changeTo = self.changeFrom
		else:
			self.changeTo = os.path.join(os.path.dirname(filename), 
				self.fn[0] + '_customized.' + self.fn[1])

	def processFile(self, *args, **kwds):
		if self.fn[1] == 'sh':
			assert(len(args) >= 2), "Error: not enough input arguments"
			varName, newVal = args
			self.__processShell(self.changeFrom, self.changeTo,
				varName, newVal, **kwds)

		if self.fn[1] == 'prototxt':
			assert(len(args) == 0), "Error: too many input arguments"
			self.__processProto(self.changeFrom, self.changeTo)

	def __processShell(self, fromFile, toFile, varList, newValList, 
		trainSpec='', valSpec=''):
		"""
		Special cases for EXAMPLE. Spec one and two should follow
		the order of train and val.
		"""
		assert (len(varList) == len(newValList)), "Error: Variables cannot be set with unmatched values"
		#print "varList is: " + varList + '\n' + "newVarList is: " + newValList
		with open(fromFile, 'r') as fdFrom:
			lines = fdFrom.readlines()

		for k in range(len(lines)):
			line = lines[k]
			if line[:6] == "RESIZE":
				lines[k] = "RESIZE=true" + '\n'			
			for i in range(len(varList)):
				varName = varList[i]
				newVal = newValList[i]
				varLen = len(varName)
				if line[:12] == '    --solver' and 'solver.prototxt' in newVal:
					line = '    --solver=' + newVal
					continue
				if varName == 'EXAMPLE':
					if "$EXAMPLE" in line:
						beg = line.find('$EXAMPLE')
						if line.find('\\') != -1:
							trainSpec += ' \\'
							valSpec += ' \\'
						if 'train' in line:
							lines[k] = line[:beg] + "$EXAMPLE/" + trainSpec + '\n'
						if 'val' in line:
							lines[k] = line[:beg] + "$EXAMPLE/" + valSpec + '\n'
				if line[:varLen] == varName:
					lines[k] = varName + '=' + newVal + '\n'
		with open(toFile, 'w') as fdTo:
			fdTo.writelines(lines)

	def __processProto(self, fromFile, toFile):
		processor = lm.LayerTuner(fromFile, toFile)
		processor.run()

#k = Dial('/home/julian/Desktop/solver.prototxt')
#k.processFile()
