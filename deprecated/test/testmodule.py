import caffemark as cm
import numpy
from datetime import datetime
import os
import csv

class CaffeTester:
	"""
	A testing module to input different levels of degradations and 
	produce analystical results. Requires parameters for testing, such
	as type of degradations, stages and testing numbers. Produce visual
	results for analysis.
	Input arguments: 
		caffeRoot, iroot, dataRoot: caffe root, test image root, and vector save path
		logName: name of log file to output the test results. Will be stored under 'stat' folder.
		modelName: name of model to use. Include the underscore at the beginning.
		Names are typically defined as '_deg()_qual(gau)_iter_xxx' to indicate details about the model.
		numOfBatch: number of testing folders to use
		numPerBatch: number of testing images per subset folder
		dataSuffix: to indicate whether a pretrained clean model or a model trained by 
		degraded images. 'deg' or 'clean'
	"""
	def __init__(self, caffeRoot, iroot, dataRoot, logName, modelName, 
		numOfBatch, numPerBatch, modelType='googlenet', dataSuffix='', protoSuffix='', debug=False,
		optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):

		self._iroot = iroot
		self._croot = caffeRoot
		self._export = dataRoot
		self._caffeInfo = range(numOfBatch)
		self._logName = logName
		self._nb = numOfBatch
		self._itemNum = numPerBatch
		self._totalNum = numOfBatch * numPerBatch
		self._suff = dataSuffix
		self.debugMode = debug
		if (numOfBatch < 1 | numOfBatch > 20):
			print "Error: cannot perform operation\n"
		self._cup = cm.Caffemark(caffeRoot, 0, numPerBatch, 
			modelType, modelName, protoSuffix, False, debug,optional_trainedfile,optional_modelfile,optional_imagemean)

	def setImages(self, batchNum):
		"""
		Requires to set images before each run.
		"""
		self._cup.reset()
		self._cup.set(os.path.join(self._iroot, 'subset' + '_' + '%03d' % (batchNum + 1)));

	def degrade(self, typ, init, step, inc, topN, nthrun, thresh=0.5, folderName=''):
		self.runNum = nthrun
		self.steps = step
		self.topNum = topN
		self.degType = typ
		self.initVal, self.increment = init, inc
		if folderName == '':
			classvec_data_path = os.path.join(self._export, 'classvec_2012', str(self.degType))
		else:
			classvec_data_path = os.path.join(self._export, 'classvec_2012', str(folderName))
		if not os.path.exists(classvec_data_path):
			os.makedirs(classvec_data_path)
		if self.debugMode:
			print classvec_data_path
		for i in range(self._nb):
			self._caffeInfo[i] = {}
			self._caffeInfo[i]['numOfMatches'] = []
			self._caffeInfo[i]['accuracy'] = []
			self._caffeInfo[i]['degradationLvl'] = []
			self._caffeInfo[i]['ratesInfo'] = []
			for deg_lvl in range(step):
				self._cup.setDegradeParam(typ, [float(init) + inc * deg_lvl])
				predicted = self._cup.runDegradation(topN, thresh)[1]
				filename = os.path.join(classvec_data_path, 
					self._logName + '_' + str(i) + '_' + str(deg_lvl) + '_' + str(self.runNum) + ".npy")
				
				numpy.save(filename, predicted)
				self._cup.clean()
				#self._caffeInfo[i]['numOfMatches'].append(r_res)
				#self._caffeInfo[i]['accuracy'].append(v_res)
				#self._caffeInfo[i]['degradationLvl'].append(deg_lvl)
				#self._caffeInfo[i]['ratesInfo'].append(rates)
		# if self.debugMode:
		# 	print self._caffeInfo
		# l1, l2, l3 = self.__record__()
		# return self.__score_calc__(l1, l2, l3)
#		self._cup.reset()

	def __record__(self):

		stat_data_path = os.path.join(self._croot, 'stat_2013')
		if not os.path.exists(stat_data_path):
			os.makedirs(stat_data_path)

		fd = open(os.path.join(stat_data_path, self._logName + '_' + str(self.runNum) + ".log"), 'w')
		
		top1_path = os.path.join(stat_data_path, 'top1')
		topN_path = os.path.join(stat_data_path, 'top' + str(self.topNum))
		rates_path = os.path.join(stat_data_path, 'rates')
		
		if not os.path.exists(top1_path):
			os.makedirs(top1_path)
		if not os.path.exists(topN_path):
			os.makedirs(topN_path)
		if not os.path.exists(rates_path):
			os.makedirs(rates_path)

		matching_prc = open(os.path.join(top1_path, 'top1_' + self._suff  + '_' + str(self.degType) + '_' + str(self.runNum) + '.csv'), 'w')
		raw_acc = open(os.path.join(topN_path, 'top' + str(self.topNum) + '_' + self._suff + '_' + str(self.degType) + '_' + str(self.runNum) + '.csv'), 'w')
		rates = open(os.path.join(rates_path, 'rates_' + self._suff + '_' + str(self.degType) + '_' + str(self.runNum) + '.csv'), 'w')
		matching_prc.write("deglvl\ttop1_matching_prc\n")
		raw_acc.write("deglvl\ttopN_matching_prc\n")
		rates.write("deglvl\tTP\tTN\tFP\tFN\tpos_denom\tneg_denom\n")
		# write the headers

		prc_entry = csv.writer(matching_prc, delimiter='\t')
		raw_acc_entry = csv.writer(raw_acc, delimiter='\t')
		rate_entry = csv.writer(rates, delimiter='\t')

		fd.write("Log file created: " + str(datetime.now()) + "\n")
		fd.write("Batches used: " + str(len(self._cups)) + " Items per batch: " + str(self._itemNum) + "\n")
		lvl_match = []
		lvl_acc = []
		lvl_rates = []
		for d_lvl in range(self.steps):
			fd.write("Degradation level " + str(d_lvl) + ":\n\t")
			fd.write("Degradation type(s): " + str(self.degType) + "\n\t")
			fd.write("Degradation parameter(s): " + str(self.initVal + d_lvl * self.increment) + "\n\t")
			avg_accruacy, match_prc = 0.0, 0.0
			avg_TP, avg_TN, avg_FP, avg_FN = 0.0, 0.0, 0.0, 0.0
			total_pos_denom, total_neg_denom = 0, 0
			fd.write("Matching percentage for each batch: ")
			for coffee in self._caffeInfo:
				match_prc += coffee['numOfMatches'][d_lvl]
				fd.write('%.4f' % (coffee['numOfMatches'][d_lvl]) + '\t')

			fd.write("\n\tRaw vector accuracy for each batch: ")
			for coffee in self._caffeInfo:
				avg_accruacy += coffee['accuracy'][d_lvl]
				fd.write('%.4f' % (coffee['accuracy'][d_lvl]) + '\t')

			fd.write("\n\tTP, TN, FP, FN rates for each batch: \n\t")
			for coffee in self._caffeInfo:
				avg_TP += coffee['ratesInfo'][d_lvl][0]
				avg_TN += coffee['ratesInfo'][d_lvl][1]
				avg_FP += coffee['ratesInfo'][d_lvl][2]
				avg_FN += coffee['ratesInfo'][d_lvl][3]
				total_pos_denom += coffee['ratesInfo'][d_lvl][4]
				total_neg_denom += coffee['ratesInfo'][d_lvl][5]

				fd.write('%.4f' % (coffee['ratesInfo'][d_lvl][0]) + '\t')
				fd.write('%.4f' % (coffee['ratesInfo'][d_lvl][1]) + '\t')
				fd.write('%.4f' % (coffee['ratesInfo'][d_lvl][2]) + '\t')
				fd.write('%.4f' % (coffee['ratesInfo'][d_lvl][3]) + '\t')
				fd.write(str(coffee['ratesInfo'][d_lvl][4]) + '\t')
				fd.write(str(coffee['ratesInfo'][d_lvl][5]) + '\n\t')

			fd.write('\n')

			match_prc /= self._nb
			avg_accruacy /= self._nb
			avg_TP /= self._nb
			avg_FP /= self._nb
			avg_TN /= self._nb
			avg_FN /= self._nb

			# prc_entry.write(str(d_lvl) + ' ' + str(d_lvl) + '\t' + str(match_prc) + '\n')
			# raw_acc_entry.write(str(d_lvl) + ' ' + str(d_lvl) + '\t' + str(avg_accruacy) + '\n')

			avg_TP = float('%.4f' % (avg_TP))
			avg_FP = float('%.4f' % (avg_FP))
			avg_TN = float('%.4f' % (avg_TN))
			avg_FN = float('%.4f' % (avg_FN))

			lvl_match.append([d_lvl, float('%.4f' % (match_prc))])
			lvl_acc.append([d_lvl, float('%.4f' % (avg_accruacy))])
			lvl_rates.append([d_lvl, avg_TP, avg_TN, avg_FP, avg_FN, total_pos_denom, total_neg_denom])
			# to eliminate excess of digits

		if self.debugMode:
			print lvl_match, lvl_acc, lvl_rates

		prc_entry.writerows(lvl_match)
		raw_acc_entry.writerows(lvl_acc)

		for rate_vec in lvl_rates:
			rate_entry.writerow(rate_vec)

		fd.close()
		matching_prc.close()
		raw_acc.close()
		rates.close()
		return lvl_match, lvl_acc, lvl_rates

	def __score_calc__(self, m_prc, v_acc, v_rate):
		tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
		for r_list in v_rate:
			tp += r_list[1]
			tn += r_list[2]
			fp += r_list[3]
			fn += r_list[4]
		tp /= len(v_rate)
		tn /= len(v_rate)
		fp /= len(v_rate)
		fn /= len(v_rate)
		score = 0.3 * sum(m_prc[1])/self.steps + 0.2 * sum(v_acc[1])/self.steps 
		+ 0.4 * tp + 0.4 * tn - 0.15 * fp - 0.15 * fn
		return score
