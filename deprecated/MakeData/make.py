import setup
import subprocess
import os
import Tkinter as tk
import tkFileDialog as tkfd 
from dial import Dial

### Important Suggestion ###
# Before doing any of following operations, make sure that #
# a googleTab folder is created and prepared with copies from #
# bvlc_googlenet. #

maker = setup.config()
dataVarMap = {'trainDataPath': 'TRAIN_DATA_ROOT', 'valDataPath': 'VAL_DATA_ROOT',
	'LMDBSavePath': 'EXAMPLE', 'valTrainTxtPath': 'DATA'}
netVarMap = {'tabTrainValProto': 'net', 'trainLMDBFileSave': 'source\tTRAIN',
	 'valLMDBFileSave': 'source\tTEST'}

def prepareData(needUnzip=False, needTrain=False):
	
	file_list = ['createNet', 'makeMean', 'trainNet']
	var_list = ['trainDataPath', 'valDataPath', 'valTrainTxtPath']
	new_var_list = [maker.getLoc(i) for i in var_list]

	for f in file_list:
		modifier = Dial(maker.getLoc(f))
		# treat special cases separately
		modifier.processFile(['TOOLS'], 
			[os.path.join(maker.getLoc('caffeRoot'), 'build/tools')])
		modifier.processFile(['EXAMPLE'], [maker.getLoc('LMDBSavePath')], 
			trainSpec=os.path.basename(maker.getLoc('trainLMDBFileSave')), 
				valSpec=os.path.basename(maker.getLoc('valLMDBFileSave')))
		# now general case
		modifier.processFile([dataVarMap[i] for i in var_list], new_var_list)

	raw_script = os.path.join(os.getcwd(), 'generateRaw.sh')
	write_script(raw_script, needUnzip, needTrain)
	assert (subprocess.call(['./generateRaw.sh']) == 0), 
		"Error executing shell script"

def modifyProto(needTrain=False):

	file_list = ['tabTrainValProto', 'tabSolverProto', 'deployProto']
	var_list = ['noiseTrainedModelPath', 'tabTrainValProto']

	# for f in file_list:
		

	# 	modifier = Dial(maker.getLoc(f))
	# 	# treate special cases separately
	# 	if needTrain: 
	# 		modifier.processFile([])
	# 	modifier.processFile([])

def write_script(name, needUnzip, needTrain):
	"""
	Hardcoded to unzip data files. Make sure their names are the 
	same as "ILSVRC2014_img_train/val/test.tar"
	"""
	with open(name, 'w') as sp:
		sp.write('#!/bin/sh\n\n')
		sp.write('cd ' + maker.getLoc('trainDataPath') + '\n')
		if needUnzip:
			sp.write('for NAME in *.tar; do\n\tmkdir ${NAME%.tar};\n\t')
			sp.write('tar -xvf $NAME -C ${NAME%.tar};\n\trm -r $NAME;\ndone;\n\n')
		sp.write('rm -r ' + maker.getLoc('trainLMDBFileSave') + '\n')
		sp.write('rm -r ' + maker.getLoc('valLMDBFileSave') + '\n')
		sp.write('cd ' + maker.getLoc('valTrainTxtPath') +'\n')
		sp.write('rm imagenet_mean.binaryproto\n\n')
		sp.write(maker.getLoc('createNet') + '\n')
		sp.write(maker.getLoc('makeMean') + '\n')
		sp.write('echo "Done Processing. Now splitting the testing folder..."\n\n')
		sp.write('cd ' + maker.getLoc('testDataPath') + '\n')
		sp.write('i=0;\nfor f in *;\n\tdo d=subset_$(printf %03d $((i/100+1)));\n')
		sp.write('\tmkdir -p $d; mv "$f" $d;\n\tlet i++;\ndone;\n\n')
		sp.write('echo "Finished splitting testing data."\n')
		if needTrain:
			sp.write('echo "Start training..."\n')
			sp.write(maker.getLoc('trainNet') + '\n')
	# make script executable
	subprocess.call(['chmod', '+x', os.path.basename(name)])

#modifyProto()
