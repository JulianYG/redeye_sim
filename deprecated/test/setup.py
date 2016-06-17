import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'Tools'))
import dial
import FileChooser as fc

def config():
	save_route = os.path.join(os.path.dirname(os.getcwd()), 'configTab')
	if not os.path.isfile(save_route):
		chooseFile = fc.Record(save_route)
	init = fc.FileChooser()
	init.load_from_file(save_route)
	return init

