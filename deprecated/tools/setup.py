import FileChooser as fc
import os
import layerModifier as lm

def config():
	save_route = os.path.join(os.path.dirname(os.getcwd()), 'configTab')
	if not os.path.isfile(save_route):
		chooseFile = fc.Record(save_route)
	init = fc.FileChooser()
	init.load_from_file(save_route)
	return init
