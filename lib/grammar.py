import argparse

def _range_format(rf):
	end_pts = rf.split(",")
	if len(end_pts) != 2:
		raise argparse.ArgumentTypeError("Must contain a minimum and a maximum separated by comma")
	elif not (end_pts[0].isdigit() and end_pts[1].isdigit()):
		raise argparse.ArgumentTypeError("Range must be an interger value indicating SNR in dB scale")
	else:
		return rf

def parse_rf(rf):
	vals = rf.split(",")
	return tuple((int(vals[0]), int(vals[1])))
