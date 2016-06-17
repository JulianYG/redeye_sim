import sys, os, csv, argparse
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from lib import utils
from ast import literal_eval as le
from collections import OrderedDict

argv = argparse.ArgumentParser(description='Convert row major csv to column based')
argv.add_argument('--i', type=str, required=True)
argv.add_argument('--o', type=str, required=True)
arg = argv.parse_args()
in_file = './' + arg.i + '.csv'
out_file = './' + arg.o + '.csv'

raw_stat = OrderedDict()
with open(in_file, 'r') as f:
	dic = OrderedDict(csv.reader(f))
	for key, value in dic.items():
		raw_stat[tuple(map(int, key.split('_')))] = le(value)
cols = utils.row2col(raw_stat)

with open(out_file, 'w') as f:
	wtr = csv.writer(f, delimiter='\t')
	wtr.writerow(['SNR','loss3/top-5','loss3/top-1','loss2/top-1',
		'loss1/top-1','loss2/top-5','loss1/top-5'])
	for key, value in cols.items():
		row = [int(key)] + [i[int(key)] for i in value.values()]
		wtr.writerow(row)

