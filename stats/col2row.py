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

stat = OrderedDict()
with open(in_file, 'r') as f:
	dic = OrderedDict(csv.reader(f))
	for key, value in dic.items():
		stat[tuple(map(int, key.split('_')))] = le(value)

new_stat = OrderedDict()
for snr, acc_dic in stat.items():
	for types, vals in acc_dic.items():
		if types not in new_stat:
			new_stat[types] = OrderedDict()
		new_stat[types][snr[0]] = vals


with open(out_file, 'w') as f:
	wtr = csv.writer(f, delimiter='\t')
	wtr.writerow(['SNR'] + [i[0] for i in stat.keys()])
	for types, accs in new_stat.items():
		wtr.writerow([types] + accs.values())