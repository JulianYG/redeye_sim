import scipy.optimize
from runTest import new_run
from point import protoPoint

hyperplane = bp(numOfBatch=10, gintvl=10, qintvl=20)
print hyperplane
# =======
# meta_file = '/home/bigbox/caffe_exp/op.txt'
# with open(meta_file, 'w') as o:
# 	o.write('params\ttop1\ttop5\tenergy\talpha\n')

def mix_performance(slist, q, b=5):
	p = protoPoint(slist, q, b)
	result = p.run_deploy()
	acc = result[0] * 0.4 + result[0] * 0.6
	energy = p.get_energy_loss()
	with open(meta_file, 'a') as opt:
		opt.write(str(slist + [q]) + '\n')
		opt.write(str(p.get_param()) + '\t' + str(result[0]) + '\t' + str(result[1]) + '\t' + str(energy) + '\t' + str(energy / acc) + '\n')
	return energy / acc

loss_acc_func = lambda x: mix_performance([x[0], x[1], x[2]], x[3])
xopt = scipy.optimize.fmin_slsqp(func=loss_acc_func,
	x0 = [2e-4, 2e-4, 2e-4, 0.0156], bounds=[(2e-4, 0.0015), 
		(2e-4, 0.0015), (2e-4, 0.0015), (0.0156, 0.5)])

with open('~/caffe_exp/final_answer.txt', 'w') as f:
	f.write(str(xopt))

print xopt
