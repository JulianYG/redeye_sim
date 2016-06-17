import landscape
import random
from runProto import run_deploy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'Tools'))
import modelcheck

rough_terrain = landscape.create_map(g_intvl=3)

well = random.choice(rough_terrain)
iters = 1000

for i in range(iters):
	own_efficiency = well.get_energy_loss / well.get_score
	neighbor_efficiency = []
	source = well.get_neighbors()
	for rock in source:
		neighbor_efficiency.append(rock.get_energy_loss / rock.get_score)
	compare = [i / own_efficiency for i in neighbor_efficiency]
	well = source[compare.index(min(compare))]

print well.get_score(), well.get_energy_loss(), well.get_name()