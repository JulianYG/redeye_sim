import math
import numpy as np

def adc(Ox, Oy, Oz, c0, bits, vref=1):
	"""
	Ox output x dimension
	Oy output y dimension
	Oz number of kernels
	"""
	dac_e0 = c0 * vref**2 * math.pow(2, bits) * 2
	cmp_e0 = 4.97e-13 * bits
	e0c = 200	# safety margin
	e0 = dac_e0 * e0c + cmp_e0
	return e0 * Ox * Oy * Oz

def snr(c):

	Vspp = 2 # peak-to-peak signal amplitude (-1, 1)
	k = 1.38e-23
	T = 353

	snr = 10 * math.log10((Vspp**2 / 8) / (6 * k * T / c))
	enob = (snr - 1.76) / 6.02

	return (enob, snr)

def convol(Kx, Ky, Kz, Ox, Oy, Oz):
	"""
	Kx kernel x dimension
	Ky kernel y dimension
	Kz kernel z dimension
	"""
	# Parameter setup
	mac_inputs = 8 # MUST be consistent with: mac.m
	mem_c = 10e-15
	amp1_cinch = 80e-15
	amp1_cload = 10e-15 # equals to amp2_cinch
	amp2_cinch = amp1_cload
	amp2_cload = mem_c
	amp_cdamp = 0
	buf_cdamp = 0

	(amp1_t, amp1_e, smpl_t, smpl_e) = mac(amp1_cinch, amp1_cload, mem_c, amp_cdamp, buf_cdamp)
	(amp2_t, amp2_e, _, _) = mac(amp2_cinch, amp2_cload, mem_c, amp_cdamp, buf_cdamp)

	Tamp = (amp1_t * math.ceil(Ky * Kz / mac_inputs) + amp2_t * math.ceil(Kx / mac_inputs)) * Oy * Oz
	Tsmpl = smpl_t * math.ceil(Ky * Kz / mac_inputs) * Oy * Oz
	conv_t = Tamp + Tsmpl

	# Keep the following lines. This is a more maintainable method to find conv_t
	# mac_t = amp_t + smpl_t
	# colvec_t = mac_t * ceil(Ky * Kz / mac_inputs)
	# time to do each mac op * number of mac ops required to do Ky*Kz additions
	# window_t = colvec_t + amp_t * ceil(Kx / mac_inputs)
	# time to generate column average + time to do the addition for Kx samples. DOES NOT require time to sample the Kx colvecs
	# conv_t = window_t * Ox * Oy * Oz
	# when the window shifts, all colvecs become obsolete, so conv must start over
	# check = conv_t - Tamp - Tsmpl

	Eamp = (amp1_e * (ceil(Ky * Kz / mac_inputs) * Kx) + amp2_e * ceil(Kx / mac_inputs)) * Ox * Oy * Oz
	Esmpl = smpl_e * Ky * Kz * Kx * Ox * Oy * Oz
	conv_e = Eamp + Esmpl
	Ops_smpl = Ky * Kz * Kx * Ox * Oy * Oz

	# Keep the following lines. This is a more maintainable method to find conv_e
	# colvec_e = smpl_e * Ky * Kz + amp_e * ceil(Ky * Kz / mac_inputs)
	# energy to sample the inputs for colvec + energy to do mac
	# window_e = colvec_e * Kx + amp_e * ceil(Kx / mac_inputs)
	# energy to generate Kx colvec results + energy to do mac; energy to sample them IS NOT required
	# conv_e = window_e * Ox * Oy * Oz
	# window_e multiplied by total number of outputs
	# check = conv_e - Eamp - Esmpl #ok<NOPRT>

	return (conv_t, Tamp, Tsmpl, conv_e, Eamp, Esmpl, Ops_smpl)

def mac(amp_cinch=80e-15, amp_cload=10e-15, mem_c=10e-15, amp_cdamp=0, buf_cdamp=0):

	# Parameters from circuit simulation 
	# High level parameters
	mac_inputs = 8	# MUST be consistent with: conv.m

	amp_ntau = 10
	amp_tc =   [2, # compensate slew time
	            1, # account for opamp wakeup time
	            2] # just to be safe: compensate parasitics, corners etc
	amp_pc =    1.2 # account for kernel addressing and propagation overhead

	smpl_c = 80e-15 # F
	smpl_r = 1e3 # Ohm

	buf_ntau = 10
	buf_tc =   [1, # compensate slew time
	            1, # account for buffer wakeup time
	            2] # just to be safe
	buf_pc =    1.2 # account for data addressing overhead

	dist_ntau = 10
	dist_tcz = 4 # just to be safe: should be safer than amp because dist works slower than theoretical 1st-order settling

	# Low level parameters
	opamp1_gbw = 2 * pi * 80e6 # rad
	opamp1_load = 1e-12 # F
	opamp1_2id = 20e-6 / (2 * pi *30e6) * opamp1_gbw # A
	opamp1_vdd = 2.5 # V
	opamp1_pc = 1 + 0.2 + 0.2 # account for bias and CMFB
	opamp1_power = opamp1_2id * opamp1_vdd * np.prod(opamp1_pc)
	opamp1_tau = 1 / opamp1_gbw # sec

	opamp2_gbw = 2 * pi * 30e6 # rad
	opamp2_load = 1e-12 # F
	opamp2_2id = 20e-6 / (2*pi*30e6) * opamp2_gbw #A
	opamp2_vdd = 2.5 # V
	opamp2_pc = 1 + 0.1 # account for bias and CMFB
	opamp2_power = opamp2_2id * opamp2_vdd * np.prod(opamp2_pc)
	opamp2_tau = 1 / opamp2_gbw # sec

	amp_load = amp_cinch * mac_inputs / 2 + amp_cload + amp_cdamp 
	amp_tau = opamp1_tau / opamp1_load * amp_load
	amp_power = opamp1_power * np.prod(amp_pc) # houyh: to self -- DO NOT scale using amp_load
	amp_t = amp_tau * amp_ntau * np.prod(amp_tc)
	amp_e = amp_power * amp_t

	# Keep the following line. This is an equivalent method to find amp_e
	#amp_e_check = amp_load * opamp1_vdd / opamp1_gmover2id * amp_ntau * prod(amp_tc)

	buf_load = mem_c / 2 + amp_cinch + buf_cdamp
	buf_tau = opamp2_tau / opamp2_load * buf_load
	buf_power = opamp2_power * np.prod(buf_pc)
	buf_t = buf_tau * buf_ntau * np.prod(buf_tc)
	buf_e = buf_power * buf_t

	dist_tau = smpl_r * smpl_c #sec
	dist_t = dist_tau * dist_ntau * dist_tcz

	smpl_t = buf_t + dist_t
	smpl_e = buf_e

	return (amp_t, amp_e, smpl_t, smpl_e)
