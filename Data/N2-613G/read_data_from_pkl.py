import numpy as np
import run_array_low
import inp
import pickle
import math

method = inp.method
molecule = inp.molecule
ansatz = inp.ansatz
n_reps = inp.n_reps
basis = inp.basis
initialization = inp.initialization
batches = inp.batches
iterations = inp.iterations
samples_per_batch = inp.samples_per_batch
prob_2 = inp.prob_2
prob_1 = inp.prob_1
n_shots = inp.n_shots
rbm_frequency_threshold= 1/n_shots
sampler_data_available = inp.sampler_data_available
#n_samples = inp.n_samples
n_samples_scale_factor = inp.n_samples_scale_factor
mbpt_max_rank = inp.mbpt_max_rank
mbpt_data_available = inp.mbpt_data_available
en_conv_thresh = inp.en_conv_thresh
rbm_training_iter = inp.rbm_training_iter
constrained_gibbs_generation = inp.constrained_gibbs_generation
n_gibbs_sampling = inp.n_gibbs_sampling
run = inp.run
ansatz_used = inp.ansatz_used
n = 2

molecule = inp.molecule
basis = inp.basis
bond_dist = inp.bond_dist
# 1. Create the figure and the first y-axis

arr = run_array_low.arr




with open('PIGen_SQD_data_mbpt_rbm_rank'+str(mbpt_max_rank)+'_sqd_PES_' + str(molecule) + '_' + str(basis) + '_shots_' + str(n_shots) + '_' + '.pkl', 'rb') as file:
	data = pickle.load(file)

#with open('mbpt_rbm_6prcnt_sqd_PES_H2O_631g_shots_250000_.pkl', 'rb') as file:
#	data4 = pickle.load(file)

#print (data)
#exit()
casci_energy = data[0]
energy = data[1]
max_diag_dim = data[2]
max_dominant_dim = data[3]
max_blacklisted_dim = data[4]
core_space_dim = data[5]




print ("    eq_dist = 1.0, bond_dist = bond_stretch * eq_dist, coord = N 0.0 0.0 0.0; N 0.0 0.0  + str(1 * bond_dist)")

print ('-----------------------------------------------------------')
print ('Molecule:', molecule)
print ('Equilibrium Bond distance taken (R_eq): 1 Angstrom')
print ('Number of Frozen Cores (Spatial orbitals): 2')
print ('Bond Stretch Factors',arr)
print ('-----------------------------------------------------------')

print ('----------------------* FCI Data*---------------------------------------')
print('FCI en',casci_energy)
print ('core_space_dim',core_space_dim)

print ('----------------------* PIGen-SQD Data *---------------------------------------')
print('PIGen-SQD En',energy)
print (np.abs(np.array(casci_energy)-np.array(energy)))
print ('PIGen_SQD Diagonalization Dimension (Maximum)', max_diag_dim)
print ('PIGen_SQD Dominant Configuration Dimension (Maximum) (|c_i|> 1e-9)', max_dominant_dim)
print ('PIGen_SQD Blacklisted Configuration Dimension (Maximum) (|c_i|< 1e-9)', max_blacklisted_dim)
