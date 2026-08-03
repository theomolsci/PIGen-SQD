import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
#from qiskit_algorithms import NumPyMinimumEigensolver
#from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.properties.particle_number import ParticleNumber
from qiskit_nature.second_q.properties.angular_momentum import AngularMomentum
from qiskit_aer.primitives.sampler import Sampler
#from qiskit.primitives import Estimator
import qiskit as qt
import numpy as np
from qiskit.circuit.library import EvolvedOperatorAnsatz
from joblib import Parallel, delayed
#from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
#from qiskit_algorithms.optimizers import CG, SPSA, L_BFGS_B, COBYLA
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
# from qiskit.providers.fake_provider import FakeMelbourne
from qiskit_aer.noise import NoiseModel
#from qiskit_aer.primitives import Estimator as AerEstimator
import pickle
from joblib import Parallel, delayed
from functools import partial
from scipy.optimize import minimize
import inp
# from mitiq import zne
import qiskit_aer
#from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import solve_qubit
from qiskit_addon_sqd.qubit import sort_and_remove_duplicates
import time
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import numpy as np
import time
from numba import njit, prange, int64, uint64, float64
from qiskit.quantum_info import SparsePauliOp

#import optimized_mbpt2
# ---------------------------------------------- Set all the parameters ------------------------------------------------------



from numba import njit, prange, threading_layer
import numpy as np

# 1. Define a dummy parallel function to force initialization
@njit(parallel=True)
def force_init_threading():
    x = 0
    for i in prange(10):
        x += i
    return x

# 2. Run it once
force_init_threading()

# 3. NOW you can safely check the layer
try:
    print(f"--- [SYSTEM] Numba Threading Layer: {threading_layer()} ---")
except ValueError:
    print("--- [SYSTEM] WARNING: Threading layer failed to init. Parallelism DISABLED. ---")







seed_val = 42
#seed_val = 5
#seed_val = 10
#seed_val = 200
#This file does not work with rev convention in the recursive RBM generation stage because back and forth is taking too much time
np.random.seed(seed_val)
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
projection_batch_size = inp.projection_batch_size
parallelized_projection = inp.parallelized_projection
n_cores_for_parallel_projection = inp.n_cores_for_parallel_projection
jw_mapped_ham_available = inp.jw_mapped_ham_available
matrix_diagonalization_type = inp.matrix_diagonalization_type
bond_dist = inp.bond_dist

print('-------------- Hyper parameters for RBM ------------------')
print ('rbm_training_iter', rbm_training_iter)
print('Number of shots', n_shots)
print (molecule)
print (basis)
print (bond_dist)
print('******************************************* Running from final code folder ***********************************************************************')




if mbpt_data_available == 'no':
    print('optimized_mbpt2 running')
    import optimized_mbpt2

# conv = 1e-5
frozen_core = 'no'
#noiseless_estimator = Estimator()
noise = 'no'
mitigation = 'no'
c_not_eff = 'no'

noise = 'yes'
# mitigation = 'yes'
# c_not_eff = 'yes'

#if noise == 'yes':
#    run = int(input())
run = 100
#bond_stretch = float(input())
# bond_stretch = 2.0
#stretch = str(bond_stretch) + 'AA'

# ------------------------------------------------
# if molecule == 'H4':
#     eq_dist = 1.0
#     stretch = str(bond_stretch) + 'AA'
#     print(stretch)
#     bond_dist = bond_stretch * eq_dist

# ---------------------------------------------------------------------------------------------------------------------------

if noise == 'yes':
    ######################### Building up the noise ##########################
    # Error probabilities
    prob_1 = prob_1  # 1-qubit gate
    prob_2 = prob_2  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = qiskit_aer.noise.depolarizing_error(prob_1, 1)
    error_2 = qiskit_aer.noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = qiskit_aer.noise.NoiseModel()
    # noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'ry', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Get basis gates from noise model
    basis_gates_noise_model = noise_model.basis_gates
    print(noise_model)

# ****************************************************** Set up molecular informations -----------------------------------------------------------

bond_dist = inp.bond_dist
#bond_dist = float(input())
if molecule == 'H4':
    driver = PySCFDriver(atom="H 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist) + "; H 0.0 0.0 " + str(
        2 * bond_dist) + "; H 0.0 0.0 " + str(3 * bond_dist), basis=basis)

if molecule == 'H2O':
    H_y_eq_dist = 0.75736617840905475162
    H_z_eq_dist = 0.58665191707013439891
    coord = 'O 0.0 0.0 0.0; H 0.0 -' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist) + '; H 0.0 ' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist)
    driver = PySCFDriver(atom='O 0.0 0.0 0.0; H 0.0 -' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist) + '; H 0.0 ' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist), basis=basis)
    frozen_core = 'yes'

if molecule == 'LiH':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
    coord = "Li 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist)
    driver = PySCFDriver(atom="Li 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist), basis=basis)

if molecule == 'N2':
    #eq_dist = 1.0
    #bond_dist = bond_stretch * eq_dist
    coord = "N 0.0 0.0 0.0; N 0.0 0.0 " + str(1 * bond_dist)
    driver = PySCFDriver(atom="N 0.0 0.0 0.0; N 0.0 0.0 " + str(1 * bond_dist), basis=basis)
    frozen_core = 'yes'

if molecule == 'BeH2':
    eq_dist = 1.0
#    bond_dist = bond_stretch * eq_dist
    coord = "Be 0.0 0.0 0.0; H 0.0 0.0 " + str(bond_stretch * eq_dist) + "; H 0.0 0.0 -" + str(bond_stretch * eq_dist)
    driver = PySCFDriver(
        atom="Be 0.0 0.0 0.0; H 0.0 0.0 " + str(bond_stretch * eq_dist) + "; H 0.0 0.0 -" + str(bond_stretch * eq_dist),
        basis=basis)
    frozen_core = 'yes'
    print(coord)

if molecule == 'H6':
    eq_dist = 1.0
    #bond_dist = bond_stretch * eq_dist
    coord = "H 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist) + "; H 0.0 0.0 " + str(
        2 * bond_dist) + "; H 0.0 0.0 " + str(3 * bond_dist) + "; H 0.0 0.0 " + str(
        4 * bond_dist) + "; H 0.0 0.0 " + str(5 * bond_dist)
    driver = PySCFDriver(atom="H 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist) + "; H 0.0 0.0 " + str(
        2 * bond_dist) + "; H 0.0 0.0 " + str(3 * bond_dist) + "; H 0.0 0.0 " + str(
        4 * bond_dist) + "; H 0.0 0.0 " + str(5 * bond_dist), basis=basis)


if molecule == 'C2H2':
    #eq_dist = 1.0
    c_eq_pos = 0.6013
    h_eq_pos = 1.6644
    bond_stretch = bond_dist
    c_pos = c_eq_pos*bond_stretch
    c_shift = c_pos - c_eq_pos
    h_pos = h_eq_pos+c_shift
    #bond_dist = bond_stretch * eq_dist
    #coord="C 0.0 0.0 0.6013; C 0.0 0.0 -0.6013; H 0.0 0.0 1.6644; H 0.0 0.0 -1.6644"
    coord="C 0.0 0.0 "+str(c_pos)+"; C 0.0 0.0 -"+str(c_pos)+"; H 0.0 0.0 "+str(h_pos)+"; H 0.0 0.0 -"+str(h_pos)
    driver = PySCFDriver(atom="C 0.0 0.0 "+str(c_pos)+"; C 0.0 0.0 -"+str(c_pos)+"; H 0.0 0.0 "+str(h_pos)+"; H 0.0 0.0 -"+str(h_pos), basis = basis)
    frozen_core = 'yes'


problem = driver.run()
print('Bond distance', bond_dist)
print(coord)
# if frozen_core == 'yes':
#     fc_transformer = FreezeCoreTransformer()
#     problem = fc_transformer.transform(problem)
if frozen_core == 'yes':
    fc_transformer = FreezeCoreTransformer()
    problem = fc_transformer.transform(problem)


if frozen_core == 'yes':
	frozen_energy_shift = problem.hamiltonian.constants['FreezeCoreTransformer']
if frozen_core =='no':
	frozen_energy_shift = 0.0

print (frozen_energy_shift)



hamiltonian = problem.hamiltonian.second_q_op()
#print(type(hamiltonian))
#print(hamiltonian.is_hermitian())
print ('JW mapping ....')

mapper = JordanWignerMapper()
# qubit_op = mapper.map(hamiltonian)
# jw_mapped_hamiltonian = mapper.map(hamiltonian)

if jw_mapped_ham_available == 'no':
    jw_mapped_hamiltonian = mapper.map(hamiltonian)
    print('qubit Hamiltonian', len(jw_mapped_hamiltonian))
    with open('jw_mapped_Hamiltonian_'+str(molecule)+'_'+str(bond_dist)+'_'+str(basis)+'.pkl', 'wb') as file:
        pickle.dump(jw_mapped_hamiltonian, file)

if jw_mapped_ham_available == 'yes':
    with open('jw_mapped_Hamiltonian_'+str(molecule)+'_'+str(bond_dist)+'_'+str(basis)+'.pkl', 'rb') as file:
        jw_mapped_hamiltonian = pickle.load(file)

print ('qubit Hamiltonian..')
print('qubit Hamiltonian', len(jw_mapped_hamiltonian))
#print (jw_mapped_hamiltonian)
print(type(jw_mapped_hamiltonian))

import numpy as np
from qiskit.quantum_info import SparsePauliOp
# Assuming 'problem' and 'JordanWignerMapper' are already defined/imported
# from qiskit_nature.second_q.mappers import JordanWignerMapper

# # Your existing code
# hamiltonian = problem.hamiltonian.second_q_op()
# mapper = JordanWignerMapper()
# jw_mapped_hamiltonian = mapper.map(hamiltonian)

# print('--- Original Qubit Hamiltonian ---')
# print(jw_mapped_hamiltonian)
# print(f"Is complex: {np.any(np.iscomplex(jw_mapped_hamiltonian.coeffs))}")
# print("-" * 35)


# --- SOLUTION: Create a new Hamiltonian with real coefficients ---

# WARNING: This operation discards the imaginary part of the coefficients.
# Please ensure this is physically appropriate for your specific problem.
# If the imaginary parts are large and not just numerical noise,
# you are effectively changing the Hamiltonian you are studying.

# 1. Extract the Pauli strings and the coefficients from the original operator
paulis = jw_mapped_hamiltonian.paulis
coeffs = jw_mapped_hamiltonian.coeffs

# 2. Take the real part of the coefficients
real_coeffs = np.real(coeffs)

#print(real_coeffs)

# 3. Create a new SparsePauliOp with the same Paulis but with the real coefficients
real_jw_hamiltonian = SparsePauliOp(paulis, coeffs=real_coeffs)

# 4. (Recommended) Simplify the new operator. This will remove any terms
#    whose coefficients became zero after taking the real part.
#real_jw_hamiltonian.simplify()


#print('\n--- Qubit Hamiltonian with Real Coefficients ---')
#print(real_jw_hamiltonian)
#print(f"Is complex: {np.any(np.iscomplex(real_jw_hamiltonian.coeffs))}")
#print("-" * 35)
jw_mapped_hamiltonian = real_jw_hamiltonian
# Now you can use 'real_jw_hamiltonian' in your subsequent algorithms.


num_particles = problem.num_particles
num_alpha_particles = num_particles[0]
num_beta_particles = num_particles[1]
number_of_particles = num_alpha_particles + num_beta_particles
num_total_particles = num_alpha_particles + num_beta_particles
number_of_particles = num_alpha_particles + num_beta_particles
num_spin_orbitals = problem.num_spin_orbitals
num_spatial_orbitals = problem.num_spatial_orbitals
nuclear_repulsion_energy_qiskit = problem.nuclear_repulsion_energy
orbital_energy_spatial = list(problem.orbital_energies)
num_alpha_sector_dets = math.comb(num_spatial_orbitals,num_alpha_particles)
num_symm_space_dets = num_alpha_sector_dets * num_alpha_sector_dets
#n_samples = n_samples_scale_factor * num_alpha_sector_dets
n_samples = int((n_samples_scale_factor/100) * num_symm_space_dets)
print ('n_samples',n_samples)
num_samples_for_gibbs = n_samples
print(len(orbital_energy_spatial))

oe = orbital_energy_spatial + orbital_energy_spatial
alpha_occupations = problem.orbital_occupations
beta_occupations = problem.orbital_occupations_b

print('number of particles:                   ', num_particles)
print('number of alpha spin particles         ', num_alpha_particles)
print('number of beta spin particles         ', num_beta_particles)
print('total number of particles              ', num_total_particles)
print('number of spin orbitals:               ', num_spin_orbitals)
print('nuclear repulsion energy:              ', nuclear_repulsion_energy_qiskit)
print('orbital energy spatial:                ', orbital_energy_spatial)
# print(oe)
print('Number of samples for rbm training', n_samples)
#*****************************************************************************************************************************************





# dominant_mbpt_det_list_qubit_conv = optimized_mbpt2.det_list
#
#
# import pprint
# input_det_list = dominant_mbpt_det_list_qubit_conv #+ sampled_dets_binary
# # det_list = sort_and_remove_duplicates(np.asarray(sampled_dets_binary))
# det_list = sort_and_remove_duplicates(np.asarray(input_det_list))
# b = solve_qubit(np.asarray(det_list), jw_mapped_hamiltonian)
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy + frozen_energy_shift
# # eigen_values.sort()
# # print()
# perturbative_diag_energy = eigen_values[0]
# print('Perturbative diag energy', perturbative_diag_energy)
# exit()




#********************************************************* :one and two electron integrals extraction: ****************************************************************************************

import re
def integer_finder_from_string(string):
    #returns integer list

    # Sample string
    #my_string = "+_12 -_10"

    # Search for integers following underscores using regular expression
    integers = re.findall(r'(?<=_)\d+', string)

    # Convert the integers to integers
    integers = [int(num) for num in integers]

#    print(integers)  # Output: [11, 0]
    return  integers



def one_body_qiskit(second_q_hamiltonian):
    h_indices = np.asarray(second_q_hamiltonian)
    one_body_indices = np.asarray(
        [i for i in h_indices if len(i) < 10])  # one body second q strings are of the length 7
    one_body_ints = np.zeros((num_spin_orbitals, num_spin_orbitals))
    for ind in one_body_indices:
        integer_from_string = integer_finder_from_string(ind)
        i = int(integer_from_string[1])
        a = int(integer_from_string[0])
        #       print(ind)
        #       print(second_q_hamiltonian[ind])
        one_body_ints[i, a] = second_q_hamiltonian[ind]
    return one_body_ints




def two_body_qiskit(second_q_hamiltonian):
    h_indices = np.asarray(second_q_hamiltonian)
    # print(h_indices)
    two_body_indices = np.asarray([i for i in h_indices if len(i) > 9])  # two body second q strings are of the length 15
    two_body_ints = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
    for ind in two_body_indices:
        integer_from_string = integer_finder_from_string(ind)
        i = int(integer_from_string[3])
        j = int(integer_from_string[2])
        b = int(integer_from_string[1])
        a = int(integer_from_string[0])
        two_body_ints[i, j, a, b] = second_q_hamiltonian[ind]
    return 2 * two_body_ints


one_body_ints = one_body_qiskit(hamiltonian)

two_body_ints = two_body_qiskit(hamiltonian)


print(two_body_ints.shape)
# print(two_body_ints)
#*********************************************************************************************************************************************




# ----------------------------------------- Initial HF state *------------------------------------------------------------------------------------------------------
init_state = HartreeFock(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, qubit_mapper=mapper)

hf_rev = (list(np.concatenate((alpha_occupations, beta_occupations))))

hf_rev = [int(a) for a in hf_rev]
# print(type(hf_rev))
print(hf_rev)


import warnings

warnings.filterwarnings("ignore")

import pyscf
import pyscf.cc
import pyscf.mcscf
import numpy as np


if initialization == 'ccsd':
    initial_ccsd = 'yes'
if initialization == 'mp2':
    initial_ccsd = 'no'
#ansatz_used = 'LUCJ'
# ansatz_used = 'ours'
ansatz_used = inp.ansatz_used

# Specify molecule properties
open_shell = False
spin_sq = 0
bond_dist = inp.bond_dist
# bond_dist = float(input())
# # Build N2 molecule
# mol = pyscf.gto.Mole()
# mol.build(
#     atom=[["N", (0, 0, 0)], ["N", (bond_dist, 0, 0)]],
#     basis="6-31g",
#     symmetry="Dooh",
# )
# n_frozen = 2
# active_space = range(n_frozen, mol.nao_nr())

# Build LiH molecule
mol = pyscf.gto.Mole()

if molecule == 'H4':
    mol.build(
        #atom=[["Li", (0, 0, 0)], ["H", (bond_dist, 0, 0)]],
        # atom=[["N", (0, 0, 0)], ["N", (bond_dist, 0, 0)]],
        #atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)], ["H", (4*bond_dist, 0, 0)], ["H", (5*bond_dist, 0, 0)]],
        atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)]],
        # atom = [["O", (0,0,0)], ]
        # basis="6-31g",
        # basis="sto-3g",
        basis=basis,
        symmetry="Coov",
    )

if molecule == 'H6':
    mol.build(
        #atom=[["Li", (0, 0, 0)], ["H", (bond_dist, 0, 0)]],
        # atom=[["N", (0, 0, 0)], ["N", (bond_dist, 0, 0)]],
        atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)], ["H", (4*bond_dist, 0, 0)], ["H", (5*bond_dist, 0, 0)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)]],
        # atom = [["O", (0,0,0)], ]
        # basis="6-31g",
        # basis="sto-3g",
        basis=basis,
        symmetry="Coov",
    )

if molecule == 'N2':
    mol.build(
        #atom=[["Li", (0, 0, 0)], ["H", (bond_dist, 0, 0)]],
        atom=[["N", (0, 0, 0)], ["N", (bond_dist, 0, 0)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)], ["H", (4*bond_dist, 0, 0)], ["H", (5*bond_dist, 0, 0)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)]],
        # atom = [["O", (0,0,0)], ]
        # basis="6-31g",
        # basis="sto-3g",
        basis=basis,
#        symmetry="Coov",
    )
if molecule == 'H2O':
    H_y_eq_dist = 0.75736617840905475162
    H_z_eq_dist = 0.58665191707013439891
    #bond_stretch = 2.5
    coord = 'O 0.0 0.0 0.0; H 0.0 -' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist) + '; H 0.0 ' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist)
    driver = PySCFDriver(atom='O 0.0 0.0 0.0; H 0.0 -' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist) + '; H 0.0 ' + str(bond_dist * H_y_eq_dist) + ' -' + str(
        bond_dist * H_z_eq_dist))
    frozen_core = 'yes'
    mol.build(
        #atom=[["Li", (0, 0, 0)], ["H", (bond_dist, 0, 0)]],
        atom=[["O", (0, 0, 0)], ["H", (0.0, - bond_dist * H_y_eq_dist, - bond_dist * H_z_eq_dist)], ["H", (0.0,  bond_dist * H_y_eq_dist, - bond_dist * H_z_eq_dist)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)], ["H", (4*bond_dist, 0, 0)], ["H", (5*bond_dist, 0, 0)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)]],
        # atom = [["O", (0,0,0)], ]
        # basis="6-31g",
        # basis="sto-3g",
        basis=basis,
#        symmetry="Coov",
    )




    c_eq_pos = 0.6013
    h_eq_pos = 1.6644
    bond_stretch = bond_dist
    c_pos = c_eq_pos*bond_stretch
    c_shift = c_pos - c_eq_pos
    h_pos = h_eq_pos+c_shift
    #bond_dist = bond_stretch * eq_dist
    #coord="C 0.0 0.0 0.6013; C 0.0 0.0 -0.6013; H 0.0 0.0 1.6644; H 0.0 0.0 -1.6644"
    coord="C 0.0 0.0 "+str(c_pos)+"; C 0.0 0.0 -"+str(c_pos)+"; H 0.0 0.0 "+str(h_pos)+"; H 0.0 0.0 -"+str(h_pos)
    driver = PySCFDriver(atom="C 0.0 0.0 "+str(c_pos)+"; C 0.0 0.0 -"+str(c_pos)+"; H 0.0 0.0 "+str(h_pos)+"; H 0.0 0.0 -"+str(h_pos), basis = basis)
    frozen_core = 'yes'



if molecule == 'C2H2':
    c_eq_pos = 0.6013
    h_eq_pos = 1.6644
    bond_stretch = bond_dist
    c_pos = c_eq_pos*bond_stretch
    c_shift = c_pos - c_eq_pos
    h_pos = h_eq_pos+c_shift
    mol.build(
        #atom=[["Li", (0, 0, 0)], ["H", (bond_dist, 0, 0)]],
        atom=[["C", (0, 0, c_pos)], ["C", (0, 0, -c_pos)], ["H", (0, 0, h_pos)], ["H", (0, 0, -h_pos)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)], ["H", (4*bond_dist, 0, 0)], ["H", (5*bond_dist, 0, 0)]],
        # atom=[["H", (0, 0, 0)], ["H", (1*bond_dist, 0, 0)], ["H", (2*bond_dist, 0, 0)], ["H", (3*bond_dist, 0, 0)]],
        # atom = [["O", (0,0,0)], ]
        # basis="6-31g",
        # basis="sto-3g",
        basis=basis,
#        symmetry="Coov",
    )
    #eq_dist = 1.0
    #bond_dist = bond_stretch * eq_dist
    #coord="C 0.0 0.0 0.6013; C 0.0 0.0 -0.6013; H 0.0 0.0 1.6644; H 0.0 0.0 -1.6644"
    #driver = PySCFDriver(atom="C 0.0 0.0 0.6013; C 0.0 0.0 -0.6013; H 0.0 0.0 1.6644; H 0.0 0.0 -1.6644", basis = basis)



# Define active space
if molecule == 'N2' or molecule == 'C2H2':
    n_frozen = 2
if molecule == 'H2O':
    n_frozen = 1
if molecule == 'H4' or molecule == 'H6':
    n_frozen = 0

active_space = range(n_frozen, mol.nao_nr())

# Get molecular integrals
scf = pyscf.scf.RHF(mol).run()
num_orbitals = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
num_elec_a = (n_electrons + mol.spin) // 2
num_elec_b = (n_electrons - mol.spin) // 2
cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

# # Compute exact energy
#exact_energy = cas.run().e_tot
if molecule == 'C2H2':
    exact_energy =0.0# -108.842683409169# -109.046671778080 #631g N2
else:
   print ('CASCI running...')
   cas_job = cas.run()
   exact_energy = cas_job.e_tot
   casci_coeff = cas_job.ci
   print ('CASCI en:       ', exact_energy)
   print ('CASCI Coeffs:   ', casci_coeff)
# exact_energy = -109.046671776590
# casci_coeff = np.zeros(5)
print(num_elec_a)
print(num_elec_b)
print(num_orbitals)
num_spin_orb = 2 * num_orbitals



print(num_orbitals)
print(n_electrons)

n_occ = num_elec_a
n_virt = num_orbitals - num_elec_a

#Next, we will create the ansatz. The LUCJ ansatz is a parameterized quantum circuit, and we will initialize it with t2 and t1 amplitudes obtained from a CCSD calculation.

if initialization == 'ccsd':
    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

    print(t1.shape)
    print(t2.shape)
    # print(t1)


if initialization == 'mp2':
    #-------------------------------------------------
    import pyscf.mcscf
    from pyscf import scf, mp
    import numpy as np


    t1 = np.zeros((n_occ, n_virt))
    t1_shape = t1.shape
    # t1_shape = (n_occ, n_virt)
    t1_mod = np.asarray([0.0]*(t1.size)).reshape(t1_shape)

    # print(t1_mod.shape)
    # print(t1_mod)
    t1 = t1_mod
    # print(t1)


    mf = scf.RHF(mol)
    mf.kernel()  # Solve Hartree-Fock equations

    # Step 3: Run MP2 calculation
    mp2 = mp.MP2(mf)
    mp2.frozen = n_frozen
    mp2.kernel()  # Solve MP2 equations

    # Step 4: Extract MP2 amplitudes
    t2 = mp2.t2  # Get the MP2 double excitation amplitudes

    # # Print results
    # print("MP2 Amplitudes (t2):\n", t2)
    #
    # print(t1.shape)
    # print(t2.shape)

    # occ = 5
    # t1 = np.zeros((occ,11))
    # print(t1.shape)
    # for i in range(5):
    #     for a in range(11):
    #         t1[i, a] += -2 * np.einsum('clk,lkc', eri[i, occ:, :occ, :occ], t2[:, :, a, :]) #diagram 1
    #         t1[i, a] += 2*np.einsum('dck,kdc', eri[occ:, occ:, occ + a, :occ], t2[i, :, :, :]) #diagram 2
    #         t1[i, a] += np.einsum('clk,klc', eri[i, occ:, :occ, :occ], t2[:, :, a, :])  #diagram 3
    #         t1[i, a] += -1 * np.einsum('cdk,kdc', eri[occ:, occ:, occ + a, :occ], t2[i, :, :, :])   #diagram 4

    occ = n_occ
    for i in range(n_occ):
        for a in range(n_virt):
            t1[i,a] += -2 * np.einsum('clk,lkc',eri[i, occ: , :occ, :occ], t2[:, :, :, a])
            t1[i, a] += 2 * np.einsum('dck, kdc', eri[occ:, occ:, occ+a, :occ], t2[i, :, :, :])
            t1[i,a] += np.einsum('clk, lkc', eri[i, occ:, :occ, :occ], t2[:, : , a, :])
            t1[i,a] += - np.einsum('dck, kcd', eri[occ:, occ: , occ+a, :occ], t2[i, :, :, :])
    # t1 = np.zeros((n_occ, n_virt))
    print(t1)
    # t1 [0,0] = 0.0
    # print(t1)


import ffsim
from qiskit import QuantumCircuit, QuantumRegister
from ffsim.variational.util import orbital_rotation_from_t1_amplitudes
from ffsim import gates, linalg

if ansatz_used == 'LUCJ':
    n_reps = n_reps
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)]

    print(alpha_beta_indices)
    print(alpha_alpha_indices)
    # exit()

    # alpha_beta_indices = [(0, 0)]
    # alpha_alpha_indices = [(1, 2)]

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )
    # s = '-0.11034036  1.00932729  0.23288141  0.37888579 -0.5909264   0.63981182 0.32270625  0.53007331  0.27854841 -0.48395177  0.58844495  0.3013404 -0.66239177 -1.1214353   0.29392115 -0.23814119  0.17378724  0.35941128 -0.51424094  0.00816906 -0.30789333 -0.15888519 -0.1590459   0.26497042 0.39154657 -0.13940232  0.33147373 -0.53206353  0.12683178 -0.11808454 -1.07131718  0.24092101 -0.36257257 -0.74019637 -0.06449269 -0.47912722    0.16774923 -0.22311928 -0.15125159 -0.02108628  0.1207545  -0.29712666    -0.21719534 -0.03305418 -0.02528324 -0.18612909 -0.91601263  0.20306919    1.86722943 -0.20211075 -0.11529633 -1.0330917   0.60820362  0.7362367    0.69328444  0.05924612 -0.5165878   0.39230147 -0.00293343 -0.56967154     -1.89456932 -0.22278144  0.15309232 -0.10735576 -0.28356452 -0.14800477    -0.24060796  0.18883659 -0.04143982 -0.02800064 -0.55649783 -0.24389646    -0.16088133 -0.09913321  0.66183903 -0.50233197 -0.19539466  0.95062752    -0.01629379  0.28566852  0.77000824 -0.00520323  0.62304048 -0.00544338    0.05058777  0.00202906 -0.01171379  0.05234723  0.58648262 -0.26213064    0.44991898  0.72130272 -0.09549314  0.6586465  -0.0890026   0.24561848    -0.00465057  0.17715194 -0.01785991 -0.20648395 -0.06862639 -0.00814276    -0.19149448  0.0689889   0.06338578  0.01314126  0.18666649 -0.31023754    0.10156536  0.07568121  0.09005816 -0.10061183  0.01353018 -0.00837904    -0.03382119 -0.12127991  0.26720747 -0.08742805 -0.03686958 -0.03287914    0.02321136 -0.09455688  0.02828618  0.07418899  0.02781761 -0.27095849    0.05211913  0.04770752 -0.06331037 -0.05313896'
    # # s = "1.2 3.4 5.6"
    # lst = list(map(float, s.split()))
    # print(lst)

    # x = (np.load('lucj_optimized_params.npy'))
    # params_lucj = []
    # for i in range(x.size):
    #     print(x[i])
    #     params_lucj.append(x[i])
    # print(params_lucj)
    # # ffsim.UCJOpSpinBalanced.final_orbital_rotation = t1
    # params_lucj = [0.1]*122
    # print(params_lucj)
    # ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(params=params_lucj , norb=num_orbitals ,n_reps=n_reps,
    #     interaction_pairs=(alpha_alpha_indices, alpha_beta_indices), with_final_orbital_rotation=True)




    nelec = (num_elec_a, num_elec_b)

    # create an empty quantum circuit
    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)

    # prepare Hartree-Fock state as the reference state and append it to the quantum circuit
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    print(circuit.decompose().decompose().decompose().count_ops())
    gates = circuit.decompose().decompose().decompose().count_ops()
    cx_count = gates['cx']
    # exit()
    # print(circuit.decompose())
    # K = orbital_rotation_from_t1_amplitudes(t1)
    # print(K)
    # print(K.shape)
    #
    # diag_coulomb_mats, orbital_rotations = linalg.double_factorized_t2(t2)
    # diag_coulomb_mats = diag_coulomb_mats.reshape(-1, num_orbitals, num_orbitals)[:n_reps]
    # diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    # exp_K1 = ffsim.qiskit.OrbitalRotationJW(num_orbitals, K.T.conj())
    # J = ffsim.qiskit.DiagCoulombEvolutionJW(num_orbitals, diag_coulomb_mats, time=-1.0)
    # exp_K2 = ffsim.qiskit.OrbitalRotationJW(num_orbitals, K)
    # circuit.append(exp_K1, qubits)
    # # circuit.append(J, qubits)
    # # circuit.append(K2, qubits)
    # print(circuit.decompose().decompose().decompose().decompose().decompose())
    # print(circuit.decompose().decompose().decompose().count_ops())
    # exit()
    circuit.measure_all()
#   ---------------------
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit_ibm_runtime import QiskitRuntimeService

    #service = QiskitRuntimeService(channel='ibm_quantum', token=QXToken)
    #backend = service.least_busy(operational=True, simulator=False)
    # backend = FakeSherbrooke()
    # backend = generic_backend

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # if num_spin_orbitals <= 32 :
    #     spin_a_layout = [0, 14, 18, 19, 20, 33, 39, 40, 41, 53, 60, 61, 62, 72, 81, 82]
    #     spin_b_layout = [2, 3, 4, 15, 22, 23, 24, 34, 43, 44, 45, 54, 64, 65, 66, 73]
    #     print(spin_a_layout)
    #     spin_a_layout = spin_a_layout[0:num_orbitals]
    #     spin_b_layout = spin_b_layout[0:num_orbitals]
    #     print(spin_a_layout)
    #
    #     initial_layout = spin_a_layout + spin_b_layout

#   ------------

    # from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    # from qiskit_ibm_runtime import QiskitRuntimeService
    #
    # # service = QiskitRuntimeService(channel='ibm_quantum', token=QXToken)
    # # backend = service.least_busy(operational=True, simulator=False)
    # # backend = FakeSherbrooke()
    # # backend = generic_backend
    #
    # from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    #
    # spin_a_layout = [0, 14, 18, 19, 20, 33, 39, 40, 41, 53, 60, 61, 62, 72, 81, 82]
    # spin_b_layout = [2, 3, 4, 15, 22, 23, 24, 34, 43, 44, 45, 54, 64, 65, 66, 73]
    # print(spin_a_layout)
    # spin_a_layout = spin_a_layout[0:num_orbitals]
    # spin_b_layout = spin_b_layout[0:num_orbitals]
    # print(spin_a_layout)
    #
    # initial_layout = spin_a_layout + spin_b_layout
    #
    # pass_manager = generate_preset_pass_manager(
    #     optimization_level=3, backend=backend, initial_layout=initial_layout
    # )
    #
    # # without PRE_INIT passes
    # isa_circuit = pass_manager.run(circuit)
    # print(f"Gate counts (w/o pre-init passes): {isa_circuit.count_ops()}")
    #
    # # with PRE_INIT passes
    # # We will use the circuit generated by this pass manager for hardware execution
    # pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    # isa_circuit = pass_manager.run(circuit)
    # print(f"Gate counts (w/ pre-init passes): {isa_circuit.count_ops()}")


#------------------------------------------------------------------------------------------------






from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler


#--------------------------------------------------

# from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeSherbrooke, FakePeekskill, FakeBoeblingenV2, FakeBelemV2
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, SamplerV2 as Sampler
# from collections import Counter


from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import QiskitRuntimeService



measured_circuit = circuit.copy()
measured_circuit.measure_all()

synthetic_noise = inp.synthetic_noise



from qiskit_addon_sqd.counts import generate_counts_uniform


if synthetic_noise == 'yes':
    noisy_sampler = Sampler(
        options=dict(backend_options=dict(noise_model=noise_model))
    )
    pass_manager = generate_preset_pass_manager(3, AerSimulator())


import numpy as np
import json
from qiskit_ibm_runtime import RuntimeEncoder
from qiskit_ibm_runtime import RuntimeDecoder
import qiskit_zigzag_layout
if sampler_data_available == 'no':
    if synthetic_noise == 'no':
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        # --------------------------
        # Qiskit cloud
        from qiskit_ibm_runtime import QiskitRuntimeService
        print ("here")
#        token = "xyz" #chayan
        token = "xyz" #rahul
        QiskitRuntimeService.save_account(
            token=token,
            channel="ibm_cloud",  # `channel` distinguishes between different account types.
            # instance= "xyz", # support-prems-us
            # instance="xyz", #Chayan, from research credit
            # instance = "xyz",
                       # "xyz", #Chayan from free 10mins
            # instance = "xyz", #from chayan_mbpt_rbm_sqd Flex plan
            instance = "xyz",#from chayan_mbpt_rbm_sqd Flex plan
            overwrite=True  # Only needed if you already have Cloud credentials.
        )

        # service=QiskitRuntimeService ()
        # #backend = service.backend(name="ibm_brisbane")
        # backend = service.backend(name="ibm_fez")

        service = QiskitRuntimeService()
        # backend = service.backend(name="ibm_brisbane")
        backend = service.backend(name="ibm_boston")
        # backend = service.least_busy(
        #     operational=True, simulator=False, min_num_qubits=127
        # )
        print(f"Using backend {backend.name}")

        #lucj implementation
        # initial_layout, _ = get_zigzag_physical_layout(num_orbitals, backend=backend)
        
        # if num_spin_orbitals > 32:
        get_zigzag_physical_layout = qiskit_zigzag_layout.get_zigzag_physical_layout
        initial_layout, _ = get_zigzag_physical_layout(num_spatial_orbitals, backend=backend)

        pass_manager = generate_preset_pass_manager(
            optimization_level=3, backend=backend, initial_layout=initial_layout
        )

        # without PRE_INIT passes
        isa_circuit = pass_manager.run(circuit)
        print(f"Gate counts (w/o pre-init passes): {isa_circuit.count_ops()}")
        

        # with PRE_INIT passes
        # We will use the circuit generated by this pass manager for hardware execution
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        isa_circuit = pass_manager.run(circuit)
        print(f"Gate counts (w/ pre-init passes): {isa_circuit.count_ops()}")
        #-------------------------------------------------------
        print ('Quantum Circuit Layout:		',isa_circuit.layout.final_index_layout())
        exit()

        noisy_sampler = Sampler(mode=backend) #Sampler(mode=backend, options={"default_shots":10000} )
        # pass_manager = generate_preset_pass_manager(optimization_level=3, backend= backend, initial_layout = initial_layout)
        #working code comment in the next line
        #pass_manager = generate_preset_pass_manager(optimization_level=3, backend= backend)

        # # without PRE_INIT passes
        # isa_circuit = pass_manager.run(circuit)
        # print(f"Gate counts (w/o pre-init passes): {isa_circuit.count_ops()}")
        #
        # # with PRE_INIT passes
        # # We will use the circuit generated by this pass manager for hardware execution
        # pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        # isa_circuit = pass_manager.run(circuit)
        # print(f"Gate counts (w/ pre-init passes): {isa_circuit.count_ops()}")


    isa_circuit = pass_manager.run(measured_circuit)
    # pub = (isa_circuit, params, 100)
    # job = noisy_sampler.run([pub])
    job = noisy_sampler.run([isa_circuit], shots=n_shots)
    # result = job.result()
    # pub_result = result[0]
    # print(pub_result.data.meas.get_counts())
    primitive_result = job.result()


#--------------------



    with open('sampler_result_'+str(molecule)+'_'+str(basis)+'_'+str(bond_dist)+'_'+str(n_shots)+'.json', "w") as file:
        json.dump(primitive_result, file, cls=RuntimeEncoder)


if sampler_data_available == 'yes':
    #with open('sampler_result_'+str(molecule)+'_'+str(basis)+'_'+str(bond_dist)+'_'+str(n_shots)+'.json', "r") as file:
    #    primitive_result = json.load(file, cls=RuntimeDecoder)
    if molecule == 'C2H2':
        with open('sampler_result_C2H2_631g_2.0_1000000.json') as file:
            primitive_result = json.load(file, cls=RuntimeDecoder)
    if molecule == 'N2':
        with open('sampler_result_N2_'+str(basis)+'_2.0_'+str(n_shots)+'.json') as file:
        #with open('sampler_result_N2_ccpvdz_1.0_4000.json') as file:
            primitive_result = json.load(file, cls=RuntimeDecoder)
    if molecule == 'H2O':
        with open('sampler_result_H2O_'+str(basis)+'_1.0_'+str(n_shots)+'.json') as file:
            primitive_result = json.load(file, cls=RuntimeDecoder)
            

#    else:
#        with open('sampler_result_'+str(molecule)+'_'+str(basis)+'_'+str(bond_dist)+'_'+str(n_shots)+'.json', "r") as file:
#            primitive_result = json.load(file, cls=RuntimeDecoder)

if mbpt_data_available == 'no':
    dominant_mbpt_det_array_qubit_conv = optimized_mbpt2.det_array
    print(dominant_mbpt_det_array_qubit_conv)
    print(dominant_mbpt_det_array_qubit_conv.shape)

if mbpt_data_available == 'yes':
    with open('det_list_mbpt_rank_'+str(mbpt_max_rank)+'_' + str(molecule) + '_' + str(basis) + '_' + str(
        bond_dist) + '.pkl', 'rb') as file:  # 'wb' means write binary mode
        dominant_mbpt_det_array_qubit_conv = pickle.load(file)
#        dominant_mbpt_det_array_qubit_conv1 = dominant_mbpt_det_array_qubit_conv.copy()
    print(dominant_mbpt_det_array_qubit_conv)
    print(dominant_mbpt_det_array_qubit_conv.shape)


print (dominant_mbpt_det_array_qubit_conv.shape)





# #------------------------ SD inclusion
# # ----------------------------------------- Initial HF state *------------------------------------------------------------------------------------------------------
# init_state = HartreeFock(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, qubit_mapper=mapper)
#
# hf_rev = (list(np.concatenate((alpha_occupations, beta_occupations))))
#
# hf_rev = [int(a) for a in hf_rev]
# # print(type(hf_rev))
# print(hf_rev)
#
#
# init_state = HartreeFock(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, qubit_mapper=mapper)
#
# ansatz = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations='sd',
#              qubit_mapper=mapper, initial_state=init_state)
#
# sd_ex_list = ansatz.excitation_list
# print(len(sd_ex_list))
#
# num_singles = 0
# singles_ex_list = []
# for ex in sd_ex_list:
#     if len(ex[0]) == 1:
#         num_singles += 1
#         singles_ex_list.append(ex)
# print(singles_ex_list)
#
# # mp2 = MP2InitialPoint()
# # mp2.ansatz = ansatz
# # mp2.problem = problem
# # result = mp2.to_numpy_array()#.tolist()
# # t2amplitudes = mp2.t2_amplitudes
# # print(t2amplitudes)
# # print('-----------------------------')
# # #print(mp2_val_red)
# # print(t2amplitudes[0,0,0,0])
# # #print(pruned_excitation_list[num_singles])
# #
# # # print(type(t2amplitudes))
# # print(t2amplitudes.shape)
# # # print('num alpha',num_alpha_particles)
# # # print(num_spatial_orbitals)
# # num_occ = num_alpha_particles
# # num_virt = num_spatial_orbitals - num_alpha_particles
# # print ('num OCC spatial:', num_occ)
# # print ('num VIRT spatial:', num_virt)
#
#
# # with open(str(stretch)+'_'+str(molecule)+'_sto6g_final_excitation_list.pkl', 'rb') as file:
# #     doubles_ex_list = pickle.load(file)
# # with open(str(stretch)+'_'+str(molecule)+'_sto6g_t_list.pkl', 'rb') as file:
# #     mp2_val_red = pickle.load(file)
#
#
# var_form = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations='sd',
#                qubit_mapper=mapper, initial_state=init_state)
#
# final_excitation_list = var_form.excitation_list
# #print (final_excitation_list)
# #exit()
# # num_params = len(final_excitation_list)
# fer_excitation_op = var_form.excitation_ops()  # getting the second_q operator for excitations in UCC
# final_excitation_list_pauli = list()
# for ex in fer_excitation_op:
#     final_excitation_list_pauli.append(mapper.map(ex))
#
# #print(final_excitation_list)
# # print('fermionic excitation operator from var_form --------------', fer_excitation_op[0])
#
# # print(len(final_excitation_list))
#
# # ******************************************************************************
#
# def excited_det_list(ex_list):
#     ex_det_list = list()
#     for i in ex_list:
#         ex_det = hf_rev.copy()
#         for j in i[0]:
#             ex_det[j] = 0
#         for k in i[1]:
#             ex_det[k] = 1
#         ex_det_list.append(ex_det)
#     return ex_det_list
#
# excited_dets = excited_det_list(final_excitation_list)
#
#
# all_dets = [hf_rev] + excited_dets
#
# all_dets_qubit_conv = [x[::-1] for x in all_dets]
#
# #print (all_dets_qubit_conv)
# det_array_sd = np.asarray(all_dets_qubit_conv)
# print (det_array_sd)
#
#
# dominant_mbpt_det_array_qubit_conv = np.vstack((dominant_mbpt_det_array_qubit_conv,det_array_sd))
# print (dominant_mbpt_det_array_qubit_conv.shape)
# #exit()
# #----------------------------------------------------------------
#














#exit()

pub_result = primitive_result[0]
counts = pub_result.data.meas.get_counts()
num_dets_sampled = len(counts)

#print(counts)
print('num_dets_sampled',num_dets_sampled)


keys = list(counts.keys())

# Convert counts into bitstring and probability arrays
bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)

sampled_dets_binary = []
print(bitstring_matrix_full)
print(bitstring_matrix_full.shape)
sampled_dets_binary_array = bitstring_matrix_full.astype(int)
print(sampled_dets_binary_array)
print(sampled_dets_binary_array.shape)


def create_alpha_beta_sectors_numpy(determinant_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a 2D array of determinants into alpha and beta sectors using
    efficient NumPy slicing.

    Args:
        determinant_array (np.ndarray): A 2D array where each row is a determinant.
                                        Must have an even number of columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two new 2D arrays:
                                       (alpha_sectors, beta_sectors).
    """
    if determinant_array.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    num_columns = determinant_array.shape[1]
    if num_columns % 2 != 0:
        raise ValueError("The number of columns must be even to split into two equal halves.")

    midpoint = num_columns // 2

    # Use array slicing, which is a highly optimized, single operation.
    # This creates a view of the original data without copying it, making it very fast.
    alpha_sectors = determinant_array[:, :midpoint]
    beta_sectors = determinant_array[:, midpoint:]

    return alpha_sectors, beta_sectors


def filter_particle_conserved_dets_numpy(dets: np.ndarray, num_particles: int) -> np.ndarray:
    """
    Filters a 2D array of determinants to keep only those with a specific
    number of particles (ones), using efficient boolean indexing.

    Args:
        dets (np.ndarray): A 2D array of binary determinants.
        num_particles (int): The required number of ones for a determinant to be kept.

    Returns:
        np.ndarray: A new 2D array containing only the rows that are
                    particle-conserved.
    """
    # 1. Calculate the sum of each row in a single vectorized operation.
    # `axis=1` tells NumPy to sum across the columns for each row.
    row_sums = np.sum(dets, axis=1)

    # 2. Create a boolean "mask" by comparing the sums to the target number.
    # This creates an array like [True, False, True, ...].
    mask = (row_sums == num_particles)

    # 3. Use the mask to select only the rows from the original array where
    # the mask is True. This is called boolean indexing and is extremely fast.
    particle_conserved_dets = dets[mask]

    return particle_conserved_dets


particle_conserved_dets = filter_particle_conserved_dets_numpy(sampled_dets_binary_array, number_of_particles)
#print(particle_conserved_dets)
print(len(particle_conserved_dets))
print(len(sampled_dets_binary_array))


def filter_spin_conserved_dets_numpy(p_conserved_dets: np.ndarray, num_alpha: int, num_beta: int) -> np.ndarray:
    """
    Filters a 2D array of particle-conserved determinants to keep only those
    that are also spin-conserved.

    Args:
        p_conserved_dets (np.ndarray): A 2D array of binary determinants.
        num_alpha (int): The required number of ones in the first half (alpha sector).
        num_beta (int): The required number of ones in the second half (beta sector).

    Returns:
        np.ndarray: A new 2D array containing only the spin-conserved rows.
    """
    # 1. Split the array into alpha and beta sectors.
    alpha_sectors, beta_sectors = create_alpha_beta_sectors_numpy(p_conserved_dets)

    # 2. Sum the particles in each sector for all rows simultaneously.
    alpha_sums = np.sum(alpha_sectors, axis=1)
    beta_sums = np.sum(beta_sectors, axis=1)

    # 3. Create a boolean mask for each condition.
    alpha_mask = (alpha_sums == num_alpha)
    beta_mask = (beta_sums == num_beta)

    # 4. Combine the masks and filter the original array.
    # The '&' performs a logical AND operation on the boolean arrays.
    final_mask = alpha_mask & beta_mask
    spin_conserved_dets = p_conserved_dets[final_mask]

    return spin_conserved_dets


particle_and_spin_conserved_dets = filter_spin_conserved_dets_numpy(particle_conserved_dets, num_alpha_particles, num_beta_particles)
num_dets_sampled = len(particle_and_spin_conserved_dets)
#print(particle_and_spin_conserved_dets)
print(particle_and_spin_conserved_dets.shape)
# print(len(particle_and_spin_conserved_dets))


def filter_prob_dict_numpy(allowed_dets_array: np.ndarray, original_dict: dict) -> dict:
    """
    Filters a dictionary based on an "allowed list" of determinants from a NumPy array.

    Args:
        allowed_dets_array (np.ndarray): A 2D NumPy array of binary determinants.
                                         These are the keys that will be kept.
        original_dict (dict): A dictionary where keys are string representations
                              of determinants (e.g., '1100') and values are
                              probabilities or counts.

    Returns:
        dict: A new dictionary containing only the key-value pairs where the key
              corresponds to a row in the allowed_dets_array.
    """
    # 1. Convert the NumPy array of allowed determinants into a set of tuples.
    # A set provides extremely fast (O(1) on average) membership checking.
    allowed_set = set(tuple(row) for row in allowed_dets_array)

    # 2. Use a dictionary comprehension to build the new, filtered dictionary.
    filtered_dict = {
        key: value for key, value in original_dict.items()
        # For each string key, convert it to a tuple of integers for the lookup
        if tuple(int(char) for char in key) in allowed_set
    }

    return filtered_dict


filtered_prob_dict = filter_prob_dict_numpy(particle_and_spin_conserved_dets, counts)
counts = filtered_prob_dict
keys = list(filtered_prob_dict.keys())



print(type(jw_mapped_hamiltonian))
print(len(sampled_dets_binary_array))

sampled_dets_binary_array = particle_and_spin_conserved_dets
num_sampled_dets = len(sampled_dets_binary_array)




#clear dictionary memory
import gc
del counts
del filtered_prob_dict
gc.collect()



#--------------------------------------------------------------------------------------------------------------------------------

def rev_to_qubit_convention_transformer(rev_conv_list):
    qubit_conv_list = [x[::-1] for x in rev_conv_list]  # because the list is not as per the rev convention
    return qubit_conv_list

def qubit_to_rev_convention_transformer(qubit_conv_list):
    rev_conv_list = [x[::-1] for x in qubit_conv_list]  # because the list is not as per the rev convention
    return rev_conv_list


def find_common_elements_from_lists(list1, list2):
    set1 = set(map(tuple, list1))
    set2 = set(map(tuple, list2))

    # Find the intersection of the two sets
    common_tuples = set1.intersection(set2)

    # Convert the common tuples back to a list of lists
    common_lists = [list(t) for t in common_tuples]

    # # Print the result
    # for item in common_lists:
    #     print(item)

    # print (common_lists)
    print(len(common_lists))

    return common_lists



def find_uncommon_rows_numpy(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Finds rows that are in either array but not in both (symmetric difference).

    Args:
        array1 (np.ndarray): The first 2D array.
        array2 (np.ndarray): The second 2D array.

    Returns:
        np.ndarray: A new 2D array containing only the uncommon rows.
    """
    # 1. Concatenate both arrays into a single large array.
    combined_array = np.concatenate((array1, array2), axis=0)

    # 2. Find the unique rows and their counts within the combined array.
    # A row present in both original arrays will have a count of 2.
    # A row present in only one will have a count of 1.
    unique_rows, counts = np.unique(combined_array, axis=0, return_counts=True)

    # 3. Filter for the rows where the count is exactly 1.
    uncommon_rows = unique_rows[counts == 1]

    return uncommon_rows


def subspace_ci_energy(ci_subspace):
    '''

    :param ci_subspace: subspace ci dets in bool, in qubit convention (i.e. for H4, HF should look like [00110011]) as an np.ndarray or list of dets
    :return: subspace ci energy and ci_coeffs
    '''
    b = solve_qubit(np.asarray(ci_subspace), jw_mapped_hamiltonian)
    eigen_values = b[0] + nuclear_repulsion_energy_qiskit + frozen_energy_shift
    eigen_values.sort()
    subspace_ci_en = eigen_values[0]
    subspace_ci_coeffs = b[1]
    subspace_ci_coeffs = subspace_ci_coeffs[:, 0].real

    return subspace_ci_en, subspace_ci_coeffs




def dominant_ci_dets(ci_coeffs, dets_list, filt_thresh):
    '''
    :param ci_coeffs:
    :param dets_list:
    :param filt_thresh:
    :return:
    '''

    dominant_ci_idx = np.where(np.abs(ci_coeffs) > filt_thresh)
    dominant_ci_idx = dominant_ci_idx[0]
    dominant_det = [dets_list[i] for i in dominant_ci_idx]
    dominant_ci_coeff = [ci_coeffs[i] for i in dominant_ci_idx]

    return dominant_det, dominant_ci_coeff






import pprint
import numpy as np
import timeit


def sort_and_remove_duplicates_manual(bitstring_matrix: np.ndarray) -> np.ndarray:
    """
    Sorts a bitstring matrix and removes duplicate entries using an efficient
    vectorized NumPy approach.

    The lowest bitstring values will be placed in the lowest-indexed rows.

    Args:
        bitstring_matrix: A 2D array of bools or integers (0/1)
                          where each row represents a single bitstring.

    Returns:
        A sorted version of `bitstring_matrix` without repeated rows.
    """
    if bitstring_matrix.size == 0:
        return np.array([[]])

    # 1. Vectorized Binary to Integer Conversion
    # Create an array of powers of 2 (e.g., [2^(N-1), 2^(N-2), ..., 1])
    n_bits = bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)

    # Convert all rows to integers in a single, fast operation
    # This is much faster than looping through each row.
    bsmat_as_ints = bitstring_matrix.dot(powers_of_2)

    # 2. Find Unique Rows and Their Original Indices
    # np.unique returns the sorted unique integer values and the indices of their
    # first appearance in the original `bsmat_as_ints` array.
    _, unique_indices = np.unique(bsmat_as_ints, return_index=True)

    # 3. Use the indices to select and sort the unique rows
    # Because `np.unique` sorts the integer values, the `unique_indices` are
    # already in an order that will produce a sorted final matrix.
    sorted_unique_matrix = bitstring_matrix[unique_indices]

    return sorted_unique_matrix






def add_mirror_dets(dets_array):
    alpha, beta = create_alpha_beta_sectors_numpy(dets_array)
    mirror = np.hstack((beta, alpha))

    dets_array_with_mirror_dets = np.vstack((dets_array, mirror))

    return dets_array_with_mirror_dets




# dominant_mbpt_det_array_qubit_conv_copy = dominant_mbpt_det_array_qubit_conv.copy()
# print(dominant_mbpt_det_array_qubit_conv_copy)
# print(dominant_mbpt_det_array_qubit_conv_copy.shape)
#
# alpha_dets , beta_dets = create_alpha_beta_sectors_numpy(dominant_mbpt_det_array_qubit_conv_copy)
# print(alpha_dets[:5, :])
# print(alpha_dets.shape)
# print(beta_dets[:5 , :])
#
# mirror_dets = np.hstack((beta_dets, alpha_dets))
# print(mirror_dets.shape)
# print(mirror_dets[:5, :])
#
# print(dominant_mbpt_det_array_qubit_conv.shape)
#
# dominant_mbpt_det_array_qubit_conv = np.vstack((dominant_mbpt_det_array_qubit_conv, mirror_dets))
#
# print(dominant_mbpt_det_array_qubit_conv.shape)


# print(sampled_dets_binary_array.shape)
# sampled_dets_binary_array = add_mirror_dets(sampled_dets_binary_array)
# print(sampled_dets_binary_array.shape)

print(dominant_mbpt_det_array_qubit_conv.shape)

dominant_mbpt_det_array_qubit_conv = add_mirror_dets(dominant_mbpt_det_array_qubit_conv)


print(dominant_mbpt_det_array_qubit_conv.shape)



#-----------------------------------------------------------------------------------------------------------

#input_det_list_qubit_conv = sampled_dets_binary + dominant_mbpt_det_array_qubit_conv

input_det_array_qubit_conv = np.vstack((sampled_dets_binary_array, dominant_mbpt_det_array_qubit_conv))
print(input_det_array_qubit_conv.shape)
#exit()
# det_list = sort_and_remove_duplicates(np.asarray(sampled_dets_binary))
import time
start_time = time.time()
sorted_det_list_qubit_conv = sort_and_remove_duplicates(input_det_array_qubit_conv)
end_time = time.time()
elapsed_time = end_time - start_time
print('Dets sorting elapsed_time', elapsed_time)
# b = solve_qubit(sorted_det_list_qubit_conv, jw_mapped_hamiltonian)
#
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy_qiskit + frozen_energy_shift
# perturbative_diag_energy = eigen_values[0]
# print('Perturbative diag energy', perturbative_diag_energy)
# print('CASCI Energy:            ', exact_energy)


print('**************************************************')

start_time = time.time()
sorted_det_array_qubit_conv = sort_and_remove_duplicates_manual(input_det_array_qubit_conv)
end_time = time.time()
elapsed_time = end_time - start_time
print('Dets sorting elapsed_time', elapsed_time)
















#-------------------------------------------------------------------------------------------------------------------------------------

#create random list of bitstrings for RBM output
import random


def generate_random_bitstrings(num_strings: int, length: int, num_ones: int) -> np.ndarray:
    """
    Generates a NumPy array of random binary strings with a specified number of ones.

    This function uses an efficient, vectorized approach, avoiding Python loops.

    Args:
        num_strings (int): The total number of bitstrings to generate (N).
        length (int): The length of each bitstring (L).
        num_ones (int): The exact number of ones each bitstring must contain.

    Returns:
        np.ndarray: A 2D NumPy array of shape (num_strings, length), where each
                    row is a randomly generated bitstring.

    Raises:
        ValueError: If the number of ones is greater than the total length
                    or less than zero.
    """
    # # --- Input Validation ---
    # if not 0 <= num_ones <= length:
    #     raise ValueError("The number of ones must be between 0 and the total length."

    # --- 1. Create the Prototype Array ---
    # This is a template with the correct number of 0s and 1s.
    # We use uint8 for memory efficiency with binary data.
    #rng = np.random.default_rng(42)  # reproducible RNG
    num_zeros = length - num_ones
    prototype_bitstring = np.array([1] * num_ones + [0] * num_zeros, dtype=np.uint8)

    # --- 2. Create the Full Matrix by Tiling the Prototype ---
    # This creates a (num_strings, length) array where every row is a copy
    # of the prototype_bitstring.
    result_array = np.tile(prototype_bitstring, (num_strings, 1))

    # --- 3. Shuffle Each Row Independently (Vectorized) ---
    # `np.apply_along_axis` applies the `np.random.permutation` function
    # to each row (axis=1) of the matrix. This is the core operation that
    # replaces the slow Python for loop.
    # shuffled_array = np.apply_along_axis(np.random.permutation, axis=1, arr=result_array)
    shuffled_array = np.apply_along_axis(np.random.permutation, axis=1, arr=result_array)

    return shuffled_array




import itertools


def combine_arrays_cartesian_numpy(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Combines two 2D NumPy arrays by creating the Cartesian product of their rows.

    Each row from array1 is concatenated with every row from array2
    to form the rows of a new, combined array.

    Args:
        array1: A 2D NumPy array of shape (N, M1).
        array2: A 2D NumPy array of shape (K, M2).

    Returns:
        np.ndarray: A new 2D array of shape (N * K, M1 + M2) containing
                    all Cartesian combinations.
    """
    # Get the number of rows from each array
    n_rows1 = array1.shape[0]
    n_rows2 = array2.shape[0]

    # --- 1. Repeat each row of the first array `n_rows2` times ---
    # This creates the left-hand side of the final combined matrix.
    # Example: If array1 is [[a1], [a2]] and n_rows2 is 3, this produces:
    # [[a1], [a1], [a1], [a2], [a2], [a2]]
    repeated_array1 = np.repeat(array1, n_rows2, axis=0)

    # --- 2. Tile the entire second array `n_rows1` times ---
    # This creates the right-hand side of the final combined matrix.
    # Example: If array2 is [[b1], [b2], [b3]] and n_rows1 is 2, this produces:
    # [[b1], [b2], [b3], [b1], [b2], [b3]]
    tiled_array2 = np.tile(array2, (n_rows1, 1))

    # --- 3. Concatenate the two resulting arrays horizontally ---
    # This stacks the arrays side-by-side to produce the final result.
    combined_array = np.concatenate([repeated_array1, tiled_array2], axis=1)

    return combined_array


def combine_arrays_elementwise(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Combines two 2D NumPy arrays in a one-to-one (element-wise) fashion.

    The 0th row of array1 is concatenated with the 0th row of array2,
    the 1st with the 1st, and so on. Assumes both arrays have the same number of rows.

    Args:
        array1: The first 2D NumPy array of shape (N, M1).
        array2: The second 2D NumPy array of shape (N, M2).

    Returns:
        A new 2D array of shape (N, M1 + M2) with the combined rows.
    """
    # For arrays of the same shape, a single concatenate is the most efficient method.
    return np.concatenate([array1, array2], axis=1)

def random_bitstrings_for_rbm(num_strings, length, num_ones):
    '''

    :param num_strings: the length of the array. Put as you like
    :param length: number of alpha or beta orbitals
    :param num_ones: number of alpha or beta particles
    :return:

    '''
    alpha_strings = generate_random_bitstrings(num_strings, length, num_ones)
    beta_strings = generate_random_bitstrings(num_strings, length, num_ones)

    alpha_beta_combined_string = combine_arrays_elementwise(alpha_strings, beta_strings)

    return alpha_beta_combined_string


#--------------------------------------------------------------------------------------------------------------------------------------











#---------------------------------------- From Sonaldeep RBM code
from sklearn.neural_network import BernoulliRBM
from collections import Counter
import math


def find_transitions_correct(n, n1, main_list, target_list):
    """
    Find the transitions required to convert main_list into target_list

    Parameters:
    n (int): Total number of orbitals.
    n1 (int): Number of alpha spin orbitals.
    main_list (list of int): The initial configuration of orbitals.
    target_list (list of int): The desired configuration of orbitals.

    Returns:
    tuple of lists: Two lists, one for source indices and one for destination indices.
    """

    # Splitting the lists into alpha and beta orbitals
    alpha_main = main_list[:n1]
    beta_main = main_list[n1:]
    alpha_target = target_list[:n1]
    beta_target = target_list[n1:]

    # check if sublists are equ(al
    if((sorted(alpha_main) != sorted(alpha_target)) or (sorted(beta_main) != sorted(beta_target))):
        print("Invalid Inputs")
        return 0
    # Function to find transitions for a specific spin type
    def find_spin_transitions(main, target):
        from_indices = []
        to_indices = []

        for i in range(len(main)):
            if main[i] == 1 and target[i] == 0:
                from_indices.append(i)
            elif main[i] == 0 and target[i] == 1:
                to_indices.append(i)

        return from_indices, to_indices

    # Finding transitions for alpha and beta spins
    alpha_from, alpha_to = find_spin_transitions(alpha_main, alpha_target)
    beta_from, beta_to = find_spin_transitions(beta_main, beta_target)

    # Adjusting beta indices to their actual positions in the combined list
    beta_from = [i + n1 for i in beta_from]
    beta_to = [i + n1 for i in beta_to]

    # Combining alpha and beta transitions
    from_indices = alpha_from + beta_from
    to_indices = alpha_to + beta_to

    # Ensuring transitions within the same spin type
    from_to_pairs = list(zip(from_indices, to_indices))

    # Filter out invalid transitions (from alpha to beta or vice versa)
    valid_transitions = [(f, t) for f, t in from_to_pairs if (f < n1) == (t < n1)]

    # Separate source and destination indices
    source_indices, dest_indices = zip(*valid_transitions) if valid_transitions else ([], [])

    # return (list(source_indices), list(dest_indices))
    return (source_indices, dest_indices)



import time


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid activation function."""
    return 1. / (1. + np.exp(-x))


def train_rbm_cd(
        training_data: np.ndarray,
        weights: np.ndarray,
        visible_bias: np.ndarray,
        hidden_bias: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        n_gibbs_steps: int
) -> tuple:
    """
    Trains an RBM using the Contrastive Divergence (CD-L) algorithm.

    This function implements the learning procedure described in the provided
    scientific text, updating the RBM parameters based on the difference
    between data statistics and model statistics.

    Args:
        training_data (np.ndarray): A 2D array of shape (n_samples, n_visible)
            containing the training vectors {v}.
        weights (np.ndarray): The RBM weight matrix (W) to be trained.
        visible_bias (np.ndarray): The visible layer bias vector (a) to be trained.
        hidden_bias (np.ndarray): The hidden layer bias vector (b) to be trained.
        n_epochs (int): The number of full passes through the training data.
        learning_rate (float): The learning rate (ϵ) for parameter updates.
        batch_size (int): The size of the minibatch (Nb) for each update step.
        n_gibbs_steps (int): The number of Gibbs sampling steps (L) for the
                             negative phase.

    Returns:
        A tuple containing the trained (weights, visible_bias, hidden_bias).
    """
    n_samples, n_visible = training_data.shape

    print("--- Starting RBM Training (Contrastive Divergence) ---")

    for epoch in range(n_epochs):
        # Shuffle data at the beginning of each epoch
        #rng = np.random.default_rng(seed=42)
        np.random.shuffle(training_data)

        # Process the data in minibatches
        for i in range(0, n_samples, batch_size):
            v_batch = training_data[i: i + batch_size]
            current_batch_size = v_batch.shape[0]

            # --- Step 1 & 2: Positive Phase (Calculate ⟨...⟩data) ---
            # Calculate p(h_k = 1 | v^(i)) for the input data
            prob_h_given_v_data = _sigmoid(np.dot(v_batch, weights) + hidden_bias)

            # Calculate the data-dependent expectations
            # <v_j h_k>_data
            data_associations = np.dot(v_batch.T, prob_h_given_v_data)
            # <h_k>_data (for bias update)
            data_hidden_avg = np.mean(prob_h_given_v_data, axis=0)
            # <v_k>_data (for bias update)
            data_visible_avg = np.mean(v_batch, axis=0)

            # --- Step 3: Negative Phase (Generate {v^(L), h^(L)}) ---
            # Start the Gibbs chain from the current data minibatch
            v_model = v_batch

            for _ in range(n_gibbs_steps):
                # Sample hidden layer given visible layer
                #np.random.seed(42)
                prob_h_model = _sigmoid(np.dot(v_model, weights) + hidden_bias)
                h_model_sample = (prob_h_model > np.random.rand(*prob_h_model.shape)).astype(np.uint8)

                # Sample visible layer given hidden layer (reconstruction)
                prob_v_model = _sigmoid(np.dot(h_model_sample, weights.T) + visible_bias)
                v_model = (prob_v_model > np.random.rand(*prob_v_model.shape)).astype(np.uint8)

            # This v_model is now v^(L)

            # Calculate p(h_k = 1 | v^(L)) for the model's "fantasy" particles
            prob_h_given_v_model = _sigmoid(np.dot(v_model, weights) + hidden_bias)

            # Calculate the model-dependent expectations
            # <v_j h_k>_model
            model_associations = np.dot(v_model.T, prob_h_given_v_model)
            # <h_k>_model
            model_hidden_avg = np.mean(prob_h_given_v_model, axis=0)
            # <v_k>_model
            model_visible_avg = np.mean(v_model, axis=0)

            # --- Step 4: Update RBM Parameters ---
            # Using equations (S.6), (S.7), and (S.8) from the text
            weights += learning_rate * (data_associations - model_associations) / current_batch_size
            visible_bias += learning_rate * (data_visible_avg - model_visible_avg)
            hidden_bias += learning_rate * (data_hidden_avg - model_hidden_avg)

        if (epoch + 1) % 1 == 0:
            # Optional: Add a metric like reconstruction error to monitor training
            recon_error = np.mean((v_batch - v_model) ** 2)
            print(f"Epoch {epoch + 1}/{n_epochs}, Reconstruction Error: {recon_error:.4f}")

    print("--- Training Complete ---")
    return weights, visible_bias, hidden_bias


def constrained_gibbs_step(
        v_current: np.ndarray,
        weights: np.ndarray,
        visible_bias: np.ndarray,
        hidden_bias: np.ndarray,
        num_particles: int,
        num_spatial_orbitals: int
) -> np.ndarray:
    """
    Performs one step of Gibbs sampling for an RBM, with constraints on
    particle number and spin conservation, as described in the provided text.

    This version assumes a blocked spin representation: the first K bits are
    spin-up orbitals, and the last K bits are spin-down orbitals.

    Args:
        v_current (np.ndarray): The current state of the visible layer. A 2D
            array of shape (n_samples, n_spin_orbitals).
        weights (np.ndarray): The RBM weight matrix of shape
            (n_spin_orbitals, n_hidden_units).
        visible_bias (np.ndarray): The RBM visible layer bias vector.
        hidden_bias (np.ndarray): The RBM hidden layer bias vector.
        num_particles (int): The total number of particles (electrons) to conserve.
        num_spatial_orbitals (int): The number of spatial orbitals (K). The
            total number of spin orbitals is 2*K.

    Returns:
        np.ndarray: The new state of the visible layer after one constrained
                    Gibbs sampling step.
    """
    # --- Input Validation ---
    if num_particles % 2 != 0:
        raise ValueError("Total number of particles must be even for spin conservation.")

    n_spin_orbitals = 2 * num_spatial_orbitals
    if v_current.shape[1] != n_spin_orbitals:
        raise ValueError(
            f"Shape of v_current ({v_current.shape[1]}) does not match n_spin_orbitals ({n_spin_orbitals}).")

    # --- Step 1 & 2: Sample the hidden layer (Standard Gibbs step) ---
    # Calculate hidden layer activation probabilities given the visible layer
    #np.random.seed(42)
    hidden_prob = 1 / (1 + np.exp(-np.dot(v_current, weights) - hidden_bias))

    # Sample the hidden layer states
    h_sample = (hidden_prob > np.random.rand(*hidden_prob.shape)).astype(np.uint8)

    # --- Step 3 (Modified): Sample the visible layer with constraints ---

    # First, calculate the occupation probabilities for the new visible layer
    # m_v_i = p(v_i = 1|h)
    visible_occupation_prob = 1 / (1 + np.exp(-np.dot(h_sample, weights.T) - visible_bias))

    # Separate probabilities into spin-up (first K indices) and spin-down (last K indices)
    prob_up = visible_occupation_prob[:, :num_spatial_orbitals]
    prob_down = visible_occupation_prob[:, num_spatial_orbitals:]

    # Normalize the probabilities for each spin sector for each sample in the batch
    # This is required for np.random.choice
    prob_up /= np.sum(prob_up, axis=1, keepdims=True)
    prob_down /= np.sum(prob_down, axis=1, keepdims=True)

    # Number of electrons to select in each spin channel
    n_up = num_particles // 2
    n_down = num_particles // 2

    # --- Tower Sampling Step (Vectorized) ---
    # We apply the sampling to each row (each sample in the batch) independently.

    # Define the items to choose from: the indices of the spatial orbitals
    orbital_indices = np.arange(num_spatial_orbitals)

    # Sample n_up spin-up orbitals for each sample in the batch
    chosen_up_indices = np.array([
        np.random.choice(orbital_indices, size=n_up, replace=False, p=p_row)
        for p_row in prob_up
    ])

    # Sample n_down spin-down orbitals for each sample in the batch
    chosen_down_indices = np.array([
        np.random.choice(orbital_indices, size=n_down, replace=False, p=p_row)
        for p_row in prob_down
    ])

    # --- Reconstruct the final visible state vector ---
    v_new = np.zeros_like(v_current, dtype=np.uint8)

    # Place '1's at the chosen spin-up locations (indices 0 to K-1)
    # This uses advanced NumPy indexing to update all samples in the batch at once.
    row_selector = np.arange(v_current.shape[0]).reshape(-1, 1)
    v_new[row_selector, chosen_up_indices] = 1

    # Place '1's at the chosen spin-down locations (indices K to 2K-1)
    # We must shift the chosen indices by K to place them in the second half.
    v_new[row_selector, chosen_down_indices + num_spatial_orbitals] = 1

    return v_new





#-------------------------------- New RBM training and generation


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid activation function."""
    return 1. / (1. + np.exp(-x))


def constrained_gibbs_sampler(
        v_start: np.ndarray,
        weights: np.ndarray,
        visible_bias: np.ndarray,
        hidden_bias: np.ndarray,
        num_particles: int,
        num_spatial_orbitals: int,
        n_steps: int
) -> np.ndarray:
    """
    Performs L steps of constrained Gibbs sampling using the Tower Sampling algorithm.

    This function follows the protocol described in the "Gibbs sampling" section
    of the provided text, ensuring particle number and spin are conserved.

    Args:
        v_start: The initial visible states to start the Gibbs chain from.
        weights: The RBM weight matrix (W).
        visible_bias: The visible layer bias vector (a).
        hidden_bias: The hidden layer bias vector (b).
        num_particles: The total number of electrons (N) to conserve.
        num_spatial_orbitals: The number of spatial orbitals (K).
        n_steps: The number of Gibbs sampling steps (L) to perform.

    Returns:
        The new visible states after L steps.
    """
    if num_particles % 2 != 0:
        raise ValueError("Total number of particles must be even for spin conservation.")

    v_model = v_start
    n_up = num_particles // 2
    orbital_indices = np.arange(num_spatial_orbitals)
    row_selector = np.arange(v_start.shape[0]).reshape(-1, 1)

    for _ in range(n_steps):
        # Step 2: Sample hidden layer from visible layer
        #np.random.seed(42)
        
        prob_h_model = _sigmoid(np.dot(v_model, weights) + hidden_bias)
        h_model_sample = (prob_h_model > np.random.rand(*prob_h_model.shape)).astype(np.uint8)

        # Step 3 (Modified): Sample visible layer using Tower Sampling
        # Calculate occupation probabilities m_v_i = p(v_i = 1|h)
        visible_occupation_prob = _sigmoid(np.dot(h_model_sample, weights.T) + visible_bias)

        # Separate probabilities into spin-up (first K) and spin-down (last K)
        prob_up = visible_occupation_prob[:, :num_spatial_orbitals]
        prob_down = visible_occupation_prob[:, num_spatial_orbitals:]

        # Normalize probabilities for each spin sector (required for np.random.choice)
        prob_up /= np.sum(prob_up, axis=1, keepdims=True)
        prob_down /= np.sum(prob_down, axis=1, keepdims=True)

        # Use Tower Sampling (implemented via np.random.choice) to select occupied orbitals
        chosen_up = np.array([np.random.choice(orbital_indices, size=n_up, replace=False, p=p) for p in prob_up])
        chosen_down = np.array([np.random.choice(orbital_indices, size=n_up, replace=False, p=p) for p in prob_down])

        # Reconstruct the new visible state vector
        v_model = np.zeros_like(v_start, dtype=np.uint8)
        v_model[row_selector, chosen_up] = 1
        v_model[row_selector, chosen_down + num_spatial_orbitals] = 1

    return v_model


# ------------ New2 RBM Training


class RBM(nn.Module):
    """
    PyTorch Implementation of a Bernoulli Restricted Boltzmann Machine (RBM).

    This class implements the core logic for an RBM, including the
    forward/backward passes (sampling) and the contrastive divergence
    gradient calculation.
    """

    def __init__(self, n_visible: int, n_hidden: int, seed: int = seed_val):
        """
        Initializes the RBM parameters.

        Args:
            n_visible (int): Number of units in the visible layer.
            n_hidden (int): Number of units in the hidden layer.
            seed (int): Random seed for reproducibility.
        """
        super(RBM, self).__init__()

        torch.manual_seed(seed)

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initialize weights and biases as nn.Parameter
        # This tells PyTorch to track gradients for these tensors.

        # Use Xavier/Glorot initialization for weights
        # This is a common practice for deep learning and works well for RBMs.
        limit = math.sqrt(6.0 / (n_visible + n_hidden))

        self.W = nn.Parameter(
            torch.rand(n_visible, n_hidden) * 2 * limit - limit
        )

        # Biases for visible and hidden layers
        self.b = nn.Parameter(torch.zeros(n_visible))  # visible_bias (a in your code)
        self.c = nn.Parameter(torch.zeros(n_hidden))  # hidden_bias (b in your code)

    def _prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability of hidden units being active given visible units.
        P(h=1 | v) = sigmoid(vW + c)
        """
        # v has shape (batch_size, n_visible)
        # W has shape (n_visible, n_hidden)
        # c has shape (n_hidden)
        # Result: (batch_size, n_hidden)
        return torch.sigmoid(torch.matmul(v, self.W) + self.c)

    def _sample_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Samples the hidden layer states (0 or 1) given visible units.
        """
        prob_h = self._prob_h_given_v(v)
        # torch.bernoulli samples from a Bernoulli distribution
        return torch.bernoulli(prob_h)

    def _prob_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability of visible units being active given hidden units.
        P(v=1 | h) = sigmoid(hW^T + b)
        """
        # h has shape (batch_size, n_hidden)
        # W.t() (transpose) has shape (n_hidden, n_visible)
        # b has shape (n_visible)
        # Result: (batch_size, n_visible)
        return torch.sigmoid(torch.matmul(h, self.W.t()) + self.b)

    def _sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Samples the visible layer states (0 or 1) given hidden units.
        This is the "reconstruction" step.
        """
        prob_v = self._prob_v_given_h(h)
        return torch.bernoulli(prob_v)

    def contrastive_divergence(self, v_batch: torch.Tensor, k: int) -> tuple:
        """
        Performs one step of Contrastive Divergence (CD-k).

        Args:
            v_batch (torch.Tensor): A batch of visible data.
            k (int): The number of Gibbs sampling steps (the 'k' in CD-k).

        Returns:
            A tuple of tensors:
            (grad_W, grad_b, grad_c)
        """

        # --- Positive Phase (Data Statistics) ---
        # 1. Calculate P(h|v_data)
        prob_h_data = self._prob_h_given_v(v_batch)

        # 2. Calculate expectations
        # We use the probabilities directly, as it's less noisy than sampling
        data_associations = torch.matmul(v_batch.t(), prob_h_data)
        data_visible_avg = v_batch.mean(dim=0)
        data_hidden_avg = prob_h_data.mean(dim=0)

        # --- Negative Phase (Model Statistics) ---
        # 3. Start Gibbs chain from the data
        v_model = v_batch

        # Perform k steps of Gibbs sampling
        for _ in range(k):
            h_model = self._sample_h_given_v(v_model)
            v_model = self._sample_v_given_h(h_model)

        # v_model is now the "fantasy" particle after k steps

        # 4. Calculate P(h|v_model)
        prob_h_model = self._prob_h_given_v(v_model)

        # 5. Calculate expectations from the model
        model_associations = torch.matmul(v_model.t(), prob_h_model)
        model_visible_avg = v_model.mean(dim=0)
        model_hidden_avg = prob_h_model.mean(dim=0)

        # --- Calculate Gradients ---
        # The update is (data_stats - model_stats)
        # This maximizes the log-likelihood
        batch_size = v_batch.size(0)

        grad_W = (data_associations - model_associations) / batch_size
        grad_b = (data_visible_avg - model_visible_avg)
        grad_c = (data_hidden_avg - model_hidden_avg)

        return grad_W, grad_b, grad_c

    def get_reconstruction_error(self, v_batch: torch.Tensor) -> float:
        """
        Calculates the Mean Squared Error of a 1-step reconstruction.
        Used for monitoring training progress.
        """
        with torch.no_grad():
            h = self._sample_h_given_v(v_batch)
            v_recon = self._prob_v_given_h(h)  # Use probabilities for stable error

            # Mean Squared Error
            error = torch.mean((v_batch - v_recon) ** 2)

        return error.item()



def constrained_gibbs_sampler_torch(
        rbm_model: RBM,
        v_start: torch.Tensor,
        num_particles: int,
        num_spatial_orbitals: int,
        n_steps: int,
        device: torch.device
) -> torch.Tensor:
    """
    Performs L steps of *constrained* Gibbs sampling using Tower Sampling.

    This function is a PyTorch-native implementation of your
    NumPy-based constrained_gibbs_sampler, ensuring particle number
    and spin are conserved.

    Args:
        rbm_model (RBM): The trained RBM model (must be on the correct device).
        v_start (torch.Tensor): The initial visible states to start the
                                Gibbs chain from (batch_size, n_visible).
        num_particles (int): The total number of electrons (N) to conserve.
        num_spatial_orbitals (int): The number of spatial orbitals (K).
        n_steps (int): The number of Gibbs sampling steps (L) to perform.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to run on.

    Returns:
        torch.Tensor: The new visible states after L steps (batch_size, n_visible).
    """
    if num_particles % 2 != 0:
        raise ValueError("Total number of particles must be even for spin conservation.")

    # Ensure n_up is an integer
    n_up = num_particles // 2
    v_model = v_start.to(device)

    # We don't need gradients for generation
    with torch.no_grad():
        for _ in range(n_steps):
            # Step 1: Sample hidden layer from visible layer
            h_model = rbm_model._sample_h_given_v(v_model)

            # Step 2: Get visible probabilities from hidden layer
            visible_occupation_prob = rbm_model._prob_v_given_h(h_model)

            # Step 3: Split into spin-up and spin-down sectors
            prob_up = visible_occupation_prob[:, :num_spatial_orbitals]
            prob_down = visible_occupation_prob[:, num_spatial_orbitals:]

            # Step 4: Normalize probabilities in each sector
            # Add a small epsilon to prevent division by zero (NaN)
            # Ensure sums are not zero before dividing
            prob_up_sum = torch.sum(prob_up, dim=1, keepdim=True)
            prob_down_sum = torch.sum(prob_down, dim=1, keepdim=True)

            # Avoid division by zero if a sector has zero probability sum
            prob_up = torch.where(prob_up_sum > 0, prob_up / (prob_up_sum + 1e-9), 0.0)
            prob_down = torch.where(prob_down_sum > 0, prob_down / (prob_down_sum + 1e-9), 0.0)

            # Handle cases where all probabilities in a row are zero (set to uniform)
            prob_up[prob_up_sum.squeeze() == 0] = 1.0 / num_spatial_orbitals
            prob_down[prob_down_sum.squeeze() == 0] = 1.0 / num_spatial_orbitals

            # Step 5: Perform Tower Sampling using torch.multinomial
            # This is a batched, parallel equivalent of np.random.choice
            chosen_up_indices = torch.multinomial(
                prob_up, num_samples=n_up, replacement=False
            )
            chosen_down_indices = torch.multinomial(
                prob_down, num_samples=n_up, replacement=False
            )

            # Step 6: Reconstruct the new visible state
            v_model.zero_()  # Reset the tensor to all zeros

            # Use scatter_ to place '1's at the chosen indices
            # dim=1 means we are scattering along the columns
            v_model.scatter_(dim=1, index=chosen_up_indices, value=1.0)
            v_model.scatter_(
                dim=1, index=(chosen_down_indices + num_spatial_orbitals), value=1.0
            )

    return v_model


#-------------------------------------------------------------------------------------------------------




def train_rbm(
    rbm_model: RBM,
    training_data: np.ndarray,
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    n_gibbs_steps: int,
    random_seed: int = seed_val
):
    """
    Trains the RBM model using the specified parameters.

    Args:
        rbm_model (RBM): The RBM model instance to train.
        training_data (np.ndarray): The dataset (e.g., from NumPy).
        n_epochs (int): Number of full passes over the data.
        learning_rate (float): Learning rate for parameter updates.
        batch_size (int): Size of minibatches.
        n_gibbs_steps (int): The 'k' for CD-k.
        random_seed (int): Seed for reproducibility.
    """
    
    # Set seed for reproducible data loading
    torch.manual_seed(random_seed)
    
    # --- 1. Set up device (use GPU if available) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training RBM on {device} ---")
    rbm_model.to(device) # Move model parameters to the device
    
    # --- 2. Create DataLoader ---
    # This handles batching and shuffling efficiently
    dataset = TensorDataset(torch.tensor(training_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Training Loop ---
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        
        for batch_data in loader:
            v_batch = batch_data[0].to(device) # Move data batch to device
            
            # --- Calculate Gradients ---
            # We don't need PyTorch's autograd (optimizer.zero_grad(), loss.backward())
            # because CD-k gives us the gradient updates directly.
            grad_W, grad_b, grad_c = rbm_model.contrastive_divergence(
                v_batch, k=n_gibbs_steps
            )
            
            # --- Manual Parameter Update ---
            # This is the SGD step, performed without an optimizer object
            with torch.no_grad():
                rbm_model.W += learning_rate * grad_W
                rbm_model.b += learning_rate * grad_b
                rbm_model.c += learning_rate * grad_c

            # Calculate reconstruction error for this batch
            epoch_loss += rbm_model.get_reconstruction_error(v_batch)

        # Print progress
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{n_epochs}, Avg. Reconstruction Error: {avg_loss:.4f}")

    end_time = time.time()
    print(f"--- Training Complete in {end_time - start_time:.2f} seconds ---")



#------------------------------------------------------------







def rbm_training_and_generation(binaries_data,n_components, learning_rate, batch_size, n_gibbs_sampling,n_alpha_orb, n_alpha_p, dominant_dets_for_gibbs, use_dominant_dets, total_dets):
    """
    Trains an RBM and generates new determinants based on a relative frequency threshold.

    Args:
        binaries_data (list): Array                   # List of lists representing the initial binary determinants.
        main_list (list): The reference determinant for filtering.
        n_components (int): Number of hidden units in the RBM.
        learning_rate (float): The learning rate for training.
        batch_size (int): The batch size for training.
        n_gibbs_sampling (int): Number of Gibbs sampling steps to generate new samples.
    """
    # data = np.array(binaries_data)
    data = binaries_data
#    np.random.seed(42)

    # Initialize and train the Restricted Boltzmann Machine
    rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate, batch_size=batch_size, n_iter=rbm_training_iter,
                       verbose=True, random_state=42)
                       

    # # Generate new samples through Gibbs sampling
    # new_samples = data
    print('************ New Sample Generation Starts *******************')
    if not use_dominant_dets:
        print('** Using random determinants for gibbs sampling starting point **')
        n_random_samples = num_samples_for_gibbs # len(binaries_data)

        # n_random_samples = 100000

        print('n_random_samples', n_random_samples)
        new_samples = np.array(random_bitstrings_for_rbm(n_random_samples,n_alpha_orb, n_alpha_p))
        new_samples = find_rows_not_in_array2_numpy(new_samples,total_dets)

        # print('rbm.gibbs(new_samples) starts')
    if use_dominant_dets:
        print('** Using dominant determinants for gibbs sampling starting point **')
#        new_samples = dominant_dets_for_gibbs
        if dominant_dets_for_gibbs.shape[0] < num_samples_for_gibbs:
	        # indices = np.random.choice(dominant_dets_for_gibbs.shape[0], size=dominant_dets_for_gibbs.shape[0], replace=False)# dominant_dets_for_gibbs
        	# new_samples = dominant_dets_for_gibbs[indices]

            n_random_samples = num_samples_for_gibbs  # len(binaries_data)
            print ('Sampling from RANDOM Dets')

            # n_random_samples = 100000

            print('n_random_samples', n_random_samples)
            new_samples = np.array(random_bitstrings_for_rbm(n_random_samples, n_alpha_orb, n_alpha_p))
            new_samples = find_rows_not_in_array2_numpy(new_samples, total_dets)

        else:
        	indices = np.random.choice(dominant_dets_for_gibbs.shape[0], size=num_samples_for_gibbs, replace=False)# dominant_dets_for_gibbs
        	print ('Sampling from DOMINANT Dets')
        	
        	new_samples = dominant_dets_for_gibbs[indices]
    start_time = time.time()
    if constrained_gibbs_generation == 'no':
        start_time = time.time()
        print('rbm.fit(data) starts')
        rbm.fit(data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('RBM fit elapsed_time', elapsed_time)
        print('rbm.fit(data) ends')
        rbm_weights = rbm.components_
        rbm_visible_bias = rbm.intercept_visible_
        rbm_hidden_bias = rbm.intercept_hidden_
        counter_gibbs = 0
        for _ in range(n_gibbs_sampling):
            print('counter_gibbs',counter_gibbs)
            print('Num samples for gibbs sampling', len(new_samples))
            # new_samples = constrained_gibbs_step(v_current=new_samples,weights=rbm_weights,visible_bias=rbm_visible_bias,hidden_bias=rbm_hidden_bias,num_particles=number_of_particles,num_spatial_orbitals=num_spatial_orbitals)
            new_samples = constrained_gibbs_sampler(v_start=new_samples,weights=rbm_weights,visible_bias=rbm_visible_bias,hidden_bias=rbm_hidden_bias,num_particles=number_of_particles,num_spatial_orbitals=num_spatial_orbitals,n_steps=n_gibbs_sampling)
            new_samples = find_rows_not_in_array2_numpy(new_samples,total_dets)
            counter_gibbs += 1


        # for _ in range(n_gibbs_sampling):
        #     new_samples = rbm.gibbs(new_samples)
    if constrained_gibbs_generation == 'yes':
        print('Generating constrained samples')
        # Use a good initialization scheme (Xavier/Glorot) for the weights
        # limit = np.sqrt(6.0 / (num_spin_orbitals + n_components))
        # initial_weights = np.random.uniform(-limit, limit, size=(num_spin_orbitals, n_components))
        # initial_visible_bias = np.zeros(num_spin_orbitals)
        # initial_hidden_bias = np.zeros(n_components)
        # rbm_weights, rbm_visible_bias, rbm_hidden_bias = train_rbm_cd(training_data=data,weights=initial_weights,visible_bias=initial_visible_bias,hidden_bias=initial_hidden_bias,n_epochs=rbm_training_iter,learning_rate=learning_rate,batch_size=batch_size,n_gibbs_steps=20)

        # start_time = time.time()
        # print('rbm.fit(data) starts')
        # rbm.fit(data)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('RBM fit elapsed_time', elapsed_time)
        # print('rbm.fit(data) ends')
        # # Access learned parameters
        # rbm_weights = rbm.components_  # shape (n_components, n_features)
        # rbm_visible_bias = rbm.intercept_visible_  # shape (n_features,)
        # rbm_hidden_bias = rbm.intercept_hidden_  # shape (n_components,)

        # rbm_weights = np.random.randn(num_spin_orbitals, n_components)
        # rbm_visible_bias = np.random.randn(num_spin_orbitals)
        # rbm_hidden_bias = np.random.randn(n_components)
        limit = np.sqrt(6.0 / (num_spin_orbitals + n_components))
        initial_weights = np.random.uniform(-limit, limit, size=(num_spin_orbitals, n_components))
        initial_visible_bias = np.zeros(num_spin_orbitals)
        initial_hidden_bias = np.zeros(n_components)
        #    rbm_model: RBM,
    # training_data: np.ndarray,
    # n_epochs: int,
    # learning_rate: float,
    # batch_size: int,
    # n_gibbs_steps: int,
    # random_seed: int = 42
        print('Using NEW RBM Module')
        rbm = RBM(n_visible=num_spin_orbitals, n_hidden=n_components)#, seed=SEED)
        #rbm_weights, rbm_visible_bias, rbm_hidden_bias = train_rbm_cd(training_data=data,weights=initial_weights,visible_bias=initial_visible_bias,hidden_bias=initial_hidden_bias,n_epochs=rbm_training_iter,learning_rate=learning_rate,batch_size=batch_size,n_gibbs_steps=20)
        train_rbm(rbm_model=rbm,training_data=data, n_epochs=rbm_training_iter, learning_rate=learning_rate, batch_size=batch_size,n_gibbs_steps=20)
        rbm_weights = rbm.W.data
        rbm_visible_bias = rbm.b.data
        rbm_hidden_bias = rbm.c.data
        # rbm_weights, rbm_visible_bias, rbm_hidden_bias = train_rbm_from_protocol(training_data=data,n_hidden=n_components,n_epochs=rbm_training_iter,learning_rate=learning_rate,batch_size=batch_size,n_gibbs_steps=n_gibbs_sampling)
        counter_gibbs = 0
        for _ in range(n_gibbs_sampling):
            print('counter_gibbs',counter_gibbs)
            print('Num samples for gibbs sampling', len(new_samples))
            # 1. Find out which device the model is on (e.g., 'cpu' or 'cuda')
            device = next(rbm.parameters()).device

            # 2. Convert your NumPy array to a PyTorch Tensor
            #    and move it to the same device as the model.
            new_samples = torch.tensor(new_samples, dtype=torch.float32).to(device)
            # new_samples = constrained_gibbs_step(v_current=new_samples,weights=rbm_weights,visible_bias=rbm_visible_bias,hidden_bias=rbm_hidden_bias,num_particles=number_of_particles,num_spatial_orbitals=num_spatial_orbitals)
            #new_samples = constrained_gibbs_sampler(v_start=new_samples,weights=rbm_weights,visible_bias=rbm_visible_bias,hidden_bias=rbm_hidden_bias,num_particles=number_of_particles,num_spatial_orbitals=num_spatial_orbitals,n_steps=n_gibbs_sampling)
            new_samples = constrained_gibbs_sampler_torch(rbm_model=rbm,v_start=new_samples,num_particles=number_of_particles,num_spatial_orbitals=num_spatial_orbitals,n_steps=n_gibbs_sampling, device=device)
            # 4. Convert the result back to a NumPy array
            # .cpu() - Moves the tensor from GPU to CPU (safe to call even if already on CPU)
            # .numpy() - Converts the CPU tensor to a NumPy array
            new_samples = new_samples.cpu().numpy().astype(np.uint8)
            new_samples = find_rows_not_in_array2_numpy(new_samples,total_dets)
            counter_gibbs += 1
    # print('new_samples',new_samples)
    print('len new_samples:     ',len(new_samples))
    end_time = time.time()
    elapsed_time = end_time-start_time
    print ('Elapsed_time', elapsed_time)

    print('rbm.gibbs(new_samples) ends')
    return new_samples.astype(int)





def weighted_random_choice(choices, weights):
  """
  Performs weighted random selection from a list of choices with given weights.

  Args:
    choices: A list of objects to choose from.
    weights: A list of weights associated with each choice. The weights must sum to 1.

  Returns:
    A randomly chosen object from the list based on the given weights.
  """

  # Check if weights sum to 1
  if not np.isclose(np.sum(weights), 1):
    raise ValueError("Weights must sum to 1.")

  # Normalize weights to be between 0 and 1
  weights = weights / np.sum(weights)

  # Create an array of cumulative weights
  cumulative_weights = np.cumsum(weights)

  # Generate a random number between 0 and 1
  #np.random.seed(42)
  random_number = np.random.random()

  # Find the index of the first element in cumulative_weights that is greater than random_number
  index = np.searchsorted(cumulative_weights, random_number)

  # Return the corresponding choice
  return choices[index]


def weighted_random_choice_numpy(choices: np.ndarray, weights: np.ndarray, num_samples: int = 1) -> np.ndarray:
    """
    Performs weighted random selection from a NumPy array of choices.

    This function is an efficient wrapper around `np.random.choice`.

    Args:
        choices (np.ndarray): A 2D array of items to choose from, where each row
                              is a single choice.
        weights (np.ndarray): A 1D array of weights associated with each choice.
                              Must have the same length as the number of rows in `choices`.
                              The weights will be normalized internally.
        num_samples (int): The number of samples to draw.

    Returns:
        np.ndarray: An array containing the randomly chosen samples.
    """
    if choices.shape[0] != weights.shape[0]:
        raise ValueError("The number of choices must match the number of weights.")

    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D array.")

    # Normalize the weights to ensure they sum to 1 for the probability distribution.
    normalized_weights = weights / np.sum(weights)
    #np.random.seed(42)

    # Get the indices of the choices based on the weights.
    chosen_indices = np.random.choice(
        a=choices.shape[0],  # Choose from the indices [0, 1, ..., N-1]
        size=num_samples,  # The number of items to select
        replace=True,  # Allow the same item to be chosen multiple times
        p=normalized_weights  # The probability associated with each index
    )

    # Return the chosen rows from the original choices array.
    return choices[chosen_indices]






#-----------------------------------------------
#Training Data Generation





def sample_list_generator_for_rbm(basis_determinants, ci_coefficients):
    '''

    :param basis_determinants:
    :param ci_coefficients:
    :return: training data for RBM
    '''
    # --- 1. Define Your System and CI Results ---

    # System parameters (e.g., for N2 with sto-3g and frozen core)
    # N_SPATIAL_ORBITALS = 6 # K
    # N_PARTICLES = 6        # N (must be even)

    # This is your basis of unique determinants from a CI calculation
    # Each row is a flat 12-bit string (6 alpha, 6 beta)
    # basis_determinants = np.array([
    #     # HF state: orbitals 0,1,2 occupied for alpha and beta
    #     [1, 1, 1, 0, 0, 0,   1, 1, 1, 0, 0, 0],
    #     # A doubly-excited state
    #     [1, 1, 0, 1, 0, 0,   1, 1, 0, 1, 0, 0],
    #     # Another important determinant
    #     [1, 0, 1, 1, 0, 0,   1, 0, 1, 1, 0, 0],
    # ], dtype=np.uint8)

    # These are the corresponding CI coefficients from diagonalization
    # ci_coefficients = np.array([0.98, -0.15, 0.12])


    # --- 2. Construct the Training Set ---

    # Set parameters from the paper
    # M = 50000  # Total size of the training set
    M = 1000000  # Total size of the training set

    # Calculate the probabilities (squared CI coefficients)
#    probabilities = ci_coefficients**2
    probabilities = np.abs(ci_coefficients)

    # Normalize the probabilities to ensure they sum to 1
    normalized_probabilities = probabilities / np.sum(probabilities)

    print("--- Preparing Training Data ---")
    print(f"Basis Determinants Shape: {basis_determinants.shape}")
    print(f"Probabilities for each determinant: {normalized_probabilities}")
    #rng = np.random.default_rng(42)
    # Use np.random.choice to draw M samples according to the probabilities
    # This efficiently creates the weighted training set in one step.
    # chosen_indices = np.random.choice(
    #     a=basis_determinants.shape[0], # Choose from indices [0, 1, 2, ...]
    #     size=M,
    #     replace=True,                  # Allow determinants to be chosen multiple times
    #     p=normalized_probabilities     # The probability for each index
    # )
    chosen_indices = np.random.choice(
        a=basis_determinants.shape[0], # Choose from indices [0, 1, 2, ...]
        size=M,
        replace=True,                  # Allow determinants to be chosen multiple times
        p=normalized_probabilities     # The probability for each index
    )

    # The final training data is constructed by selecting the chosen determinants
    training_data = basis_determinants[chosen_indices]
    return training_data





def find_rows_not_in_array2_numpy(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Finds rows that are present in array1 but not in array2.

    Args:
        array1 (np.ndarray): The array to check from (e.g., new determinants).
        array2 (np.ndarray): The array to check against (e.g., old determinants).

    Returns:
        np.ndarray: A new array containing rows from array1 that are not in array2.
    """
    # If the second array is empty, all of the first array is unique.
    if array2.size == 0:
        return array1
    # If the first array is empty, there's nothing to return.
    if array1.size == 0:
        return array1

    # An efficient way to compare rows in NumPy is to view them as a single
    # item of a structured void type.
    # This creates a 1D view of each array where each element is a full row.
    dtype = np.dtype((np.void, array1.dtype.itemsize * array1.shape[1]))
    array1_view = np.ascontiguousarray(array1).view(dtype)
    array2_view = np.ascontiguousarray(array2).view(dtype)

    # Use np.in1d with invert=True to find elements in array1_view not in array2_view
    mask = np.in1d(array1_view.ravel(), array2_view.ravel(), assume_unique=False, invert=True)

    # Use this mask to select the unique rows from the original array1
    unique_rows = array1[mask]

    return unique_rows

################################  Geneartion_STEP  ###########################


#n_components = 23
n_hidden_factor = inp.n_hidden_factor
n_components =n_hidden_factor * num_spin_orbitals #2*num_spin_orbitals
learning_rate = 0.001 #0.001984586278339788
batch_size = 10 # 90
n_gibbs_sampling = n_gibbs_sampling
# n_gibbs_sampling = 1000



# use davidson's diagonalizer
from qiskit_addon_sqd.qubit import project_operator_to_subspace
from qiskit_addon_sqd.qubit import sort_and_remove_duplicates





def remove_duplicate_manual(data):
    unique_ordered_data = []
    seen = set()

    for sublist in data:
        # Convert sublist to a tuple to check if we've seen it
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in seen:
            unique_ordered_data.append(sublist)
            seen.add(sublist_tuple)
    return unique_ordered_data


#----------------------------- operator projector efficient

import timeit
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from scipy.sparse import coo_matrix, spmatrix
from scipy.sparse.linalg import eigsh
#from qiskit.quantum_info import Pauli, SparsePauliOp


# --- New, Efficient Implementation -------------------------------------

def _int_conversion_numpy(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Efficiently converts a binary matrix to a 1D array of integers."""
    # Explicitly cast the input matrix to an integer type for the dot product.
    # This prevents TypeErrors with bitwise operations later.
    int_bitstring_matrix = bitstring_matrix.astype(np.uint64)
    n_bits = int_bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return int_bitstring_matrix.dot(powers_of_2)


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sorts and removes duplicates from a bitstring matrix."""
    int_reps = _int_conversion_numpy(bitstring_matrix)
    _, unique_indices = np.unique(int_reps, return_index=True)
    # The indices from np.unique are already sorted
    return bitstring_matrix[unique_indices]


def project_operator_to_subspace_efficient_v2(
        bitstring_matrix: np.ndarray,
        hamiltonian,  # : SparsePauliOp,
        *,
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with high memory efficiency.

    This version uses pre-allocation and batching to minimize peak memory usage.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64 for uint64 conversion.")

    d, n_qubits = bitstring_matrix.shape

    # --- 1. Pre-computation ---
    if verbose: print("Pre-computing integer representations...")
    # This array is used frequently, so compute it once.
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    # Separate Pauli terms into diagonal (only I, Z) and off-diagonal (contains X or Y)
    diag_paulis, diag_coeffs = [], []
    off_diag_paulis, off_diag_coeffs = [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- IMPROVEMENT: Pre-allocate arrays for COO matrix data ---
    # Start with an estimated size to reduce the number of resizes.
    # Estimate: all diagonal elements + average of N non-zeroes per off-diagonal term.
    estimated_nnz = d + len(off_diag_paulis) * n_qubits
    coo_rows = np.empty(estimated_nnz, dtype=np.int32)
    coo_cols = np.empty(estimated_nnz, dtype=np.int32)
    coo_data = np.empty(estimated_nnz, dtype="complex128")
    nnz_counter = 0

    # --- 2. Process Diagonal Pauli Terms (with Batching) ---
    if verbose: print("Processing diagonal Pauli terms in batches...")
    if diag_paulis:
        diagonal_values = np.zeros(d, dtype="complex128")
        batch_size = 512  # Process 512 Pauli terms at a time to keep parities matrix small

        for i in range(0, len(diag_paulis), batch_size):
            batch_paulis = diag_paulis[i:i + batch_size]
            batch_coeffs = diag_coeffs[i:i + batch_size]

            # The z_matrix is now smaller: (batch_size, n_qubits)
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)

            # The intermediate parities matrix is much smaller: (d, batch_size)
            parities_batch = (-1) ** (bitstring_matrix @ z_matrix_batch.T)

            # Accumulate the results for the batch
            diagonal_values += parities_batch @ np.array(batch_coeffs)

        # Add diagonal values to our pre-allocated COO arrays
        coo_rows[:d] = np.arange(d)
        coo_cols[:d] = np.arange(d)
        coo_data[:d] = diagonal_values
        nnz_counter = d

    # --- 3. Process Off-Diagonal Pauli Terms (Writing Directly to Arrays) ---
    if verbose: print("Processing off-diagonal Pauli terms...")
    for i, pauli in enumerate(off_diag_paulis):
        coeff = off_diag_coeffs[i]
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]

        connected_ints = int_array_rows ^ x_mask
        col_indices = np.searchsorted(int_array_rows, connected_ints)

        in_bounds_mask = col_indices < d
        valid_mask = np.zeros(d, dtype=bool)
        valid_in_bounds_indices = col_indices[in_bounds_mask]
        valid_mask[in_bounds_mask] = (int_array_rows[valid_in_bounds_indices] == connected_ints[in_bounds_mask])

        if not np.any(valid_mask):
            continue

        row_indices = np.arange(d)[valid_mask]
        final_col_indices = col_indices[valid_mask]

        z_parities = np.sum(bitstring_matrix[valid_mask] & pauli_z_rev, axis=1)
        # FIX: Cast phase to complex type to accommodate multiplication by a complex number
        phase = ((-1) ** z_parities).astype("complex128")
        phase *= (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        data_segment = coeff * phase
        num_new_entries = len(row_indices)

        # --- IMPROVEMENT: Grow arrays if needed and fill them ---
        if nnz_counter + num_new_entries > len(coo_rows):
            new_size = max(len(coo_rows) * 2, nnz_counter + num_new_entries)
            coo_rows.resize(new_size, refcheck=False)
            coo_cols.resize(new_size, refcheck=False)
            coo_data.resize(new_size, refcheck=False)

        start, end = nnz_counter, nnz_counter + num_new_entries
        coo_rows[start:end] = row_indices
        coo_cols[start:end] = final_col_indices
        coo_data[start:end] = data_segment
        nnz_counter += num_new_entries

    # --- 4. Construct Final Sparse Matrix ---
    if nnz_counter == 0:
        return coo_matrix((d, d), dtype="complex128")

    # --- IMPROVEMENT: Trim unused space from pre-allocated arrays ---
    final_rows = coo_rows[:nnz_counter]
    final_cols = coo_cols[:nnz_counter]
    final_data = coo_data[:nnz_counter]

    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype="complex128")




#----------------------- Memory Efficient Projection ------------------------------

import time
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import Pauli, SparsePauliOp
from memory_profiler import profile  # For demonstrating memory savings


# --- Helper Functions ---

def _int_conversion_numpy(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Efficiently converts a binary matrix to a 1D array of integers."""
    int_bitstring_matrix = bitstring_matrix.astype(np.uint64)
    n_bits = int_bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return int_bitstring_matrix.dot(powers_of_2)


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sorts and removes duplicates from a bitstring matrix."""
    int_reps = _int_conversion_numpy(bitstring_matrix)
    _, unique_indices = np.unique(int_reps, return_index=True)
    return bitstring_matrix[unique_indices]

    # hamiltonian = SparsePauliOp(hamiltonian.paulis, coeffs=hamiltonian.coeffs.real)
    #
    # coeff_dtype = np.float64

# --- Memory-Efficient Implementation ---

@profile
def project_operator_to_subspace_memory_efficient(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = 10000,
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with high memory efficiency
    by processing the off-diagonal components in batches.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    d, n_qubits = bitstring_matrix.shape

    # --- 1. Pre-computation ---
    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    # --- 2. Separate Pauli Terms ---
    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- 3. Process Diagonal Terms (Already Memory Efficient) ---
    if verbose: print("Processing diagonal Pauli terms...")
    if diag_paulis:
        z_matrix = np.array([p.z[::-1] for p in diag_paulis], dtype=bool)
        parities = (-1) ** (bitstring_matrix @ z_matrix.T)
        diagonal_values = parities @ np.array(diag_coeffs)
        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    # --- 4. Process Off-Diagonal Terms in Batches ---
    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in batches of size {batch_size}...")
    for i, pauli in enumerate(off_diag_paulis):
        coeff = off_diag_coeffs[i]
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        # --- BATCHING IMPROVEMENT ---
        # Loop over the determinants in smaller chunks to keep intermediate arrays small.
        for batch_start in range(0, d, batch_size):
            batch_end = min(batch_start + batch_size, d)

            # Create small slices of the main arrays
            int_array_batch = int_array_rows[batch_start:batch_end]
            bitstring_batch = bitstring_matrix[batch_start:batch_end]

            # All these arrays are now small (size <= batch_size)
            connected_ints_batch = int_array_batch ^ x_mask
            col_indices_batch = np.searchsorted(int_array_rows, connected_ints_batch)

            in_bounds_mask = col_indices_batch < d
            valid_mask = np.zeros_like(in_bounds_mask, dtype=bool)
            valid_in_bounds_indices = col_indices_batch[in_bounds_mask]

            # This check is now much smaller and safer
            valid_mask[in_bounds_mask] = (
                        int_array_rows[valid_in_bounds_indices] == connected_ints_batch[in_bounds_mask])

            if not np.any(valid_mask):
                continue

            # Row indices must be offset by the batch start position
            row_indices = (np.arange(len(valid_mask)) + batch_start)[valid_mask]
            final_col_indices = col_indices_batch[valid_mask]

            z_parities = np.sum(bitstring_batch[valid_mask] & pauli_z_rev, axis=1)
            phase = (-1) ** z_parities

            data_segment = coeff * phase * y_phase_factor

            all_rows.append(row_indices)
            all_cols.append(final_col_indices)
            all_data.append(data_segment)

    # --- 5. Construct Final Sparse Matrix ---
    if not all_data:
        return coo_matrix((d, d), dtype="complex128")

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # Sum up duplicate entries that might result from different Pauli terms
    # connecting the same two determinants.
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype="complex128").tocsr().tocoo()
    #return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype=coeff_dtype).tocsr().tocoo()
#------------------------------------------------------------------------


import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import SparsePauliOp


# Assuming _int_conversion_numpy is defined elsewhere
@profile
def project_operator_to_subspace_memory_efficient2(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = projection_batch_size,
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace, forcing the result to be
    a real-valued matrix with memory-efficient dtypes.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    d, n_qubits = bitstring_matrix.shape

    # --- 1. Setup Memory-Efficient Data Types ---
    coeff_dtype = np.float64
    if d < np.iinfo(np.int32).max:
        index_dtype = np.int32
        if verbose: print(f"Using 'int32' for matrix indices (d={d}).")
    else:
        index_dtype = np.int64
        if verbose: print(f"Using 'int64' for matrix indices (d={d}).")

    if np.iscomplexobj(hamiltonian.coeffs):
        if verbose:
            print("WARNING: Input Hamiltonian has complex coefficients.")
            print("The real part of all computed matrix elements will be taken.")

    # --- 2. Pre-computation ---
    if verbose: print("Pre-computing integer representations and sorting...")
    # This algorithm requires the integer array to be sorted for np.searchsorted
    int_array_rows = np.sort(_int_conversion_numpy(bitstring_matrix))

    all_rows, all_cols, all_data = [], [], []

    # --- 3. Separate Pauli Terms ---
    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- 4. Process Diagonal Terms ---
    if verbose: print("Processing diagonal Pauli terms...")
    if diag_paulis:
        z_matrix = np.array([p.z[::-1] for p in diag_paulis], dtype=bool)
        parities = (-1) ** (bitstring_matrix @ z_matrix.T)

        # Take the real part of the coefficients before the dot product
        real_diag_coeffs = np.array(diag_coeffs).real
        diagonal_values = parities @ real_diag_coeffs

        all_rows.append(np.arange(d, dtype=index_dtype))
        all_cols.append(np.arange(d, dtype=index_dtype))
        all_data.append(diagonal_values.astype(coeff_dtype, copy=False))

    # --- 5. Process Off-Diagonal Terms in Batches ---
    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in batches of {batch_size}...")
    for i, pauli in enumerate(off_diag_paulis):
        coeff = off_diag_coeffs[i]
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        for batch_start in range(0, d, batch_size):
            batch_end = min(batch_start + batch_size, d)
            int_array_batch = int_array_rows[batch_start:batch_end]
            bitstring_batch = bitstring_matrix[batch_start:batch_end]

            connected_ints_batch = int_array_batch ^ x_mask
            col_indices_batch = np.searchsorted(int_array_rows, connected_ints_batch)

            in_bounds_mask = col_indices_batch < d
            valid_mask = np.zeros_like(in_bounds_mask, dtype=bool)

            # Check only indices that are within the bounds of the array
            valid_in_bounds_indices = col_indices_batch[in_bounds_mask]
            valid_mask[in_bounds_mask] = (
                    int_array_rows[valid_in_bounds_indices] == connected_ints_batch[in_bounds_mask]
            )

            if not np.any(valid_mask):
                continue

            row_indices = (np.arange(len(valid_mask)) + batch_start)[valid_mask]
            final_col_indices = col_indices_batch[valid_mask]
            z_parities = np.sum(bitstring_batch[valid_mask] & pauli_z_rev, axis=1)
            phase = (-1) ** z_parities

            # Calculate the full, complex matrix element
            data_segment = coeff * phase * y_phase_factor

            # Take the real part before appending
            all_rows.append(row_indices.astype(index_dtype, copy=False))
            all_cols.append(final_col_indices.astype(index_dtype, copy=False))
            all_data.append(data_segment.real.astype(coeff_dtype, copy=False))

    # --- 6. Construct Final Sparse Matrix ---
    if not all_data:
        return coo_matrix((d, d), dtype=coeff_dtype)

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype=coeff_dtype).tocsr().tocoo()


#------------------------ Memory Efficient and Parallelized ----------------
import time
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import Pauli, SparsePauliOp
from joblib import Parallel, delayed
from memory_profiler import profile


# --- Helper Functions ---

def _int_conversion_numpy(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Efficiently converts a binary matrix to a 1D array of integers."""
    int_bitstring_matrix = bitstring_matrix.astype(np.uint64)
    n_bits = int_bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return int_bitstring_matrix.dot(powers_of_2)


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sorts and removes duplicates from a bitstring matrix."""
    int_reps = _int_conversion_numpy(bitstring_matrix)
    _, unique_indices = np.unique(int_reps, return_index=True)
    return bitstring_matrix[unique_indices]


def _process_pauli_chunk(
        pauli_chunk, bitstring_matrix, int_array_rows, batch_size
) -> tuple:
    """
    Worker function to process a chunk of off-diagonal Pauli terms.
    This function is designed to be called in parallel by joblib.
    """
    chunk_rows, chunk_cols, chunk_data = [], [], []
    d = bitstring_matrix.shape[0]

    for pauli, coeff in pauli_chunk:
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        for batch_start in range(0, d, batch_size):
            batch_end = min(batch_start + batch_size, d)
            int_array_batch = int_array_rows[batch_start:batch_end]
            bitstring_batch = bitstring_matrix[batch_start:batch_end]

            connected_ints_batch = int_array_batch ^ x_mask
            col_indices_batch = np.searchsorted(int_array_rows, connected_ints_batch)

            in_bounds_mask = col_indices_batch < d
            valid_mask = np.zeros_like(in_bounds_mask, dtype=bool)
            if np.any(in_bounds_mask):
                valid_in_bounds_indices = col_indices_batch[in_bounds_mask]
                valid_mask[in_bounds_mask] = (
                        int_array_rows[valid_in_bounds_indices] == connected_ints_batch[in_bounds_mask]
                )

            if not np.any(valid_mask):
                continue

            row_indices = (np.arange(len(valid_mask)) + batch_start)[valid_mask]
            final_col_indices = col_indices_batch[valid_mask]

            z_parities = np.sum(bitstring_batch[valid_mask] & pauli_z_rev, axis=1)
            phase = (-1) ** z_parities

            data_segment = coeff * phase * y_phase_factor

            chunk_rows.append(row_indices)
            chunk_cols.append(final_col_indices)
            chunk_data.append(data_segment)

    return chunk_rows, chunk_cols, chunk_data


@profile
def project_operator_to_subspace_memory_efficient_parallelized(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = 10000,
        n_jobs: int = -1,
        verbose: bool = False,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with high memory and time efficiency
    by using batching for memory and parallelization for speed.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    d, _ = bitstring_matrix.shape

    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    if verbose: print("Processing diagonal Pauli terms...")
    if diag_paulis:
        z_matrix = np.array([p.z[::-1] for p in diag_paulis], dtype=bool)
        parities = (-1) ** (bitstring_matrix @ z_matrix.T)
        diagonal_values = parities @ np.array(diag_coeffs)
        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    if verbose: print(
        f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel on {n_jobs if n_jobs != -1 else 'all'} cores...")

    # --- PARALLELIZATION IMPROVEMENT ---
    # Split the off-diagonal Paulis into chunks to be processed in parallel.
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    num_paulis = len(off_diag_terms)

    # Determine chunk size for parallel jobs (e.g., 10 Pauli terms per job)
    pauli_chunk_size = max(1, num_paulis // (abs(n_jobs) if n_jobs != 0 else 1) // 4)
    pauli_chunks = [
        off_diag_terms[i: i + pauli_chunk_size]
        for i in range(0, num_paulis, pauli_chunk_size)
    ]

    # Use joblib to run the worker function in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_pauli_chunk)(
            chunk, bitstring_matrix, int_array_rows, batch_size
        ) for chunk in pauli_chunks
    )

    # Collect results from all parallel jobs
    for chunk_rows, chunk_cols, chunk_data in results:
        if chunk_rows:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_data.extend(chunk_data)

    # --- Construct Final Sparse Matrix ---
    if not all_data:
        return coo_matrix((d, d), dtype="complex128")

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # Sum up duplicate entries
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype="complex128").tocsr().tocoo()



#Memoery and time efficient projection 2

import time
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import Pauli, SparsePauliOp
from joblib import Parallel, delayed
from memory_profiler import profile


# --- Helper Functions ---

def _int_conversion_numpy(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Efficiently converts a binary matrix to a 1D array of integers."""
    int_bitstring_matrix = bitstring_matrix.astype(np.uint64)
    n_bits = int_bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return int_bitstring_matrix.dot(powers_of_2)


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sorts and removes duplicates from a bitstring matrix."""
    int_reps = _int_conversion_numpy(bitstring_matrix)
    _, unique_indices = np.unique(int_reps, return_index=True)
    return bitstring_matrix[unique_indices]


def _process_pauli_chunk(
        pauli_chunk, bitstring_matrix, int_array_rows, batch_size
) -> tuple:
    """
    Worker function to process a chunk of off-diagonal Pauli terms.
    This function is designed to be called in parallel by joblib.
    """
    chunk_rows, chunk_cols, chunk_data = [], [], []
    d = bitstring_matrix.shape[0]

    for pauli, coeff in pauli_chunk:
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        for batch_start in range(0, d, batch_size):
            batch_end = min(batch_start + batch_size, d)
            int_array_batch = int_array_rows[batch_start:batch_end]
            bitstring_batch = bitstring_matrix[batch_start:batch_end]

            connected_ints_batch = int_array_batch ^ x_mask
            col_indices_batch = np.searchsorted(int_array_rows, connected_ints_batch)

            in_bounds_mask = col_indices_batch < d
            valid_mask = np.zeros_like(in_bounds_mask, dtype=bool)
            if np.any(in_bounds_mask):
                valid_in_bounds_indices = col_indices_batch[in_bounds_mask]
                valid_mask[in_bounds_mask] = (
                        int_array_rows[valid_in_bounds_indices] == connected_ints_batch[in_bounds_mask]
                )

            if not np.any(valid_mask):
                continue

            row_indices = (np.arange(len(valid_mask)) + batch_start)[valid_mask]
            final_col_indices = col_indices_batch[valid_mask]

            z_parities = np.sum(bitstring_batch[valid_mask] & pauli_z_rev, axis=1)
            phase = (-1) ** z_parities

            data_segment = coeff * phase * y_phase_factor

            chunk_rows.append(row_indices)
            chunk_cols.append(final_col_indices)
            chunk_data.append(data_segment)

    return chunk_rows, chunk_cols, chunk_data


# --- Memory and Time-Efficient Parallel Implementation ---

@profile
def project_operator_to_subspace_memory_efficient_parallelized_2(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = projection_batch_size,#10000,
        n_jobs: int = n_cores_for_parallel_projection,
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with maximal memory and time efficiency
    by using batching for all expensive operations and parallelization for speed.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    d, _ = bitstring_matrix.shape

    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- MAX MEMORY IMPROVEMENT: "Double Batching" for Diagonal Terms ---
    if verbose: print("Processing diagonal Pauli terms with double batching...")
    if diag_paulis:
        diagonal_values = np.zeros(d, dtype="complex128")
        diag_batch_size = 256  # Process 256 diagonal Paulis at a time

        # Outer loop: batches over Pauli strings
        for i in range(0, len(diag_paulis), diag_batch_size):
            batch_paulis = diag_paulis[i:i + diag_batch_size]
            batch_coeffs = np.array(diag_coeffs[i:i + diag_batch_size])
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)

            # Inner loop: batches over determinants
            for j in range(0, d, batch_size):
                bitstring_sub_batch = bitstring_matrix[j:j + batch_size]

                # The intermediate parities matrix is now tiny: (batch_size, diag_batch_size)
                parities_sub_batch = (-1) ** (bitstring_sub_batch @ z_matrix_batch.T)

                # Accumulate results into the correct slice of the final array
                diagonal_values[j:j + batch_size] += parities_sub_batch @ batch_coeffs

        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel...")

    # --- PARALLELIZATION FOR SPEED (Off-Diagonal Part) ---
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    num_paulis = len(off_diag_terms)

    # Determine chunk size for parallel jobs
    if n_jobs != 0:
        effective_jobs = abs(n_jobs) if n_jobs != -1 else 1  # Default to 1 if n_jobs is 0 or invalid
        pauli_chunk_size = max(1, num_paulis // (effective_jobs * 4))
    else:  # n_jobs=0 is not a valid joblib input, treat as 1
        pauli_chunk_size = num_paulis

    if pauli_chunk_size == 0: pauli_chunk_size = 1

    pauli_chunks = [
        off_diag_terms[i: i + pauli_chunk_size]
        for i in range(0, num_paulis, pauli_chunk_size)
    ]

    # Use joblib to run the worker function in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_pauli_chunk)(
            chunk, bitstring_matrix, int_array_rows, batch_size
        ) for chunk in pauli_chunks
    )

    # Collect results from all parallel jobs
    for chunk_rows, chunk_cols, chunk_data in results:
        if chunk_rows:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_data.extend(chunk_data)

    # --- Construct Final Sparse Matrix ---
    if not all_data:
        return coo_matrix((d, d), dtype="complex128")

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # Sum up duplicate entries
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype="complex128").tocsr().tocoo()













#----------- Mwmory Efficient and Parallelized with Threading ---------------------------

import time
import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import Pauli, SparsePauliOp
from joblib import Parallel, delayed
from memory_profiler import profile


# --- Helper Functions ---

def _int_conversion_numpy(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Efficiently converts a binary matrix to a 1D array of integers."""
    int_bitstring_matrix = bitstring_matrix.astype(np.uint64)
    n_bits = int_bitstring_matrix.shape[1]
    powers_of_2 = 2 ** np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
    return int_bitstring_matrix.dot(powers_of_2)


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Sorts and removes duplicates from a bitstring matrix."""
    int_reps = _int_conversion_numpy(bitstring_matrix)
    _, unique_indices = np.unique(int_reps, return_index=True)
    return bitstring_matrix[unique_indices]


def _process_pauli_chunk(
        pauli_chunk, bitstring_matrix, int_array_rows, batch_size
) -> tuple:
    """
    Worker function to process a chunk of off-diagonal Pauli terms.
    This function is designed to be called in parallel by joblib.
    """
    chunk_rows, chunk_cols, chunk_data = [], [], []
    d = bitstring_matrix.shape[0]

    for pauli, coeff in pauli_chunk:
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        for batch_start in range(0, d, batch_size):
            batch_end = min(batch_start + batch_size, d)
            int_array_batch = int_array_rows[batch_start:batch_end]
            bitstring_batch = bitstring_matrix[batch_start:batch_end]

            connected_ints_batch = int_array_batch ^ x_mask
            col_indices_batch = np.searchsorted(int_array_rows, connected_ints_batch)

            in_bounds_mask = col_indices_batch < d
            valid_mask = np.zeros_like(in_bounds_mask, dtype=bool)
            if np.any(in_bounds_mask):
                valid_in_bounds_indices = col_indices_batch[in_bounds_mask]
                valid_mask[in_bounds_mask] = (
                        int_array_rows[valid_in_bounds_indices] == connected_ints_batch[in_bounds_mask]
                )

            if not np.any(valid_mask):
                continue

            row_indices = (np.arange(len(valid_mask)) + batch_start)[valid_mask]
            final_col_indices = col_indices_batch[valid_mask]

            z_parities = np.sum(bitstring_batch[valid_mask] & pauli_z_rev, axis=1)
            phase = (-1) ** z_parities

            data_segment = coeff * phase * y_phase_factor

            chunk_rows.append(row_indices)
            chunk_cols.append(final_col_indices)
            chunk_data.append(data_segment)

    return chunk_rows, chunk_cols, chunk_data


# --- Memory and Time-Efficient Parallel Implementation ---

@profile
def project_operator_to_subspace_memory_efficient_parallelized_3(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = 10000,
        n_jobs: int = -1,
        parallel_backend: str = 'threading',
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with maximal memory and time efficiency
    by using batching for memory and parallelization for speed.

    Args:
        ...
        parallel_backend (str): The backend for joblib. 'threading' is recommended
            for memory efficiency as it avoids copying large arrays to workers.
        ...
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    d, _ = bitstring_matrix.shape

    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- "Double Batching" for Diagonal Terms ---
    if verbose: print("Processing diagonal Pauli terms with double batching...")
    if diag_paulis:
        diagonal_values = np.zeros(d, dtype="complex128")
        diag_batch_size = 256  # Process 256 diagonal Paulis at a time

        for i in range(0, len(diag_paulis), diag_batch_size):
            batch_paulis = diag_paulis[i:i + diag_batch_size]
            batch_coeffs = np.array(diag_coeffs[i:i + diag_batch_size])
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)

            for j in range(0, d, batch_size):
                bitstring_sub_batch = bitstring_matrix[j:j + batch_size]
                parities_sub_batch = (-1) ** (bitstring_sub_batch @ z_matrix_batch.T)
                diagonal_values[j:j + batch_size] += parities_sub_batch @ batch_coeffs

        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel...")

    # --- Parallelization with Threading Backend ---
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    num_paulis = len(off_diag_terms)

    if n_jobs != 0:
        effective_jobs = abs(n_jobs) if n_jobs != -1 else 1
        pauli_chunk_size = max(1, num_paulis // (effective_jobs * 4))
    else:
        pauli_chunk_size = num_paulis
    if pauli_chunk_size == 0: pauli_chunk_size = 1

    pauli_chunks = [
        off_diag_terms[i: i + pauli_chunk_size]
        for i in range(0, num_paulis, pauli_chunk_size)
    ]

    # Use joblib with the specified backend. 'threading' avoids copying the large
    # bitstring_matrix and int_array_rows to each worker, saving significant memory.
    results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_process_pauli_chunk)(
            chunk, bitstring_matrix, int_array_rows, batch_size
        ) for chunk in pauli_chunks
    )

    # Collect results from all parallel jobs
    for chunk_rows, chunk_cols, chunk_data in results:
        if chunk_rows:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_data.extend(chunk_data)

    # --- Construct Final Sparse Matrix ---
    if not all_data:
        return coo_matrix((d, d), dtype="complex128")

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # Sum up duplicate entries
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype="complex128").tocsr().tocoo()




#--------------- With only real Hamiltonian values

import numpy as np
from scipy.sparse import coo_matrix, spmatrix
from qiskit.quantum_info import SparsePauliOp
from joblib import Parallel, delayed


# Assuming _int_conversion_numpy and _process_pauli_chunk are defined elsewhere

def project_operator_to_subspace_memory_efficient_parallelized_4(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = 10000,
        n_jobs: int = -1,
        parallel_backend: str = 'threading',
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with maximal memory and time efficiency.
    This version is optimized for Hamiltonians with purely real coefficients.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    # --- OPTIMIZATION: Determine dtype based on Hamiltonian coefficients ---
    # np.iscomplexobj is a fast way to check if the array contains complex numbers.
    if np.iscomplexobj(hamiltonian.coeffs):
        if verbose: print("Hamiltonian has complex coefficients. Using 'complex128' dtype.")
        dtype = np.complex128
    else:
        if verbose: print("Hamiltonian has real coefficients. Using 'float64' dtype for optimization.")
        dtype = np.float64
    # -----------------------------------------------------------------------

    d, _ = bitstring_matrix.shape

    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- "Double Batching" for Diagonal Terms ---
    if verbose: print("Processing diagonal Pauli terms with double batching...")
    if diag_paulis:
        # --- MODIFIED LINE ---
        diagonal_values = np.zeros(d, dtype=dtype)
        # -------------------
        diag_batch_size = 256

        for i in range(0, len(diag_paulis), diag_batch_size):
            batch_paulis = diag_paulis[i:i + diag_batch_size]
            batch_coeffs = np.array(diag_coeffs[i:i + diag_batch_size], dtype=dtype)
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)

            for j in range(0, d, batch_size):
                bitstring_sub_batch = bitstring_matrix[j:j + batch_size]
                parities_sub_batch = (-1) ** (bitstring_sub_batch @ z_matrix_batch.T)
                diagonal_values[j:j + batch_size] += parities_sub_batch @ batch_coeffs

        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel...")

    # --- Parallelization for Off-Diagonal Terms (no changes needed here) ---
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    num_paulis = len(off_diag_terms)

    if n_jobs != 0:
        effective_jobs = abs(n_jobs) if n_jobs != -1 else 1
        pauli_chunk_size = max(1, num_paulis // (effective_jobs * 4))
    else:
        pauli_chunk_size = num_paulis
    if pauli_chunk_size == 0: pauli_chunk_size = 1

    pauli_chunks = [
        off_diag_terms[i: i + pauli_chunk_size]
        for i in range(0, num_paulis, pauli_chunk_size)
    ]

    # Note: Ensure that _process_pauli_chunk also uses the appropriate dtype
    # when calculating its 'chunk_data'. Since the input coeffs are real,
    # the output should naturally be real.
    results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_process_pauli_chunk)(
            chunk, bitstring_matrix, int_array_rows, batch_size
        ) for chunk in pauli_chunks
    )

    for chunk_rows, chunk_cols, chunk_data in results:
        if chunk_rows:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_data.extend(chunk_data)

    # --- Construct Final Sparse Matrix ---
    if not all_data:
        # --- MODIFIED LINE ---
        return coo_matrix((d, d), dtype=dtype)
        # -------------------

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # --- MODIFIED LINE ---
    # Construct the final matrix using the determined dtype
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype=dtype).tocsr().tocoo()
    # -------------------

@profile
def project_operator_to_subspace_memory_efficient_parallelized_4_real(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = projection_batch_size,
        n_jobs: int = n_cores_for_parallel_projection,# -1,
        parallel_backend: str = 'threading',
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator onto a Hilbert subspace with maximal memory and time efficiency.
    This version is optimized for Hamiltonians with purely real coefficients.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings must have length < 64.")

    # --- OPTIMIZATION: Determine dtype based on Hamiltonian coefficients ---
    # np.iscomplexobj is a fast way to check if the array contains complex numbers.
    if np.iscomplexobj(hamiltonian.coeffs.real):
        if verbose: print("Hamiltonian has complex coefficients. Using 'complex128' dtype.")
        dtype = np.complex128
    else:
        if verbose: print("Hamiltonian has real coefficients. Using 'float64' dtype for optimization.")
        dtype = np.float64
    # -----------------------------------------------------------------------

    d, _ = bitstring_matrix.shape

    if verbose: print("Pre-computing integer representations...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    all_rows, all_cols, all_data = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli)
            off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli)
            diag_coeffs.append(coeff)

    # --- "Double Batching" for Diagonal Terms ---
    if verbose: print("Processing diagonal Pauli terms with double batching...")
    if diag_paulis:
        # --- MODIFIED LINE ---
        diagonal_values = np.zeros(d, dtype=dtype)
        # -------------------
        diag_batch_size = 256

        for i in range(0, len(diag_paulis), diag_batch_size):
            batch_paulis = diag_paulis[i:i + diag_batch_size]
            batch_coeffs = np.array(diag_coeffs[i:i + diag_batch_size], dtype=dtype)
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)

            for j in range(0, d, batch_size):
                bitstring_sub_batch = bitstring_matrix[j:j + batch_size]
                parities_sub_batch = (-1) ** (bitstring_sub_batch @ z_matrix_batch.T)
                diagonal_values[j:j + batch_size] += parities_sub_batch @ batch_coeffs

        all_rows.append(np.arange(d))
        all_cols.append(np.arange(d))
        all_data.append(diagonal_values)

    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel...")

    # --- Parallelization for Off-Diagonal Terms (no changes needed here) ---
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    num_paulis = len(off_diag_terms)

    if n_jobs != 0:
        effective_jobs = abs(n_jobs) if n_jobs != -1 else 1
        pauli_chunk_size = max(1, num_paulis // (effective_jobs * 4))
    else:
        pauli_chunk_size = num_paulis
    if pauli_chunk_size == 0: pauli_chunk_size = 1

    pauli_chunks = [
        off_diag_terms[i: i + pauli_chunk_size]
        for i in range(0, num_paulis, pauli_chunk_size)
    ]

    # Note: Ensure that _process_pauli_chunk also uses the appropriate dtype
    # when calculating its 'chunk_data'. Since the input coeffs are real,
    # the output should naturally be real.
    results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_process_pauli_chunk)(
            chunk, bitstring_matrix, int_array_rows, batch_size
        ) for chunk in pauli_chunks
    )

    for chunk_rows, chunk_cols, chunk_data in results:
        if chunk_rows:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_data.extend(chunk_data)

    # --- Construct Final Sparse Matrix ---
    if not all_data:
        # --- MODIFIED LINE ---
        return coo_matrix((d, d), dtype=dtype)
        # -------------------

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)

    # --- MODIFIED LINE ---
    # Construct the final matrix using the determined dtype
    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype=dtype).tocsr().tocoo()
    # -------------------









# latest using parallel and real hamiltonian

# (Keep the helper functions _int_conversion_numpy and sort_and_remove_duplicates as they are)

# --- OPTIMIZED AND CORRECTED HELPER FUNCTION for THREADING ---

# --- 1. REPLACE your helper function with this one ---

def _process_pauli_chunk_real_threaded(
        pauli_chunk: list[tuple[Pauli, float]],
        bitstring_matrix: np.ndarray,
        int_array_rows: np.ndarray,
        int_map: dict,
        coeff_dtype: type,
        index_dtype: type,
) -> tuple[list, list, list]:
    """
    Worker function that is GUARANTEED to return flat 1D lists for rows, cols, and data.
    """
    # These will be flat lists
    chunk_rows, chunk_cols, chunk_data = [], [], []

    for pauli, coeff in pauli_chunk:
        pauli_x_rev, pauli_z_rev = pauli.x[::-1], pauli.z[::-1]
        x_mask = _int_conversion_numpy(pauli_x_rev.reshape(1, -1))[0]

        connected_ints = np.bitwise_xor(int_array_rows, x_mask)

        target_cols = [int_map.get(ci, -1) for ci in connected_ints]
        valid_indices = [i for i, col in enumerate(target_cols) if col != -1]

        if not valid_indices:
            continue

        # valid_rows and valid_cols are 1D NumPy arrays
        valid_rows = np.array(valid_indices, dtype=index_dtype)
        valid_cols = np.array([target_cols[i] for i in valid_indices], dtype=index_dtype)

        z_parities = np.sum(bitstring_matrix[valid_indices] & pauli_z_rev, axis=1)
        phase = (-1) ** z_parities
        y_phase_factor = (1j) ** np.sum(pauli_x_rev & pauli_z_rev)

        data_segment = coeff * phase * y_phase_factor

        # .extend() adds the elements of the arrays to the flat lists
        chunk_rows.extend(valid_rows.tolist())
        chunk_cols.extend(valid_cols.tolist())
        chunk_data.extend([data_segment.real] * len(valid_rows))

    return chunk_rows, chunk_cols, chunk_data

# --- REWRITTEN MAIN FUNCTION WITH MEMORY CONTROLS ---
# --- 2. REPLACE your main function with this one ---

@profile
def project_operator_real_optimized(
        bitstring_matrix: np.ndarray,
        hamiltonian: SparsePauliOp,
        *,
        batch_size: int = 20000,
        n_jobs: int = -1,
        verbose: bool = True,
) -> spmatrix:
    """
    Projects a Pauli operator, forcing a real-valued result and using memory-efficient
    dtypes. Uses a THREADING backend for parallelization to minimize memory.
    """
    # --- (Setup part is unchanged) ---
    if bitstring_matrix.shape[1] > 63: raise ValueError("Bitstrings must have length < 64.")
    d, _ = bitstring_matrix.shape
    coeff_dtype = np.float64
    if d < np.iinfo(np.int32).max:
        index_dtype = np.int32
    else:
        index_dtype = np.int64
    if verbose: print(f"Using '{np.dtype(index_dtype).name}' for matrix indices.")
    if np.iscomplexobj(hamiltonian.coeffs) and verbose:
        print("WARNING: Input Hamiltonian has complex coefficients...")
    int_array_rows = _int_conversion_numpy(bitstring_matrix)

    # This will be a list of final NumPy arrays to concatenate
    final_row_arrays, final_col_arrays, final_data_arrays = [], [], []

    diag_paulis, diag_coeffs, off_diag_paulis, off_diag_coeffs = [], [], [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            off_diag_paulis.append(pauli); off_diag_coeffs.append(coeff)
        else:
            diag_paulis.append(pauli); diag_coeffs.append(coeff)

    # --- (Diagonal processing is unchanged) ---
    if verbose: print("Processing diagonal Pauli terms with double batching...")
    if diag_paulis:
        diagonal_values = np.zeros(d, dtype=coeff_dtype)
        diag_batch_size = 256
        for i in range(0, len(diag_paulis), diag_batch_size):
            batch_paulis = diag_paulis[i:i + diag_batch_size]
            batch_coeffs_real = np.array(diag_coeffs[i:i + diag_batch_size]).real
            z_matrix_batch = np.array([p.z[::-1] for p in batch_paulis], dtype=bool)
            for j in range(0, d, batch_size):
                bitstring_sub_batch = bitstring_matrix[j:j + batch_size]
                parities_sub_batch = (-1) ** (bitstring_sub_batch @ z_matrix_batch.T)
                diagonal_values[j:j + batch_size] += parities_sub_batch @ batch_coeffs_real
        final_row_arrays.append(np.arange(d, dtype=index_dtype))
        final_col_arrays.append(np.arange(d, dtype=index_dtype))
        final_data_arrays.append(diagonal_values)

    if verbose: print(f"Processing {len(off_diag_paulis)} off-diagonal terms in parallel...")
    int_map = {val: i for i, val in enumerate(int_array_rows)}
    off_diag_terms = list(zip(off_diag_paulis, off_diag_coeffs))
    # ... (pauli_chunks creation is unchanged) ...
    num_paulis = len(off_diag_terms)
    if n_jobs != 0:
        effective_jobs = abs(n_jobs) if n_jobs != -1 else 1
        pauli_chunk_size = max(1, num_paulis // (effective_jobs * 4))
    else:
        pauli_chunk_size = num_paulis
    if pauli_chunk_size == 0: pauli_chunk_size = 1
    pauli_chunks = [off_diag_terms[i:i + pauli_chunk_size] for i in range(0, num_paulis, pauli_chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_pauli_chunk_real_threaded)(
            chunk, bitstring_matrix, int_array_rows, int_map,
            coeff_dtype, index_dtype
        ) for chunk in pauli_chunks
    )

    # --- CORRECTED RESULT COLLECTION LOGIC ---
    # Create one large flat list for each of rows, cols, and data
    offdiag_rows_list, offdiag_cols_list, offdiag_data_list = [], [], []
    for r_chunk, c_chunk, d_chunk in results:
        offdiag_rows_list.extend(r_chunk)
        offdiag_cols_list.extend(c_chunk)
        offdiag_data_list.extend(d_chunk)

    # If any off-diagonal elements were found, convert the flat lists to NumPy arrays
    if offdiag_data_list:
        final_row_arrays.append(np.array(offdiag_rows_list, dtype=index_dtype))
        final_col_arrays.append(np.array(offdiag_cols_list, dtype=index_dtype))
        final_data_arrays.append(np.array(offdiag_data_list, dtype=coeff_dtype))
    # --- END CORRECTION ---

    if not final_data_arrays:
        return coo_matrix((d, d), dtype=coeff_dtype)

    final_rows = np.concatenate(final_row_arrays)
    final_cols = np.concatenate(final_col_arrays)
    final_data = np.concatenate(final_data_arrays)

    return coo_matrix((final_data, (final_rows, final_cols)), shape=(d, d), dtype=coeff_dtype).tocsr().tocoo()


#----------- Projection using numba








from scipy.sparse.linalg import eigsh

def subspace_diagonalizer(det_array):
    print('Hamiltonian Projection')
    start_time = time.time()
    # H_op_sub = project_operator_to_subspace_efficient_v2(
    #     bitstring_matrix=det_array,
    #     hamiltonian=jw_mapped_hamiltonian
    # # )
    if parallelized_projection == 'no':
        H_op_sub = project_operator_to_subspace_memory_efficient2(
            bitstring_matrix=det_array,
            hamiltonian=jw_mapped_hamiltonian
        )
    if parallelized_projection == 'yes':
        H_op_sub = project_operator_to_subspace_memory_efficient_parallelized_4_real(
            bitstring_matrix=det_array,
            hamiltonian=jw_mapped_hamiltonian
        )


    #np.save('N2_projected_hamiltonian_0.75AA.npy',H_op_sub)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('New projection elapsed_time',elapsed_time)

    # Ensure it's CSR for efficient matvecs
    H_op_sub = H_op_sub.tocsr()

    # Quick hermiticity check and symmetrize if necessary
    skew = H_op_sub - H_op_sub.getH()
    max_skew = 0.0 if skew.nnz == 0 else abs(skew.data).max()
    if max_skew > 1e-10:
        print(f"Warning: H not perfectly Hermitian (max off = {max_skew:.3e}). Symmetrizing.")
        H_op_sub = 0.5 * (H_op_sub + H_op_sub.getH())

    if np.iscomplexobj(H_op_sub.data):
        print("Original projected matrix has complex entries. Taking the real part.")
        H_op_sub = H_op_sub.real

    start_time = time.time()
    print('Diagonalization starts (sparse eigsh)')
    d_eigval, d_eigvec = eigsh(H_op_sub, k=1, which='SA', tol=1e-8, maxiter=200)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Diagonalization elapsed_time',elapsed_time)

    final_energy = d_eigval[0] + nuclear_repulsion_energy_qiskit + frozen_energy_shift
    return final_energy, d_eigvec[:, 0]   # return complex vector if present




#------------------------------- matvec implementation -----------------------------------------------------------------



from joblib import Parallel, delayed, cpu_count
import numpy as np
import time
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import time
from numba import njit, prange, int64, uint64, float64
from qiskit.quantum_info import SparsePauliOp

from numba import config, threading_layer

# Print what threading layer Numba picked
print("Threading Layer used:", threading_layer())
import numpy as np
from numba import njit, prange, int64, float64, int32
import time
import os
import psutil

# Force Numba to use all cores
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count())


# =============================================================================
# PART 1: JIT KERNELS (Hash Map & MatVec)
# =============================================================================


#
# @njit(fastmath=True)
# def build_lookup_table(det_list, size_power):
#     """
#     Allocates a massive hash map (~50GB for 80M items) for O(1) access.
#     Assumes det_list is unique (which is true if it's a valid basis).
#     """
#     table_size = 1 << size_power
#     mask = table_size - 1
#     # Initialize with -1
#     keys = np.full(table_size, -1, dtype=np.int64)
#     values = np.full(table_size, -1, dtype=np.int32)
#
#     for idx in range(len(det_list)):
#         det = det_list[idx]
#         h = (det * 0x9e3779b97f4a7c15) & mask
#         while keys[h] != -1:
#             h = (h + 1) & mask
#         keys[h] = det
#         values[h] = idx
#     return keys, values, mask
#
#
# @njit(inline='always')
# def lookup(det, keys, values, mask):
#     h = (det * 0x9e3779b97f4a7c15) & mask
#     while True:
#         k = keys[h]
#         if k == -1: return -1
#         if k == det: return values[h]
#         h = (h + 1) & mask
#
#
# @njit(parallel=True, fastmath=True)
# def _turbo_grouped_matvec(psi_in, psi_out, det_list,
#                           unique_x_masks,
#                           z_group_ptr, z_masks_flat, coeffs_flat,
#                           lookup_keys, lookup_vals, lookup_mask,
#                           diag_potentials):
#     """
#     Parallel Sigma Build.
#     """
#     n_dets = len(det_list)
#     n_x_groups = len(unique_x_masks)
#
#     for i in prange(n_dets):
#         current_det = det_list[i]
#
#         # 1. Diagonal Part
#         val = diag_potentials[i] * psi_in[i]
#
#         # 2. Off-Diagonal Part
#         for g in range(n_x_groups):
#             x_mask = unique_x_masks[g]
#             target_det = current_det ^ x_mask
#
#             target_idx = lookup(target_det, lookup_keys, lookup_vals, lookup_mask)
#
#             if target_idx != -1:
#                 start = z_group_ptr[g]
#                 end = z_group_ptr[g + 1]
#                 term_sum = 0.0
#                 target_amp = psi_in[target_idx]
#
#                 # Interaction Loop
#                 for k in range(start, end):
#                     z_mask = z_masks_flat[k]
#                     c = coeffs_flat[k]
#
#                     # Inlined Parity
#                     temp = current_det & z_mask
#                     z_parity = 0
#                     while temp > 0:
#                         temp &= (temp - 1)
#                         z_parity += 1
#
#                     sign = 1.0 if (z_parity & 1) == 0 else -1.0
#                     term_sum += c * sign
#
#                 val += term_sum * target_amp
#
#         psi_out[i] = val
#








if matrix_diagonalization_type == 'dci':

    @njit(fastmath=True)
    def build_lookup_table(det_list, size_power):
        """
        Allocates a massive hash map (~50GB for 80M items) for O(1) access.
        Assumes det_list is unique (which is true if it's a valid basis).
        """
        table_size = 1 << size_power
        mask = table_size - 1
        # Initialize with -1
        keys = np.full(table_size, -1, dtype=np.int64)
        values = np.full(table_size, -1, dtype=np.int32)

        for idx in range(len(det_list)):
            det = det_list[idx]
            h = (det * 0x9e3779b97f4a7c15) & mask
            while keys[h] != -1:
                h = (h + 1) & mask
            keys[h] = det
            values[h] = idx
        return keys, values, mask


    @njit(inline='always')
    def lookup(det, keys, values, mask):
        h = (det * 0x9e3779b97f4a7c15) & mask
        while True:
            k = keys[h]
            if k == -1: return -1
            if k == det: return values[h]
            h = (h + 1) & mask


    @njit(parallel=True, fastmath=True)
    def _turbo_grouped_matvec(psi_in, psi_out, det_list,
                              unique_x_masks,
                              z_group_ptr, z_masks_flat, coeffs_flat,
                              lookup_keys, lookup_vals, lookup_mask,
                              diag_potentials):
        """
        Parallel Sigma Build.
        """
        n_dets = len(det_list)
        n_x_groups = len(unique_x_masks)

        for i in prange(n_dets):
            current_det = det_list[i]

            # 1. Diagonal Part
            val = diag_potentials[i] * psi_in[i]

            # 2. Off-Diagonal Part
            for g in range(n_x_groups):
                x_mask = unique_x_masks[g]
                target_det = current_det ^ x_mask

                target_idx = lookup(target_det, lookup_keys, lookup_vals, lookup_mask)

                if target_idx != -1:
                    start = z_group_ptr[g]
                    end = z_group_ptr[g + 1]
                    term_sum = 0.0
                    target_amp = psi_in[target_idx]

                    # Interaction Loop
                    for k in range(start, end):
                        z_mask = z_masks_flat[k]
                        c = coeffs_flat[k]

                        # Inlined Parity
                        temp = current_det & z_mask
                        z_parity = 0
                        while temp > 0:
                            temp &= (temp - 1)
                            z_parity += 1

                        sign = 1.0 if (z_parity & 1) == 0 else -1.0
                        term_sum += c * sign

                    val += term_sum * target_amp

            psi_out[i] = val


    # =============================================================================
    # PART 2: HAMILTONIAN CLASS
    # =============================================================================
    # --- ADD THIS NEW JIT KERNEL AT THE TOP LEVEL ---
    @njit(fastmath=True, parallel=True)
    def _fast_add_diagonal(diag_val, det_list, z_mask, c_val):
        """
        JIT-compiled kernel to update diagonal potentials instantly.
        Parallelized for maximum speed.
        """
        n = len(det_list)
        for i in prange(n):
            det = det_list[i]
            # Popcount parity check
            temp = det & z_mask
            p = 0
            while temp > 0:
                temp &= (temp - 1)
                p += 1

            sign = 1.0 if (p & 1) == 0 else -1.0
            diag_val[i] += c_val * sign


    # --- UPDATED HAMILTONIAN CLASS ---
    class SparseHamiltonian:
        def __init__(self, basis_strings, qubit_op):
            print("--- [INIT] Initializing Hamiltonian (Optimized) ---")
            self.t_start = time.time()

            # 1. Store Basis (Optimized Load)
            raw_basis = np.array(basis_strings)
            if raw_basis.dtype.kind in {'U', 'S', 'O'}:
                try:
                    # Check the first element. If it's a binary string '101', convert.
                    sample = str(raw_basis[0])
                    if set(sample).issubset({'0', '1'}):
                        # Fast List Comprehension for conversion
                        self.det_list = np.array([int(s, 2) for s in raw_basis], dtype=np.int64)
                    else:
                        self.det_list = raw_basis.astype(np.int64)
                except:
                    self.det_list = raw_basis.astype(np.int64)
            else:
                self.det_list = raw_basis.astype(np.int64)

            self.det_list = np.sort(self.det_list)
            self.n_dets = len(self.det_list)
            print(f"    [DATA] {self.n_dets:,} determinants.")

            # 2. Dynamic RAM Allocation
            # Target: 4x slots for speed (Load factor 0.25)
            target_size = self.n_dets * 4
            self.size_power = int(np.ceil(np.log2(target_size)))
            self.size_power = max(self.size_power, 16)

            mem_mb = (1 << self.size_power) * 12 / (1024 ** 2)
            print(f"    [RAM] Allocating Hash Map (2^{self.size_power} slots, ~{mem_mb:.2f} MB)...")

            self.l_keys, self.l_vals, self.l_mask = build_lookup_table(self.det_list, self.size_power)

            # 3. Process Terms
            self._preprocess_hamiltonian(qubit_op)
            print(f"    [READY] Build complete in {time.time() - self.t_start:.2f}s")

        def _preprocess_hamiltonian(self, qubit_op):
            if hasattr(qubit_op, 'to_list'):
                pauli_list = qubit_op.to_list()
            else:
                pauli_list = qubit_op

            diag_val = np.zeros(self.n_dets, dtype=np.float64)
            groups = {}

            print(f"    [PROC] Processing {len(pauli_list)} Hamiltonian terms...")

            for p_str, coeff in pauli_list:
                if isinstance(coeff, complex):
                    if abs(coeff.imag) > 1e-12: pass

                x_mask = 0;
                z_mask = 0;
                y_count = 0
                for q, char in enumerate(reversed(p_str)):
                    if char == 'X':
                        x_mask |= (1 << q)
                    elif char == 'Z':
                        z_mask |= (1 << q)
                    elif char == 'Y':
                        x_mask |= (1 << q); z_mask |= (1 << q); y_count += 1

                c_val = np.real(coeff)
                if (y_count & 1):
                    c_val = np.real(coeff * (1j) ** y_count)
                else:
                    c_val = c_val * ((-1) ** (y_count >> 1))

                if abs(c_val) < 1e-14: continue

                if x_mask == 0:
                    # !!! FIX IS HERE: CALL JIT KERNEL !!!
                    _fast_add_diagonal(diag_val, self.det_list, z_mask, c_val)
                else:
                    if x_mask not in groups: groups[x_mask] = []
                    groups[x_mask].append((z_mask, c_val))

            self.diag_potentials = diag_val

            unique_x = []
            z_ptr = [0];
            z_masks = [];
            coeffs = []
            for x_m, terms in groups.items():
                unique_x.append(x_m)
                for z, c in terms: z_masks.append(z); coeffs.append(c)
                z_ptr.append(len(z_masks))

            self.unique_x_masks = np.array(unique_x, dtype=np.int64)
            self.z_group_ptr = np.array(z_ptr, dtype=np.int32)
            self.z_masks_flat = np.array(z_masks, dtype=np.int64)
            self.coeffs_flat = np.array(coeffs, dtype=np.float64)

        def matvec(self, v):
            v_in = np.ascontiguousarray(v, dtype=np.float64)
            v_out = np.zeros_like(v_in)
            _turbo_grouped_matvec(
                v_in, v_out, self.det_list,
                self.unique_x_masks,
                self.z_group_ptr, self.z_masks_flat, self.coeffs_flat,
                self.l_keys, self.l_vals, self.l_mask,
                self.diag_potentials
            )
            return v_out

        @property
        def diagonal(self):
            return self.diag_potentials



    # =============================================================================
    # PART 3: PURE DAVIDSON SOLVER
    # =============================================================================

    def davidson_diagonalization(hamiltonian, n_roots=1, tol=1e-5, max_iter=50, subspace_limit=40):
        """
        Generalized Davidson Algorithm with Diagonal Preconditioning.
        """
        n_dets = hamiltonian.n_dets
        t_start = time.time()
        print(f"\n--- [START] Davidson Solver (Preconditioned) ---")

        # 1. Initial Guess (Lowest Diagonal Elements)
        # Since diagonal is likely roughly sorted by energy,
        # we pick the indices of the lowest diagonal values.
        diag = hamiltonian.diagonal
        # Optimization: Use argpartition for O(N) selection instead of O(N log N) sort
        guess_indices = np.argpartition(diag, n_roots)[:n_roots]

        V = []  # Subspace Vectors
        AV = []  # H * V

        # Init first vectors
        for idx in guess_indices:
            v = np.zeros(n_dets, dtype=np.float64)
            v[idx] = 1.0
            V.append(v)
            AV.append(hamiltonian.matvec(v))

        print(f"    [ITER 0] Subspace built. Ref E: {diag[guess_indices[0]]  + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.6f}")

        for it in range(max_iter):
            dim = len(V)

            # 2. Build Subspace Matrix T (dim x dim)
            T = np.zeros((dim, dim), dtype=np.float64)
            for i in range(dim):
                for j in range(i, dim):
                    val = np.dot(V[i], AV[j])
                    T[i, j] = val
                    T[j, i] = val

            # 3. Diagonalize Subspace
            eigvals_sub, eigvecs_sub = np.linalg.eigh(T)
            current_e = eigvals_sub[0]

            # 4. Compute Residual & Convergence
            # Ritz vector: C_0*V_0 + ... + C_n*V_n
            c = eigvecs_sub[:, 0]
            ritz_v = np.zeros(n_dets, dtype=np.float64)
            ritz_Av = np.zeros(n_dets, dtype=np.float64)

            for i in range(dim):
                ritz_v += c[i] * V[i]
                ritz_Av += c[i] * AV[i]

            residual = ritz_Av - current_e * ritz_v
            res_norm = np.linalg.norm(residual)

            elapsed = time.time() - t_start
            print(f"    [ITER {it + 1}] Dim: {dim} | E: {current_e + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.8f} | Res: {res_norm:.2e} | Time: {elapsed:.1f}s")

            if res_norm < tol:
                print(f"    [DONE] Converged in {it + 1} iterations.")
                return current_e, ritz_v

            # 5. Preconditioning (Davidson Correction)
            # q = r / (D - E)
            precond_denom = diag - current_e
            # Numerical stability: avoid division by very small numbers
            precond_denom = np.where(np.abs(precond_denom) < 1e-5, 1e-5, precond_denom)

            new_v = residual / precond_denom

            # 6. Gram-Schmidt Orthogonalization
            for v_old in V:
                proj = np.dot(v_old, new_v)
                new_v -= proj * v_old

            norm_new = np.linalg.norm(new_v)

            # 7. Append to Subspace
            if norm_new > 1e-9:
                new_v /= norm_new
                V.append(new_v)
                # Parallel Sigma Build for the new vector
                AV.append(hamiltonian.matvec(new_v))
            else:
                print("    [WARN] Subspace collapse (Linear Dependency). Stopping.")
                break

            # 8. Restart Logic (Control RAM)
            if len(V) >= subspace_limit:
                print("    [INFO] Subspace limit reached. Restarting...")
                # Keep only the current best Ritz vector
                V = [ritz_v]
                AV = [ritz_Av]

        return current_e, ritz_v


    # =============================================================================
    # WRAPPER FUNCTION
    # =============================================================================

    def run_ci_davidson(basis_strings, qubit_op, nuclear_repulsion, frozen_shift):
        """
        Entry point.
        basis_strings: Must be sorted list of ints or strings.
        """
        # Initialize optimized Hamiltonian
        H = SparseHamiltonian(basis_strings, qubit_op)

        # Solve
        energy_elec, ci_vector = davidson_diagonalization(H, n_roots=1, tol=1e-5)

        total_energy = energy_elec + nuclear_repulsion + frozen_shift
        return total_energy, ci_vector















if matrix_diagonalization_type == 'csr':
    import numpy as np
    from numba import njit, prange, int64, float64, int32
    import scipy.sparse as sp
    import time
    import os

    # Force Numba to use all cores
    os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count())


    # =============================================================================
    # PART 1: JIT KERNELS (MUST BE AT TOP LEVEL)
    # =============================================================================

    @njit(fastmath=True, parallel=True)
    def _fast_add_diagonal(diag_val, det_list, z_mask, c_val):
        """JIT kernel to update diagonal potentials."""
        n = len(det_list)
        for i in prange(n):
            det = det_list[i]
            temp = det & z_mask
            p = 0
            while temp > 0:
                temp &= (temp - 1)
                p += 1
            sign = 1.0 if (p & 1) == 0 else -1.0
            diag_val[i] += c_val * sign


    @njit(parallel=True, fastmath=True)
    def count_nnz_per_row(det_list, unique_x_masks, lookup_keys, lookup_vals, lookup_mask):
        """PASS 1: Count connections."""
        n_dets = len(det_list)
        row_counts = np.zeros(n_dets, dtype=np.int32)
        n_groups = len(unique_x_masks)

        for i in prange(n_dets):
            current_det = det_list[i]
            count = 1  # Diagonal always exists

            for g in range(n_groups):
                x_mask = unique_x_masks[g]
                target_det = current_det ^ x_mask

                # Lookup
                h = (target_det * 0x9e3779b97f4a7c15) & lookup_mask
                while True:
                    k = lookup_keys[h]
                    if k == -1: break
                    if k == target_det:
                        count += 1
                        break
                    h = (h + 1) & lookup_mask

            row_counts[i] = count
        return row_counts


    @njit(parallel=True, fastmath=True)
    def fill_csr_data(det_list, unique_x_masks, z_group_ptr, z_masks_flat, coeffs_flat,
                      lookup_keys, lookup_vals, lookup_mask, diag_potentials,
                      indptr, indices, data):
        """PASS 2: Fill CSR arrays."""
        n_dets = len(det_list)
        n_groups = len(unique_x_masks)

        for i in prange(n_dets):
            current_det = det_list[i]
            row_start = indptr[i]
            offset = 0

            # 1. Diagonal
            indices[row_start + offset] = i
            data[row_start + offset] = diag_potentials[i]
            offset += 1

            # 2. Off-Diagonal
            for g in range(n_groups):
                x_mask = unique_x_masks[g]
                target_det = current_det ^ x_mask

                h = (target_det * 0x9e3779b97f4a7c15) & lookup_mask
                target_idx = -1
                while True:
                    k = lookup_keys[h]
                    if k == -1: break
                    if k == target_det:
                        target_idx = lookup_vals[h]
                        break
                    h = (h + 1) & lookup_mask

                if target_idx != -1:
                    term_sum = 0.0
                    start = z_group_ptr[g]
                    end = z_group_ptr[g + 1]

                    for k in range(start, end):
                        z_mask = z_masks_flat[k]
                        c = coeffs_flat[k]
                        temp = current_det & z_mask
                        p = 0
                        while temp > 0: temp &= (temp - 1); p += 1
                        sign = 1.0 if (p & 1) == 0 else -1.0
                        term_sum += c * sign

                    indices[row_start + offset] = target_idx
                    data[row_start + offset] = term_sum
                    offset += 1


    @njit(fastmath=True)
    def _build_lookup(det_list, size_power):
        table_size = 1 << size_power
        mask = table_size - 1
        keys = np.full(table_size, -1, dtype=np.int64)
        values = np.full(table_size, -1, dtype=np.int32)
        for idx in range(len(det_list)):
            det = det_list[idx]
            h = (det * 0x9e3779b97f4a7c15) & mask
            while keys[h] != -1: h = (h + 1) & mask
            keys[h] = det
            values[h] = idx
        return keys, values, mask


    # =============================================================================
    # PART 2: HAMILTONIAN CLASS (FIXED)
    # =============================================================================

    class SparseHamiltonian:
        def __init__(self, basis_strings, qubit_op):
            print("--- [INIT] Pre-Building Sparse Matrix (CSR) ---")
            t0 = time.time()

            # 1. Basis Setup
            raw_basis = np.array(basis_strings)
            if raw_basis.dtype.kind in {'U', 'S', 'O'}:
                try:
                    sample = str(raw_basis[0])
                    if set(sample).issubset({'0', '1'}):
                        self.det_list = np.array([int(s, 2) for s in raw_basis], dtype=np.int64)
                    else:
                        self.det_list = raw_basis.astype(np.int64)
                except:
                    self.det_list = raw_basis.astype(np.int64)
            else:
                self.det_list = raw_basis.astype(np.int64)

            self.det_list = np.sort(self.det_list)
            self.n_dets = len(self.det_list)

            # 2. Hash Map Setup
            target_size = self.n_dets * 4
            self.size_power = int(np.ceil(np.log2(target_size)))
            self.size_power = max(self.size_power, 16)

            print(f"    [RAM] Allocating Hash Map (2^{self.size_power} slots)...")
            self.l_keys, self.l_vals, self.l_mask = _build_lookup(self.det_list, self.size_power)

            # 3. Process Operator Terms (This was missing logic before)
            self._preprocess_hamiltonian(qubit_op)

            # 4. BUILD CSR MATRIX
            print(f"    [CSR] Pass 1: Counting Non-Zero elements (Parallel)...")
            row_counts = count_nnz_per_row(
                self.det_list, self.unique_x_masks,
                self.l_keys, self.l_vals, self.l_mask
            )

            total_nnz = np.sum(row_counts)
            ram_est = (total_nnz * 12 + self.n_dets * 4) / (1024 ** 3)
            print(f"    [CSR] Matrix Size: {total_nnz:,} entries (~{ram_est:.2f} GB)")

            indptr = np.zeros(self.n_dets + 1, dtype=np.int32)
            np.cumsum(row_counts, out=indptr[1:])

            print(f"    [CSR] Pass 2: Filling Matrix Data (Parallel)...")
            indices = np.empty(total_nnz, dtype=np.int32)
            data = np.empty(total_nnz, dtype=np.float64)

            fill_csr_data(
                self.det_list, self.unique_x_masks,
                self.z_group_ptr, self.z_masks_flat, self.coeffs_flat,
                self.l_keys, self.l_vals, self.l_mask, self.diag_potentials,
                indptr, indices, data
            )

            self.csr_H = sp.csr_matrix((data, indices, indptr), shape=(self.n_dets, self.n_dets))
            print(f"    [DONE] Matrix Built in {time.time() - t0:.1f}s")

            # Cleanup
            del self.l_keys, self.l_vals, row_counts, indices, data

        def _preprocess_hamiltonian(self, qubit_op):
            if hasattr(qubit_op, 'to_list'):
                pauli_list = qubit_op.to_list()
            else:
                pauli_list = qubit_op

            diag_val = np.zeros(self.n_dets, dtype=np.float64)
            groups = {}

            print(f"    [PROC] Processing {len(pauli_list)} Hamiltonian terms...")

            for p_str, coeff in pauli_list:
                if isinstance(coeff, complex):
                    if abs(coeff.imag) > 1e-12: pass

                x_mask = 0;
                z_mask = 0;
                y_count = 0
                for q, char in enumerate(reversed(p_str)):
                    if char == 'X':
                        x_mask |= (1 << q)
                    elif char == 'Z':
                        z_mask |= (1 << q)
                    elif char == 'Y':
                        x_mask |= (1 << q);
                        z_mask |= (1 << q);
                        y_count += 1

                c_val = np.real(coeff)
                if (y_count & 1):
                    c_val = np.real(coeff * (1j) ** y_count)
                else:
                    c_val = c_val * ((-1) ** (y_count >> 1))

                if abs(c_val) < 1e-14: continue

                if x_mask == 0:
                    _fast_add_diagonal(diag_val, self.det_list, z_mask, c_val)
                else:
                    if x_mask not in groups: groups[x_mask] = []
                    groups[x_mask].append((z_mask, c_val))

            self.diag_potentials = diag_val

            unique_x = []
            z_ptr = [0];
            z_masks = [];
            coeffs = []
            for x_m, terms in groups.items():
                unique_x.append(x_m)
                for z, c in terms: z_masks.append(z); coeffs.append(c)
                z_ptr.append(len(z_masks))

            self.unique_x_masks = np.array(unique_x, dtype=np.int64)
            self.z_group_ptr = np.array(z_ptr, dtype=np.int32)
            self.z_masks_flat = np.array(z_masks, dtype=np.int64)
            self.coeffs_flat = np.array(coeffs, dtype=np.float64)

        def matvec(self, v):
            """Ultra-Fast CSR Matrix-Vector Product"""
            return self.csr_H.dot(v)

        @property
        def diagonal(self):
            return self.diag_potentials


    # =============================================================================
    # PART 3: PURE DAVIDSON SOLVER
    # =============================================================================

    def davidson_diagonalization(hamiltonian, n_roots=1, tol=1e-5, max_iter=50, subspace_limit=40):
        """
        Generalized Davidson Algorithm with Diagonal Preconditioning.
        """
        n_dets = hamiltonian.n_dets
        t_start = time.time()
        print(f"\n--- [START] Davidson Solver (Preconditioned) ---")

        # 1. Initial Guess (Lowest Diagonal Elements)
        # Since diagonal is likely roughly sorted by energy,
        # we pick the indices of the lowest diagonal values.
        diag = hamiltonian.diagonal
        # Optimization: Use argpartition for O(N) selection instead of O(N log N) sort
        guess_indices = np.argpartition(diag, n_roots)[:n_roots]

        V = []  # Subspace Vectors
        AV = []  # H * V

        # Init first vectors
        for idx in guess_indices:
            v = np.zeros(n_dets, dtype=np.float64)
            v[idx] = 1.0
            V.append(v)
            AV.append(hamiltonian.matvec(v))

        print(
            f"    [ITER 0] Subspace built. Ref E: {diag[guess_indices[0]] + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.6f}")

        for it in range(max_iter):
            dim = len(V)

            # 2. Build Subspace Matrix T (dim x dim)
            T = np.zeros((dim, dim), dtype=np.float64)
            for i in range(dim):
                for j in range(i, dim):
                    val = np.dot(V[i], AV[j])
                    T[i, j] = val
                    T[j, i] = val

            # 3. Diagonalize Subspace
            eigvals_sub, eigvecs_sub = np.linalg.eigh(T)
            current_e = eigvals_sub[0]

            # 4. Compute Residual & Convergence
            # Ritz vector: C_0*V_0 + ... + C_n*V_n
            c = eigvecs_sub[:, 0]
            ritz_v = np.zeros(n_dets, dtype=np.float64)
            ritz_Av = np.zeros(n_dets, dtype=np.float64)

            for i in range(dim):
                ritz_v += c[i] * V[i]
                ritz_Av += c[i] * AV[i]

            residual = ritz_Av - current_e * ritz_v
            res_norm = np.linalg.norm(residual)

            elapsed = time.time() - t_start
            print(
                f"    [ITER {it + 1}] Dim: {dim} | E: {current_e + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.8f} | Res: {res_norm:.2e} | Time: {elapsed:.1f}s")

            if res_norm < tol:
                print(f"    [DONE] Converged in {it + 1} iterations.")
                return current_e, ritz_v

            # 5. Preconditioning (Davidson Correction)
            # q = r / (D - E)
            precond_denom = diag - current_e
            # Numerical stability: avoid division by very small numbers
            precond_denom = np.where(np.abs(precond_denom) < 1e-5, 1e-5, precond_denom)

            new_v = residual / precond_denom

            # 6. Gram-Schmidt Orthogonalization
            for v_old in V:
                proj = np.dot(v_old, new_v)
                new_v -= proj * v_old

            norm_new = np.linalg.norm(new_v)

            # 7. Append to Subspace
            if norm_new > 1e-9:
                new_v /= norm_new
                V.append(new_v)
                # Parallel Sigma Build for the new vector
                AV.append(hamiltonian.matvec(new_v))
            else:
                print("    [WARN] Subspace collapse (Linear Dependency). Stopping.")
                break

            # 8. Restart Logic (Control RAM)
            if len(V) >= subspace_limit:
                print("    [INFO] Subspace limit reached. Restarting...")
                # Keep only the current best Ritz vector
                V = [ritz_v]
                AV = [ritz_Av]

        return current_e, ritz_v


    # =============================================================================
    # WRAPPER FUNCTION
    # =============================================================================

    def run_ci_davidson(basis_strings, qubit_op, nuclear_repulsion, frozen_shift):
        """
        Entry point.
        basis_strings: Must be sorted list of ints or strings.
        """
        # Initialize optimized Hamiltonian
        H = SparseHamiltonian(basis_strings, qubit_op)

        # Solve
        energy_elec, ci_vector = davidson_diagonalization(H, n_roots=1, tol=1e-5)

        total_energy = energy_elec + nuclear_repulsion + frozen_shift
        return total_energy, ci_vector


# # =============================================================================
# # PART 3: PURE DAVIDSON SOLVER
# # =============================================================================
#
# def davidson_diagonalization(hamiltonian, n_roots=1, tol=1e-5, max_iter=50, subspace_limit=40):
#     """
#     Generalized Davidson Algorithm with Diagonal Preconditioning.
#     """
#     n_dets = hamiltonian.n_dets
#     t_start = time.time()
#     print(f"\n--- [START] Davidson Solver (Preconditioned) ---")
#
#     # 1. Initial Guess (Lowest Diagonal Elements)
#     # Since diagonal is likely roughly sorted by energy,
#     # we pick the indices of the lowest diagonal values.
#     diag = hamiltonian.diagonal
#     # Optimization: Use argpartition for O(N) selection instead of O(N log N) sort
#     guess_indices = np.argpartition(diag, n_roots)[:n_roots]
#
#     V = []  # Subspace Vectors
#     AV = []  # H * V
#
#     # Init first vectors
#     for idx in guess_indices:
#         v = np.zeros(n_dets, dtype=np.float64)
#         v[idx] = 1.0
#         V.append(v)
#         AV.append(hamiltonian.matvec(v))
#
#     print(f"    [ITER 0] Subspace built. Ref E: {diag[guess_indices[0]]  + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.6f}")
#
#     for it in range(max_iter):
#         dim = len(V)
#
#         # 2. Build Subspace Matrix T (dim x dim)
#         T = np.zeros((dim, dim), dtype=np.float64)
#         for i in range(dim):
#             for j in range(i, dim):
#                 val = np.dot(V[i], AV[j])
#                 T[i, j] = val
#                 T[j, i] = val
#
#         # 3. Diagonalize Subspace
#         eigvals_sub, eigvecs_sub = np.linalg.eigh(T)
#         current_e = eigvals_sub[0]
#
#         # 4. Compute Residual & Convergence
#         # Ritz vector: C_0*V_0 + ... + C_n*V_n
#         c = eigvecs_sub[:, 0]
#         ritz_v = np.zeros(n_dets, dtype=np.float64)
#         ritz_Av = np.zeros(n_dets, dtype=np.float64)
#
#         for i in range(dim):
#             ritz_v += c[i] * V[i]
#             ritz_Av += c[i] * AV[i]
#
#         residual = ritz_Av - current_e * ritz_v
#         res_norm = np.linalg.norm(residual)
#
#         elapsed = time.time() - t_start
#         print(f"    [ITER {it + 1}] Dim: {dim} | E: {current_e + nuclear_repulsion_energy_qiskit + frozen_energy_shift:.8f} | Res: {res_norm:.2e} | Time: {elapsed:.1f}s")
#
#         if res_norm < tol:
#             print(f"    [DONE] Converged in {it + 1} iterations.")
#             return current_e, ritz_v
#
#         # 5. Preconditioning (Davidson Correction)
#         # q = r / (D - E)
#         precond_denom = diag - current_e
#         # Numerical stability: avoid division by very small numbers
#         precond_denom = np.where(np.abs(precond_denom) < 1e-5, 1e-5, precond_denom)
#
#         new_v = residual / precond_denom
#
#         # 6. Gram-Schmidt Orthogonalization
#         for v_old in V:
#             proj = np.dot(v_old, new_v)
#             new_v -= proj * v_old
#
#         norm_new = np.linalg.norm(new_v)
#
#         # 7. Append to Subspace
#         if norm_new > 1e-9:
#             new_v /= norm_new
#             V.append(new_v)
#             # Parallel Sigma Build for the new vector
#             AV.append(hamiltonian.matvec(new_v))
#         else:
#             print("    [WARN] Subspace collapse (Linear Dependency). Stopping.")
#             break
#
#         # 8. Restart Logic (Control RAM)
#         if len(V) >= subspace_limit:
#             print("    [INFO] Subspace limit reached. Restarting...")
#             # Keep only the current best Ritz vector
#             V = [ritz_v]
#             AV = [ritz_Av]
#
#     return current_e, ritz_v
#
#
# # =============================================================================
# # WRAPPER FUNCTION
# # =============================================================================
#
# def run_ci_davidson(basis_strings, qubit_op, nuclear_repulsion, frozen_shift):
#     """
#     Entry point.
#     basis_strings: Must be sorted list of ints or strings.
#     """
#     # Initialize optimized Hamiltonian
#     H = SparseHamiltonian(basis_strings, qubit_op)
#
#     # Solve
#     energy_elec, ci_vector = davidson_diagonalization(H, n_roots=1, tol=1e-5)
#
#     total_energy = energy_elec + nuclear_repulsion + frozen_shift
#     return total_energy, ci_vector





def det_array_to_string_converter(det_array):
    basis_strings = np.array(
        [''.join(row.astype(str)) for row in det_array],
        dtype=str
    )
    return basis_strings
# # ------------------------ Example ------------------------
#
#
#
# my_basis = ["00110011", "01010101", "11000011", "11001100"]
# # qubit_op = ...  # Your SparsePauliOp JW Hamiltonian
#
# import pickle
# if molecule == 'H2O':
#     with open('det_list_mbpt_rank_4_H2O_631g_1.0.pkl', 'rb') as file:
#         det_list = pickle.load(file)
# if molecule == 'N2':
#     with open('det_list_mbpt_rank_4_N2_631g_1.0.pkl', 'rb') as file:
#         det_list = pickle.load(file)
#
# print(det_list)
# print(det_list.shape)
# det_list = sort_and_remove_duplicates(det_list)
# print(det_list.shape)
#
#
# basis_strings = np.array(
#     [''.join(row.astype(str)) for row in det_list],
#     dtype=str
# )
#
# print(basis_strings)
# my_basis = basis_strings
#
#
#
#
# energy, ci_vec = run_ci_davidson(my_basis, qubit_op, nuclear_repulsion_energy_qiskit, frozen_energy_shift)
# print(ci_vec)
#
# print('*******************************************************************************')
# print("CI Davidson Energy:", energy)
# print(exact_energy)
# print(abs(exact_energy - energy))









#************************-----------------------------------------------------------------------------------------------------------------------





import numpy as np
import time
from scipy.sparse import spmatrix
from scipy.sparse.linalg import eigsh, LinearOperator


#------------------------------------------------------------------------------------------------------------------------------



it_energy = []
it_diag_subspace_dim = []
it_dominant_subspace_dim = []
it_blacklisted_subspace_dim = []
it_ci_coeff = []
# Energy calculation from sampled dets from hardware

hf_qubit = rev_to_qubit_convention_transformer([hf_rev])
hf_array = np.asarray(hf_qubit)
sampled_dets_binary_array = np.vstack((hf_array, sampled_dets_binary_array))
sorted_sampled_dets_binary_array = sort_and_remove_duplicates_manual(sampled_dets_binary_array)
# E_sampled_dets, ci_coeffs_sampled_dets = subspace_diagonalizer(sorted_sampled_dets_binary_array)
sorted_sampled_dets_binary_strings = det_array_to_string_converter(sampled_dets_binary_array)
print(sorted_sampled_dets_binary_strings.size)
# E_sampled_dets, ci_coeffs_sampled_dets = run_ci_davidson(sorted_sampled_dets_binary_strings, jw_mapped_hamiltonian, nuclear_repulsion_energy_qiskit, frozen_energy_shift)
#
# print('ci_coeffs_sampled_dets',ci_coeffs_sampled_dets)
# print('ci_coeffs_sampled_dets size',ci_coeffs_sampled_dets.size)
# print('Energy from sampled dets', E_sampled_dets)
# print('Difference from CASCI for sampled det energy', abs(E_sampled_dets-exact_energy))
# print('Number of dets in the diagonalization subspace:', len(sorted_sampled_dets_binary_array))


# exit()
#
# dominant_loc = np.where(np.abs(ci_coeffs_sampled_dets) > 1e-9)[0]
# print(dominant_loc)
# dominant_dets_array_qubit_conv_old = sorted_sampled_dets_binary_array[dominant_loc]
# dominant_ci_coeffs_old =ci_coeffs_sampled_dets[dominant_loc]
# num_dominant_dets = len(dominant_dets_array_qubit_conv_old)
#
# blacklisted_loc = np.where(np.abs(ci_coeffs_sampled_dets) < 1e-9)[0]
# print(blacklisted_loc)
# blacklisted_dets_array_qubit_conv_old = sorted_sampled_dets_binary_array[blacklisted_loc]
# num_blacklisted_dets = len(blacklisted_dets_array_qubit_conv_old)
# print(blacklisted_dets_array_qubit_conv_old)
# print(blacklisted_dets_array_qubit_conv_old.shape)
#
#
#
# it_energy.append(E_sampled_dets)
# it_ci_coeff.append(ci_coeffs_sampled_dets)
# it_diag_subspace_dim.append(len(sorted_sampled_dets_binary_array))
# it_dominant_subspace_dim.append(num_dominant_dets)
# it_blacklisted_subspace_dim.append(num_blacklisted_dets)






#----------------------------
dominant_mbpt_det_array_qubit_conv = np.vstack((np.asarray(hf_qubit),dominant_mbpt_det_array_qubit_conv))
sorted_mbpt_dets_array = sort_and_remove_duplicates(dominant_mbpt_det_array_qubit_conv)
print(sorted_mbpt_dets_array)
print(sorted_mbpt_dets_array.shape)
# E_mbpt, ci_mbpt = subspace_diagonalizer(sorted_mbpt_dets_array)
sorted_mbpt_dets_string = det_array_to_string_converter(sorted_mbpt_dets_array)
print(sorted_mbpt_dets_string.shape)
E_mbpt, ci_mbpt = run_ci_davidson(sorted_mbpt_dets_string, jw_mapped_hamiltonian, nuclear_repulsion_energy_qiskit, frozen_energy_shift)
print(ci_mbpt)
print(ci_mbpt.shape)
np.save('energy_N2_ccpvdz_0th_it.npy',E_mbpt)
# b = solve_qubit(np.asarray(sorted_mbpt_dets_array), jw_mapped_hamiltonian)
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy_qiskit + frozen_energy_shift
# eigen_values.sort()
# # print()
# E_mbpt = eigen_values[0]
print('Perturbative diag energy', E_mbpt)
print('Difference with CASCI:           ', abs(exact_energy-E_mbpt))



dominant_loc = np.where(np.abs(ci_mbpt) > 1e-9)[0]
print(dominant_loc)
dominant_dets_array_qubit_conv_old = sorted_mbpt_dets_array[dominant_loc]
dominant_ci_coeffs_old =ci_mbpt[dominant_loc]
num_dominant_dets = len(dominant_dets_array_qubit_conv_old)


blacklisted_loc = np.where(np.abs(ci_mbpt) < 1e-9)[0]
print(blacklisted_loc)
blacklisted_dets_array_qubit_conv_old = sorted_mbpt_dets_array[blacklisted_loc]
num_blacklisted_dets = len(blacklisted_dets_array_qubit_conv_old)
print(blacklisted_dets_array_qubit_conv_old)
print(blacklisted_dets_array_qubit_conv_old.shape)



it_energy.append(E_mbpt)
it_ci_coeff.append(ci_mbpt)
it_diag_subspace_dim.append(len(sorted_mbpt_dets_array))
it_dominant_subspace_dim.append(num_dominant_dets)
it_blacklisted_subspace_dim.append(num_blacklisted_dets)



#----------------------------




sorted_old_dets_array_qubit_conv = sorted_det_array_qubit_conv
# print(sorted_old_dets_array_qubit_conv)
# print(sorted_old_dets_array_qubit_conv.shape)
print('Projection and Diagonalization.....')
# E_old, ci_coeffs_old = subspace_diagonalizer(sorted_old_dets_array_qubit_conv)

sorted_old_dets_string_qubit_conv = det_array_to_string_converter(sorted_old_dets_array_qubit_conv)
E_old, ci_coeffs_old = run_ci_davidson(sorted_old_dets_string_qubit_conv, jw_mapped_hamiltonian, nuclear_repulsion_energy_qiskit, frozen_energy_shift)

print(E_old)
print(ci_coeffs_old)
print('Exact CASCI Energy:                                              ', exact_energy)
print('Difference with CASCI:                                           ', abs(E_old-exact_energy))
print('Diagonalization dimension for sampled_dets + mbpt selected dets:',len(sorted_old_dets_array_qubit_conv))
# E_old, ci_coeffs_old = subspace_ci_energy(sorted_dominant_old_dets)
# print(E_old)
# print(ci_coeffs_old)
# print(ci_coeffs_old.shape)

hf_qubit_conv_list = rev_to_qubit_convention_transformer([hf_rev])

# rbm_input_dets_array_qubit_conv = np.delete(sorted_old_dets_array_qubit_conv, 0, axis=0)#.tolist()
# print(rbm_input_dets_array_qubit_conv.shape)

ci_coeffs_old_copy = ci_coeffs_old.copy()
print(ci_coeffs_old_copy.shape)
ci_coeffs_without_hf = np.delete(ci_coeffs_old_copy, 0, axis=0)
print(ci_coeffs_without_hf.shape)


dominant_loc = np.where(np.abs(ci_coeffs_old) > 1e-9)[0]
print(dominant_loc)
dominant_dets_array_qubit_conv_old = sorted_old_dets_array_qubit_conv[dominant_loc]
dominant_ci_coeffs_old =ci_coeffs_old_copy[dominant_loc]
dominant_ci_coeffs_without_hf = np.delete(dominant_ci_coeffs_old, 0, axis=0)
num_dominant_dets = len(dominant_dets_array_qubit_conv_old)

rbm_input_dets_array_qubit_conv = np.delete(dominant_dets_array_qubit_conv_old, 0, axis=0)#.tolist()
print(rbm_input_dets_array_qubit_conv.shape)


# rbm_input_dets_rev = qubit_to_rev_convention_transformer(rbm_input_dets_qubit_conv)
sample_list = sample_list_generator_for_rbm(rbm_input_dets_array_qubit_conv, dominant_ci_coeffs_without_hf)
print(sample_list)
print(sample_list.shape)
print(rbm_input_dets_array_qubit_conv)

# print(ci_coeffs_old)
# print(len(ci_coeffs_old))
# print(np.where(np.abs(ci_coeffs_old) > 1e-10)[0].size)

# blacklisted_dets_list_qubit_conv = []
blacklisted_loc = np.where(np.abs(ci_coeffs_old) < 1e-9)[0]
print(blacklisted_loc)
blacklisted_dets_array_qubit_conv_old = sorted_old_dets_array_qubit_conv[blacklisted_loc]
num_blacklisted_dets = len(blacklisted_dets_array_qubit_conv_old)
print(blacklisted_dets_array_qubit_conv_old)
print(blacklisted_dets_array_qubit_conv_old.shape)

print(sorted_old_dets_array_qubit_conv.shape)
print(dominant_dets_array_qubit_conv_old.shape)
print(blacklisted_dets_array_qubit_conv_old.shape)


iteration_data = []
iteration_data.append([exact_energy])
it_energy.append(E_old)
it_ci_coeff.append(ci_coeffs_old)
it_diag_subspace_dim.append(len(sorted_old_dets_array_qubit_conv))
it_dominant_subspace_dim.append(num_dominant_dets)
it_blacklisted_subspace_dim.append(num_blacklisted_dets)

print('iteration will start after this')
count_iter = 0
exit_counter = 0
exit_cond_en_subseq_it = []


E_sdtq = E_old.copy()
hf_qubit = hf_rev[::-1]
print(hf_qubit)
#import rbm_particle_spin_conserved_training
#from rbm_particle_spin_conserved_training import ConstrainedRBM2
import time



use_dominant_dets = inp.use_dominant_dets_for_gibbs  #False
while True:
    print ('seed value:', seed_val)
    print('count_iter', count_iter)
    print ('num_gibbs_sampling:', n_gibbs_sampling)
    count_iter += 1
    generated_samples = rbm_training_and_generation(binaries_data=sample_list, n_components=n_components,learning_rate=learning_rate, batch_size=batch_size,n_gibbs_sampling=n_gibbs_sampling, n_alpha_orb=num_spatial_orbitals, n_alpha_p=num_elec_a, dominant_dets_for_gibbs=dominant_dets_array_qubit_conv_old,use_dominant_dets=use_dominant_dets,total_dets=blacklisted_dets_array_qubit_conv_old)
    print('Generated_samples',generated_samples)
    print(generated_samples.shape)
    p_conserved_dets = filter_particle_conserved_dets_numpy(generated_samples, num_total_particles)# filter_particle_conserved_dets(generated_samples.tolist(),num_total_particles)
    p_sp_conserved_dets = filter_spin_conserved_dets_numpy(p_conserved_dets,num_alpha_particles, num_beta_particles)
    print('number of particle and spin conserved dets', len(p_sp_conserved_dets))

    #------------------
    # new_dets = find_rows_not_in_array2_numpy(p_sp_conserved_dets, sorted_old_dets_array_qubit_conv) #find_elements_not_in_list2(p_conserved_dets, sorted_old_dets_array_qubit_conv.tolist())
    new_dets = find_rows_not_in_array2_numpy(p_sp_conserved_dets,
                                             sorted_old_dets_array_qubit_conv)  # find_elements_not_in_list2(p_conserved_dets, sorted_old_dets_array_qubit_conv.tolist())
    print('No. of generated new dets:',len(new_dets))
#    if len(new_dets) == 0:
#        break
    print('Shape new dets:                          ', new_dets.shape)
    print('dominant_dets_array_qubit_conv_old.shape :',dominant_dets_array_qubit_conv_old.shape)
    total_subspace = np.vstack((dominant_dets_array_qubit_conv_old,new_dets)) # sorted_old_dets_array_qubit_conv.tolist() + new_dets
    print('Num dets in total subspace:      ',len(total_subspace))
    print('Num dets in blacklisted subspace:     ',len(blacklisted_dets_array_qubit_conv_old))
    print('find_elements_not_in_list2')
    start_time = time.time()

    diag_subspace = find_rows_not_in_array2_numpy(total_subspace, blacklisted_dets_array_qubit_conv_old) # find_elements_not_in_list2(total_subspace,blacklisted_dets_list_qubit_conv)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('find_elements_not_in_list2 elapsed_time',elapsed_time)
    print('Dets sorting')
    start_time = time.time()

    # # ----------------- Mirror ----------------------
    #
    # # diag_subspace_copy = diag_subspace.copy()
    # # alpha_dets_new, beta_dets_new = create_alpha_beta_sectors_numpy(diag_subspace_copy)
    # #
    # # mirror_dets_new = np.hstack((beta_dets_new, alpha_dets_new))
    print ('Adding MIRROR DETS...................')
    print ('diag subspace shape', diag_subspace.shape)
    diag_subspace = add_mirror_dets(diag_subspace)
    print ('Mirror added diag subspace shape:', diag_subspace.shape)
    # #np.vstack((diag_subspace, mirror_dets_new))
    #
    # # -----------------------------------------------

    sorted_dominant_new_dets = sort_and_remove_duplicates_manual(diag_subspace)
    diag_subspace_dim = len(sorted_dominant_new_dets)
    print(sorted_dominant_new_dets)
    print(type(sorted_dominant_new_dets))
    print('sorted dominant new dets shape:',sorted_dominant_new_dets.shape)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Dets sorting elapsed_time',elapsed_time)
    # E_new, ci_coeffs_new = subspace_ci_energy(sorted_dominant_new_dets)
    # E_new, ci_coeffs_new = subspace_diagonalizer(sorted_dominant_new_dets)


    sorted_dominant_new_dets_string = det_array_to_string_converter(sorted_dominant_new_dets)
    E_new, ci_coeffs_new = run_ci_davidson(sorted_dominant_new_dets_string, jw_mapped_hamiltonian, nuclear_repulsion_energy_qiskit, frozen_energy_shift)


    
    print('E_old',E_old)
    print('E_new',E_new)
    print('CASCI Energy', exact_energy)
    print('Num ci_coeffs_new', len(ci_coeffs_new))
    dominant_loc_new = np.where(np.abs(ci_coeffs_new) > 1e-9)[0]
    print('num_non_zero_coeffs:     ',len(dominant_loc_new))
    print('Energy Diff from previous iteration', abs(E_new-E_old))
    print('Energy Difference from CASCI', abs(E_new-exact_energy))
    print('Diagonalization Subspace:', diag_subspace_dim)

    it_energy.append(E_new)
    it_diag_subspace_dim.append(diag_subspace_dim)
    it_ci_coeff.append(ci_coeffs_new)
#    if abs(E_new-E_old) < en_conv_thresh:
#        break

    # if abs(E_new-E_old) < 1e-4:
    #     use_dominant_dets = True
        # exit_cond_en_subseq_it.append(abs(E_new-E_old))
        # exit_counter += 1
        # break
    # if exit_counter > 2:
    #     if exit_cond_en_subseq_it[0] < 1e-13 and exit_cond_en_subseq_it[1] < 1e-13:
    #     break

    blacklisted_loc = np.where(np.abs(ci_coeffs_new) < 1e-9)[0]
    print('blacklist generation starts')
    blacklisted_dets_array_qubit_conv_new = sorted_dominant_new_dets[blacklisted_loc]
    blacklisted_dets_array_qubit_conv_old = np.vstack((blacklisted_dets_array_qubit_conv_old, blacklisted_dets_array_qubit_conv_new))
    #sort_and_remove_duplicates_manual
    num_blacklisted_dets = len(blacklisted_dets_array_qubit_conv_old)
    it_blacklisted_subspace_dim.append(num_blacklisted_dets)
#    E_old = E_new

    np.save('blacklisted_det_N2_count.npy',num_blacklisted_dets)
#    np.save(
#        'sorted_old_dets_mbpt_rank_' + str(mbpt_max_rank) + '_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(
#            basis) + '_' + str(
#            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '.npy', sorted_dominant_new_dets)

#    np.save(
#        'blacklisted_dets_mbpt_rank_' + str(mbpt_max_rank) + '_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(
#            basis) + '_' + str(
#            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '.npy', blacklisted_dets_array_qubit_conv_old)

#    np.save('ci_coeffs_mbpt_rank_' + str(mbpt_max_rank) + '_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(
#        basis) + '_' + str(
#        bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '.npy', ci_coeffs_new)

    # hf_qubit_conv = rev_to_qubit_convention_transformer([hf_rev])

    dominant_dets_array_qubit_conv_old = sorted_dominant_new_dets[dominant_loc_new]
    dominant_ci_coeffs_old = ci_coeffs_new[dominant_loc_new]
    dominant_ci_coeffs_without_hf = np.delete(dominant_ci_coeffs_old, 0, axis=0)
    num_dominant_dets = len(dominant_dets_array_qubit_conv_old)
    it_dominant_subspace_dim.append(num_dominant_dets)

    rbm_input_dets_array_qubit_conv = np.delete(dominant_dets_array_qubit_conv_old, 0, axis=0)  # .tolist()
    print(rbm_input_dets_array_qubit_conv.shape)

    # rbm_input_dets_rev = qubit_to_rev_convention_transformer(rbm_input_dets_qubit_conv)
    sample_list = sample_list_generator_for_rbm(rbm_input_dets_array_qubit_conv, dominant_ci_coeffs_without_hf)
    sorted_old_dets_array_qubit_conv = sorted_dominant_new_dets

    en_arr = np.array(it_energy)#, dtype=float)
    diag_subspace_arr = np.array(it_diag_subspace_dim)

    np.save('energy_mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '.npy', en_arr)
    np.save('diag_subspace_mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '.npy', diag_subspace_arr)

    print('Energy Iteration so far                                  :', it_energy)
    print('Diagonalization space dimension so far                   :', it_diag_subspace_dim)
    print('No. of dominant dets so far:                             ', it_dominant_subspace_dim)
    print('No. of blacklisted dets so far:                          ', it_blacklisted_subspace_dim)
    print('--------------------------------------------------------------------------------------')
    if len(new_dets) == 0:
        break
    
    if abs(E_new-E_old) < en_conv_thresh:
        break
    E_old = E_new
    iteration_data = []
    iteration_data.append([exact_energy])
    iteration_data.append(it_energy)
    iteration_data.append(it_diag_subspace_dim)
    iteration_data.append(it_dominant_subspace_dim)
    iteration_data.append(it_blacklisted_subspace_dim)
    iteration_data.append(casci_coeff)
    iteration_data.append(it_ci_coeff)
    print('dominant_dets_array_qubit_conv_old.shape:            ',dominant_dets_array_qubit_conv_old.shape)
    #dominant_dets_array_qubit_conv_old = add_mirror_dets(dominant_dets_array_qubit_conv_old)
    #dominant_dets_array_qubit_conv_old = sort_and_remove_duplicates_manual(dominant_dets_array_qubit_conv_old)
    print('dominant_dets_array_qubit_conv_old.shape with mirror dets:',dominant_dets_array_qubit_conv_old.shape)
    if synthetic_noise == 'yes':
        with open('mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
                bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '_two_qubit_noise_' + str(
                prob_2) + '.pkl','wb') as file:  # 'wb' means write binary mode
            pickle.dump(iteration_data, file)

    if synthetic_noise == 'no':
        with open('mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
                bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '_real_hardware_noise.pkl','wb') as file:  # 'wb' means write binary mode
            pickle.dump(iteration_data, file)



print(len(diag_subspace))
print('MBPT Driven RBM-SQD Energy',abs(exact_energy-E_new))

print('Initial sampled energy',abs(exact_energy-E_sdtq))
# ml_output_excitation = trans_idx_finder(binaries_data=sample_list, main_list=hf_rev, n_components=n_components, learning_rate=learning_rate, batch_size=batch_size, n_gibbs_sampling=n_gibbs_sampling)
# print(ml_output_excitation)
iteration_data = []
iteration_data.append([exact_energy])
iteration_data.append(it_energy)
iteration_data.append(it_diag_subspace_dim)
iteration_data.append(it_dominant_subspace_dim)
iteration_data.append(it_blacklisted_subspace_dim)
iteration_data.append(casci_coeff)
iteration_data.append(it_ci_coeff)


if synthetic_noise == 'yes':
    with open('mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '_two_qubit_noise_' + str(
            prob_2) + '.pkl','wb') as file:  # 'wb' means write binary mode
        pickle.dump(iteration_data, file)

if synthetic_noise == 'no':
    with open('mbpt_rank_'+str(mbpt_max_rank)+'_rbm_sqd_' + str(run) + '_' + str(molecule) + '_' + str(basis) + '_' + str(
            bond_dist) + '_n_reps_' + str(n_reps) + '_' + str(ansatz_used) + '_seed_val_'+str(seed_val)+'_real_hardware_noise.pkl','wb') as file:  # 'wb' means write binary mode
        pickle.dump(iteration_data, file)

print (it_ci_coeff)
print ('seed value:',seed_val)
print ('Molecule', molecule)
print ('Bond dist', bond_dist)
print ('******************************************************The END************************************************************')




