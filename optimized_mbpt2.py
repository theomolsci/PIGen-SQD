# from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers.pyscfd import PySCFDriver
from qiskit_algorithms import NumPyMinimumEigensolver
#from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
#from qiskit_nature.second_q.properties.particle_number import ParticleNumber
#from qiskit_nature.second_q.properties.angular_momentum import AngularMomentum
#from qiskit_aer.primitives.sampler import Sampler
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
from qiskit_aer.primitives import Estimator as AerEstimator
import pickle
from joblib import Parallel, delayed
from functools import partial
from scipy.optimize import minimize
import inp
# from mitiq import zne
import qiskit_aer
#from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
from qiskit.quantum_info import SparsePauliOp
import time

# ---------------------------------------------- Set all the parameters ------------------------------------------------------


method = inp.method
# sub_type = inp.sub_type
#initial_pps = inp.initial_pps
#adapt_threshold = inp.adapt_threshold
#pps_aps_decoupling_threshold = inp.pps_aps_decoupling_threshold
#operator_ordering = inp.operator_ordering
# operator_ordering = 'ADUCC'
#pool_type = inp.pool_type
molecule = inp.molecule
#software = inp.software
#param_optimizer = inp.param_optimizer
basis = inp.basis
t2_thresh = inp.t2_thresh
t1_thresh = inp.t1_thresh
s_thresh = inp.s_thresh
two_body_int_thresh = inp.two_body_int_thresh
bond_dist = inp.bond_dist
#bond_dist = float(input())
sampler_data_available = inp.sampler_data_available
run = inp.run
# conv = 1e-5
frozen_core = 'no'
#noiseless_estimator = Estimator()
noise = 'no'
mitigation = 'no'
c_not_eff = 'no'
mbpt_max_rank = inp.mbpt_max_rank

# noise = 'yes'
# mitigation = 'yes'
# c_not_eff = 'yes'

# if noise == 'yes':
#     run = int(input())

#bond_stretch = float(input())
bond_stretch = bond_dist
# bond_stretch = 2.0
stretch = str(bond_stretch) + 'AA'

# ------------------------------------------------
if molecule == 'H4':
    eq_dist = 1.0
    stretch = str(bond_stretch) + 'AA'
    print(stretch)
    bond_dist = bond_stretch * eq_dist

# ---------------------------------------------------------------------------------------------------------------------------

if noise == 'yes':
    ######################### Building up the noise ##########################
    # Error probabilities
    prob_1 = 1e-3  # 1-qubit gate
    prob_2 = 1e-2  # 2-qubit gate

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

if molecule == 'H4':
    driver = PySCFDriver(atom="H 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist) + "; H 0.0 0.0 " + str(
        2 * bond_dist) + "; H 0.0 0.0 " + str(3 * bond_dist), basis=basis)
    coord = "H 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist) + "; H 0.0 0.0 " + str(
        2 * bond_dist) + "; H 0.0 0.0 " + str(3 * bond_dist)

if molecule == 'H2O':
    H_y_eq_dist = 0.75736617840905475162
    H_z_eq_dist = 0.58665191707013439891
    coord = 'O 0.0 0.0 0.0; H 0.0 -' + str(bond_stretch * H_y_eq_dist) + ' -' + str(
        bond_stretch * H_z_eq_dist) + '; H 0.0 ' + str(bond_stretch * H_y_eq_dist) + ' -' + str(
        bond_stretch * H_z_eq_dist)
    driver = PySCFDriver(atom='O 0.0 0.0 0.0; H 0.0 -' + str(bond_stretch * H_y_eq_dist) + ' -' + str(
        bond_stretch * H_z_eq_dist) + '; H 0.0 ' + str(bond_stretch * H_y_eq_dist) + ' -' + str(
        bond_stretch * H_z_eq_dist), basis=basis)
    frozen_core = 'yes'

if molecule == 'LiH':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
    coord = "Li 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist)
    driver = PySCFDriver(atom="Li 0.0 0.0 0.0; H 0.0 0.0 " + str(1 * bond_dist), basis=basis)

if molecule == 'N2':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
    coord = "N 0.0 0.0 0.0; N 0.0 0.0 " + str(1 * bond_dist)
    driver = PySCFDriver(atom="N 0.0 0.0 0.0; N 0.0 0.0 " + str(1 * bond_dist), basis=basis)
    frozen_core = 'yes'

if molecule == 'BeH2':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
    coord = "Be 0.0 0.0 0.0; H 0.0 0.0 " + str(bond_stretch * eq_dist) + "; H 0.0 0.0 -" + str(bond_stretch * eq_dist)
    driver = PySCFDriver(
        atom="Be 0.0 0.0 0.0; H 0.0 0.0 " + str(bond_stretch * eq_dist) + "; H 0.0 0.0 -" + str(bond_stretch * eq_dist),
        basis=basis)
    frozen_core = 'yes'
    print(coord)

if molecule == 'H6':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
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
    print (coord)



if molecule == 'H2O_tri':
    eq_dist = 1.0
    bond_dist = bond_stretch * eq_dist
    coord = "O 1.436782 -0.763439 -0.009035; H 2.368835 -0.892409 0.028995; H 1.267991 0.168979 0.059135; O -1.4128 -0.887834 -0.093919; H -1.740681 -1.270861 0.703265; H -0.522848 -1.20639 -0.189994; O -0.074775 1.653285 0.100186; H -0.751959 0.985251 0.076561; H -0.214998 2.199333 -0.655816"
    driver = PySCFDriver(atom="C 0.0 0.0 "+str(c_pos)+"; C 0.0 0.0 -"+str(c_pos)+"; H 0.0 0.0 "+str(h_pos)+"; H 0.0 0.0 -"+str(h_pos), basis = basis)



problem = driver.run()

print(coord)
# if frozen_core == 'yes':
#     fc_transformer = FreezeCoreTransformer()
#     problem = fc_transformer.transform(problem)
if frozen_core == 'yes':
    fc_transformer = FreezeCoreTransformer()
    problem = fc_transformer.transform(problem)

if frozen_core == 'yes':
    frozen_energy_shift = problem.hamiltonian.constants['FreezeCoreTransformer']
if frozen_core == 'no':
    frozen_energy_shift = 0.0

print(frozen_energy_shift)

hamiltonian = problem.hamiltonian.second_q_op()
#print(hamiltonian)
#print(len(hamiltonian))

# h_indices = np.asarray(hamiltonian)
# one_body_indices = [print(len(i)) for i in h_indices]# if len(i) < 10]  # one body second q strings are of the length 7
# exit()

#print(type(hamiltonian))
#print(hamiltonian.is_hermitian())

mapper = JordanWignerMapper()
qubit_op = mapper.map(hamiltonian)
print('qubit Hamiltonian', len(qubit_op))

num_particles = problem.num_particles
num_alpha_particles = num_particles[0]
num_beta_particles = num_particles[1]
num_total_particles = num_alpha_particles + num_beta_particles
num_spin_orbitals = problem.num_spin_orbitals
num_spatial_orbitals = problem.num_spatial_orbitals
nuclear_repulsion_energy = problem.nuclear_repulsion_energy
nuclear_repulsion_energy_qiskit = nuclear_repulsion_energy
orbital_energy_spatial = list(problem.orbital_energies)

print(len(orbital_energy_spatial))

oe = orbital_energy_spatial + orbital_energy_spatial
alpha_occupations = problem.orbital_occupations
beta_occupations = problem.orbital_occupations_b

print('number of particles:                   ', num_particles)
print('number of alpha spin particles         ', num_alpha_particles)
print('number of beta spin particles         ', num_beta_particles)
print('total number of particles              ', num_total_particles)
print('number of spin orbitals:               ', num_spin_orbitals)
print('nuclear repulsion energy:              ', nuclear_repulsion_energy)
print('orbital energy spatial:                ', orbital_energy_spatial)
print(oe)
# *****************************************************************************************************************************************


# ********************************************************* :one and two electron integrals extraction: ****************************************************************************************

import re


def integer_finder_from_string(string):
    # returns integer list

    # Sample string
    # my_string = "+_12 -_10"

    # Search for integers following underscores using regular expression
    integers = re.findall(r'(?<=_)\d+', string)

    # Convert the integers to integers
    integers = [int(num) for num in integers]

    #    print(integers)  # Output: [11, 0]
    return integers


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
    two_body_indices = np.asarray(
        [i for i in h_indices if len(i) > 9])  #
    print('len(two_body_indices)',len(two_body_indices))
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
# *********************************************************************************************************************************************


# **********************************************************************************************************************************************


# ----------------------------------------- Initial HF state *------------------------------------------------------------------------------------------------------
init_state = HartreeFock(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, qubit_mapper=mapper)

ansatz = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations='sd',
             qubit_mapper=mapper, initial_state=init_state)

sd_ex_list = ansatz.excitation_list
print(len(sd_ex_list))

num_singles = 0
singles_ex_list = []
for ex in sd_ex_list:
    if len(ex[0]) == 1:
        num_singles += 1
        singles_ex_list.append(ex)
print(singles_ex_list)

# mp2 = MP2InitialPoint()
# mp2.ansatz = ansatz
# mp2.problem = problem
# result = mp2.to_numpy_array()#.tolist()
# t2amplitudes = mp2.t2_amplitudes
# print(t2amplitudes)
# print('-----------------------------')
# #print(mp2_val_red)
# print(t2amplitudes[0,0,0,0])
# #print(pruned_excitation_list[num_singles])
#
# # print(type(t2amplitudes))
# print(t2amplitudes.shape)
# # print('num alpha',num_alpha_particles)
# # print(num_spatial_orbitals)
# num_occ = num_alpha_particles
# num_virt = num_spatial_orbitals - num_alpha_particles
# print ('num OCC spatial:', num_occ)
# print ('num VIRT spatial:', num_virt)


# with open(str(stretch)+'_'+str(molecule)+'_sto6g_final_excitation_list.pkl', 'rb') as file:
#     doubles_ex_list = pickle.load(file)
# with open(str(stretch)+'_'+str(molecule)+'_sto6g_t_list.pkl', 'rb') as file:
#     mp2_val_red = pickle.load(file)

with open(str(stretch) + '_' + str(molecule) + '_' + str(basis) + '_final_excitation_list.pkl', 'rb') as file:
    doubles_ex_list = pickle.load(file)
with open(str(stretch) + '_' + str(molecule) + '_' + str(basis) + '_t_list.pkl', 'rb') as file:
    mp2_val_red = pickle.load(file)

#print(doubles_ex_list)
print(len(doubles_ex_list))
print(len(mp2_val_red))
print('****************************************************************')

# Convert each element to a standard Python float
mp2_val_red = [float(x) for x in mp2_val_red]
#print(mp2_val_red)
sorted_mp2 = sorted(mp2_val_red, key=abs)
#print(sorted_mp2)
print(np.where(np.abs(np.asarray(mp2_val_red)) > t2_thresh)[0].size)
dominant_doubles_position_in_list = np.where(np.abs(np.asarray(mp2_val_red)) > t2_thresh)[0]

dominant_doubles_ex_list = [doubles_ex_list[i] for i in dominant_doubles_position_in_list]
print(len(dominant_doubles_ex_list))

#print(dominant_doubles_ex_list)
#print(len(dominant_doubles_ex_list))
#print(len(mp2_val_red))



def mp2_dict(ex_list, mp2_t2_val):
    test_keys = ex_list
    test_values = mp2_t2_val

    # Printing original keys-value lists
#    print("Original key list is : " + str(test_keys))
#    print("Original value list is : " + str(test_values))

    # create a list of tuples using enumerate()
    tuples = [(key, value)
              for i, (key, value) in enumerate(zip(test_keys, test_values))]

    # convert list of tuples to dictionary using dict()
    t2_mp2 = dict(tuples)
    return t2_mp2


t2_mp2 = mp2_dict(doubles_ex_list, mp2_val_red)


def t2_tensor(ex_list):
    t2_tens = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
    for inds in ex_list:
#        print('inds', inds)
        k = inds[0][0]
        l = inds[0][1]
        c = inds[1][0]
        d = inds[1][1]
        t2_tens[k, l, c, d] = t2_mp2[inds]
    return t2_tens


t2_tensor1 = t2_tensor(doubles_ex_list)
#print(doubles_ex_list)


def v_t2_c(sing_ex_list, two_body, t2_mp2_val):
    # singles_guess = np.zeros((num_spin_orbitals, num_spin_orbitals))
    singles_guess = [0.0] * num_singles
    m = 0
    for sing_ind in sing_ex_list:
        i = sing_ind[0][0]
        a = sing_ind[1][0]
        # print(i,a)
        singles_guess[m] += np.einsum('clk,lkc', two_body[i, :, :, :], t2_mp2_val[:, :, a, :])
        singles_guess[m] += np.einsum('dck,kdc', two_body[:, :, a, :], t2_mp2_val[i, :, :, :])
        singles_guess[m] += np.einsum('clk,lkc', two_body[i, :, :, :], t2_mp2_val[:, :, :, a])
        singles_guess[m] += np.einsum('cdk,kdc', two_body[:, :, a, :], t2_mp2_val[i, :, :, :])
        # print(m)
        m += 1

    return singles_guess


initial_singles = v_t2_c(singles_ex_list, two_body_ints, t2_tensor1)
initial_singles = list(initial_singles)

#print('initial singles:    ', initial_singles)


large_singles = [x for x in initial_singles if abs(x)>t1_thresh]
num_large_singles = len(large_singles)

#print(singles_ex_list)


dominant_singles_position_in_list = np.where(np.abs(np.asarray(initial_singles)) > t1_thresh)[0]
dominant_singles_ex_list = [singles_ex_list[i] for i in dominant_singles_position_in_list]

# print(dominant_singles_ex_list)
# print(len(singles_ex_list))
# print(len(dominant_singles_ex_list))




# print(two_body_ints)
# print(t2_tensor1)

non_zero_indices = np.nonzero(t2_tensor1)
non_zero_values = t2_tensor1[non_zero_indices]
# print(non_zero_values)
# print(np.sum(abs(non_zero_values) > t2_thresh))

non_zero_indices = np.nonzero(two_body_ints)
non_zero_values = two_body_ints[non_zero_indices]
# print(non_zero_values / 2)
# print(np.sum(abs(non_zero_values) > two_body_int_thresh))

# print(hamiltonian)
# print(initial_singles)
# print('initial guess with all non-zero elements:         ', mp2_val_red)
print('number of non zero params (SD):                   ', len(mp2_val_red))
# print('new excitation list:                              ', sd_ex_list)
print('total number of excitations:                       ', len(sd_ex_list))



initial_params_sd = initial_singles + mp2_val_red

sd_ex_list = singles_ex_list + doubles_ex_list


def custom_ex(num_spatial_orbitals, num_particles):
    return sd_ex_list


var_form = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations=custom_ex,
               qubit_mapper=mapper, initial_state=init_state)

final_excitation_list = var_form.excitation_list
# num_params = len(final_excitation_list)
fer_excitation_op = var_form.excitation_ops()  # getting the second_q operator for excitations in UCC
final_excitation_list_pauli = list()
for ex in fer_excitation_op:
    final_excitation_list_pauli.append(mapper.map(ex))

print(final_excitation_list)
# print('fermionic excitation operator from var_form --------------', fer_excitation_op[0])

# print(len(final_excitation_list))

# ******************************************************************************


print('--------------------------- Denominator calculation ---------------------------------------------')

den_vec = list()
for i in final_excitation_list:
    den = 0.0
    for j in i[0]:
        den += oe[j]
    for k in i[1]:
        den -= oe[k]
    den_vec.append(den)
#print('Denominator vector (D_mu): ', den_vec)
print(len(den_vec))
print(den_vec)


def denominator(ex_list):
    den_vec = list()
    for i in ex_list:
        den = 0.0
        for j in i[0]:
            den += oe[j]
        for k in i[1]:
            den -= oe[k]
        den_vec.append(den)
    return den_vec


print('-------------------------------------------------------------------------------------------------')

print('-----------------------* Inital State Calculation *------------------------------')
# hf_rev = [1,1,0,0,1,1,0,0]     #for H4
hf_rev = (list(np.concatenate((alpha_occupations, beta_occupations))))

hf_rev = [int(a) for a in hf_rev]
# print(type(hf_rev))
print(hf_rev)

import sparse
import scipy
import numpy as np
from scipy.sparse import coo_matrix

import numpy as np
import sparse


def find_large_elements_from_numpy(numpy_array: np.ndarray, threshold: float):
    """
    Finds elements in a dense NumPy array with absolute values greater than a
    specified threshold and sorts them in descending order.

    This function works by first converting the dense array to a sparse format
    to efficiently handle arrays that are mostly zero.

    Args:
        numpy_array (np.ndarray): The input NumPy array (can be of any dimension).
        threshold (float): The minimum absolute value for an element to be included
                           in the result.

    Returns:
        A tuple containing:
        - sorted_values (np.ndarray): A 1D array of the found values, sorted in
                                      descending order of their absolute values.
        - locations_list (list[list]): A list of lists, where each inner list is the
                                       coordinate of a found value. The list is
                                       ordered corresponding to sorted_values.
    """
    # 1. Convert the dense NumPy array to a sparse COO tensor.
    #    This step efficiently finds all non-zero elements and their coordinates.
    sparse_tensor = sparse.COO.from_numpy(numpy_array)

    # 2. Create a boolean mask to identify which non-zero elements have an
    #    absolute value greater than the threshold.
    mask = np.abs(sparse_tensor.data) > threshold

    # 3. Apply the mask to the data and coordinates of the sparse tensor.
    filtered_values = sparse_tensor.data[mask]
    filtered_coords = sparse_tensor.coords[:, mask]

    # 4. Get the indices that would sort the filtered values by their absolute value in descending order.
    if filtered_values.size > 0:
        abs_values = np.abs(filtered_values)
        sorted_indices = np.argsort(abs_values)[::-1]

        # 5. Apply the sorted indices to both the values and their coordinates.
        sorted_values = filtered_values[sorted_indices]
        sorted_coords_array = filtered_coords[:, sorted_indices]

        # 6. Convert the coordinates array into a list of lists.
        #    The array has shape (ndim, num_found), so we transpose it and convert to a list.
        locations_list = sorted_coords_array.T.tolist()
    else:
        # Handle case where no elements meet the threshold
        sorted_values = np.array([])
        locations_list = []

    return sorted_values, locations_list


from typing import List, Tuple, Any


def convert_coords_to_halved_tuples(coords_list: List[List[Any]]) -> List[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
    """
    Converts a list of lists into a list of tuples, where each tuple contains two halves of the original inner list.

    This function takes a list where each inner list has an even number of
    elements (e.g., [i, j, a, b, c, d]) and transforms it into a list of
    tuples where the coordinates are split in half,
    e.g., [((i, j, a), (b, c, d)), ...].

    Args:
        coords_list: A list of lists, where each inner list contains an even
                     number of elements.

    Returns:
        A new list where each item is a tuple containing two tuples,
        representing the two halves of the original inner list.
        Returns an empty list if the input is empty.

    Raises:
        ValueError: If any inner list does not contain an even number of elements.
    """
    result = []
    for coord in coords_list:
        # Check if the length of the inner list is even
        if len(coord) % 2 != 0:
            raise ValueError(
                f"Each coordinate list must contain an even number of elements to be halved, but found a list with length {len(coord)}: {coord}")

        # Find the midpoint of the list
        midpoint = len(coord) // 2

        # Create a tuple containing the two halves
        halved_tuple = (tuple(coord[:midpoint]), tuple(coord[midpoint:]))
        result.append(halved_tuple)

    return result


# print(t2_tensor1)
# print(t2_tensor1.shape)
# print(initial_singles)
# print(singles_ex_list)

dominant_t1_val = []
dominant_t1_ex_list = []
for i in range(len(initial_singles)):
    if abs(initial_singles[i]) > t1_thresh:
        dominant_t1_val.append(initial_singles[i])
        dominant_t1_ex_list.append(singles_ex_list[i])


dominant_t2_val = []
dominant_t2_ex_list = []
for i in range(len(mp2_val_red)):
    if abs(mp2_val_red[i]) > t2_thresh:
        dominant_t2_val.append(mp2_val_red[i])
        dominant_t2_ex_list.append(doubles_ex_list[i])


# print(dominant_t1_val)
# print(dominant_t1_ex_list)
# print(len(dominant_t1_ex_list))

dominant_t2, dominant_t2_excitations_list = find_large_elements_from_numpy(t2_tensor1, t2_thresh)

# print(dominant_t2_excitations_list)

dominant_t2_excitations_list = convert_coords_to_halved_tuples(dominant_t2_excitations_list)

# print(dominant_t2)
# print(dominant_t2_excitations_list)
# print(len(dominant_t2_excitations_list))

# # Convert each element to a standard Python float
# mp2_val_red = [float(x) for x in mp2_val_red]
# print(mp2_val_red)
# sorted_mp2 = sorted(mp2_val_red, key=abs)
# print(sorted_mp2)
# print(np.where(np.asarray(mp2_val_red)>1e-10)[0].size)

# print(t2_tensor1[1,5,2,6])
# print(t2_tensor1[4,5,6,7])

print(len(dominant_t2_excitations_list))

dominant_t2_tensor = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
for num in range(len(dominant_t2_excitations_list)):
    i = dominant_t2_ex_list[num][0]
    j = dominant_t2_ex_list[num][1]
    a = dominant_t2_ex_list[num][0]
    b = dominant_t2_ex_list[num][1]
    dominant_t2_tensor[i,j,a,b] = dominant_t2_val[num]


# print(dominant_t2_tensor)
# print(dominant_t2_tensor[1,5,2,6])
#
# triples = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))

# triples[i,j,k,a,b,c] = np.einsum('m,m',two_body_ints[i,j,a,:], dominant_t2_tensor[:,k,b,c])


# ----------------------- Construct and select dominant S operators


s_value = []
s_excitation = []



occ_idx_alpha = [i for i in range(num_alpha_particles)]
occ_idx_beta = [i+num_spatial_orbitals for i in range(num_beta_particles)]
occ_idx = occ_idx_alpha + occ_idx_beta
print('occ idx',occ_idx)

virt_idx_alpha = [i for i in range(num_alpha_particles,num_spatial_orbitals)]
virt_idx_beta = [i + num_spatial_orbitals for i in range(num_beta_particles,num_spatial_orbitals)]
virt_idx = virt_idx_alpha + virt_idx_beta
print('virt idx',virt_idx)
# exit()
#
# if molecule == 'H6':
#     # #H6
#     occ_idx = [0, 1, 2, 6, 7, 8]
#     virt_idx = [3, 4, 5, 9, 10, 11]
#
# if molecule == 'H2O':
#     # #H2O
#     occ_idx = [0, 1, 2, 3, 6, 7, 8, 9]
#     virt_idx = [4, 5, 10, 11]
#
# if molecule == 'N2':
#     # N2
#     occ_idx = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
#     virt_idx = [5, 6, 7, 13, 14, 15]
#
# if molecule == 'H4':
#     # H4
#     occ_idx = [0, 1, 4, 5]
#     virt_idx = [2, 3, 6, 7]




#--------------------------------------------------------------------------------------------------------------------------------



# #------------------------------ Symmetry space diagonalization
#
#
# print("-----------------------------Symmetry Space Diagonalization---------------------------")
# import numpy as np
# from itertools import combinations
# from scipy.linalg import eigh
#
#
# # n_spatial_orbitals = num_spatial_orbitals
# # n_spin_orbitals = 2*n_spatial_orbitals
# # n_alpha_electrons = 5
# # n_beta_electrons = 5
#
# def generate_spin_sector_determinants(n_orbitals, n_electrons):
#     """Generate all possible Slater determinants in the given spin-orbital space."""
#     return list(combinations(range(n_orbitals), n_electrons))
#
#
# def shift_tuples(input_list, shift_by):
#     return [tuple(x + shift_by for x in tup) for tup in input_list]
#
#
#
# def all_dets(n_spatial_orbs,n_alpha_e):
#     alpha_dets = generate_spin_sector_determinants(n_spatial_orbs, n_alpha_e)
#     #beta_dets = [(a + n_spatial_orbitals, b + n_spatial_orbitals) for (a, b) in alpha_dets]
#     beta_dets = shift_tuples(alpha_dets, n_spatial_orbs)
#     det_list = [(a, b) for a in alpha_dets for b in beta_dets]  # gives the occupied positions of the electrons
#     n_spin_orbs = n_spatial_orbs*2
#
#     all_dets_list = []
#     for occ in det_list:
#         true_vac = [0] * (n_spin_orbs)
#         for i in occ:
#             for j in i:
#                 true_vac[j] = 1
#         all_dets_list.append(true_vac[::-1])
#     return all_dets_list# det_list# beta_dets# all_dets_list
#
# # print (generate_spin_sector_determinants(6,3))
# #
#
# print('num_spatial_orbitals',num_spatial_orbitals)
# all_dets_symm = all_dets(num_spatial_orbitals,num_alpha_particles)
# print(len(all_dets_symm))
#
#
# det_list = [x[::-1] for x in all_dets_symm]
#
# from qiskit_addon_sqd.qubit import solve_qubit
# from qiskit_addon_sqd.qubit import sort_and_remove_duplicates
#
#
# det_list = sort_and_remove_duplicates(np.asarray(det_list))
# b = solve_qubit(np.asarray(det_list), qubit_op)
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy + frozen_energy_shift
# symmetry_space_diag_energy = eigen_values[0]
# print('Symmetry Space energy', symmetry_space_diag_energy)
# print(len(det_list))
# symmetry_subspace_ci_coeffs = b[1]
# symmetry_subspace_ci_coeffs = symmetry_subspace_ci_coeffs[:, 0].real
# print('Symmetry Space eigen vector', symmetry_subspace_ci_coeffs)
# num_symmetry_space_dominant_dets = np.where(np.abs(symmetry_subspace_ci_coeffs) > 1e-10)[0].size
# #print(np.where(np.abs(symmetry_subspace_ci_coeffs) > 1e-10)[0].size)
# print(num_symmetry_space_dominant_dets)



#--------------------------------------------------------------------------------------------------------------------------------























dominant_s_o_ex_list = []
dominant_s_o_val = []
s_thresh = s_thresh
for i in occ_idx:
    for j in occ_idx:
        for a in virt_idx:
            for m in occ_idx:
                # if j > i and j < m:
                if j > i:
                    temp_ex_list = [((i, j), (a, m))]

                    #denom_s = denominator(temp_ex_list)[0]
                    # s_val = np.divide(two_body_ints[i, j, a, m], denom_s)
                    s_val = two_body_ints[i, j, a, m]
                    # print('---------------------------------------------')
                    # print(s_val)
                    # print(temp_ex_list)
                    if abs(s_val) > s_thresh:
                        dominant_s_o_val.append(float(s_val))
                        dominant_s_o_ex_list.append(temp_ex_list[0])
                    # if abs(s_val) > s_thresh:
                    #     dominant_s_ex_list.append(temp_ex_list[0])
                    #     dominant_s_val.append(s_val)



dominant_s_o_twc_ex_list = [] #twc: two-way contractible
dominant_s_o_twc_val = []
s_thresh = s_thresh
for m in occ_idx:
    for k in occ_idx:
        for b in virt_idx:
            for n in occ_idx:
                # if j > i and j < m:
                # if j > i:
                temp_twc_ex_list = [((m, k), (b, n))]

                #denom_s = denominator(temp_ex_list)[0]
                # s_val = np.divide(two_body_ints[i, j, a, m], denom_s)
                s_val_twc = two_body_ints[m,k,b,n]
                # print('---------------------------------------------')
                # print(s_val)
                # print(temp_ex_list)
                if abs(s_val_twc) > s_thresh:
                    dominant_s_o_twc_val.append(float(s_val_twc))
                    dominant_s_o_twc_ex_list.append(temp_twc_ex_list[0])
                # if abs(s_val) > s_thresh:
                #     dominant_s_ex_list.append(temp_ex_list[0])
                #     dominant_s_val.append(s_val)

print(len(dominant_s_o_val))
print(len(dominant_s_o_twc_val))

# print(dominant_s_o_val)
# print(dominant_s_o_ex_list)
# print(len(dominant_s_o_val))

dominant_s_v_ex_list = []
dominant_s_v_val = []
for i in occ_idx:
    for e in virt_idx:
        for a in virt_idx:
            for b in virt_idx:
                # if j > i and j < m:
                if b > a:
                    temp_ex_list = [((i, e), (a, b))]

#                    denom_s = denominator(temp_ex_list)[0]
#                    s_val = np.divide(two_body_ints[i, j, a, m], denom_s)
                    s_val = two_body_ints[i, e, a, b]
                    # print('---------------------------------------------')
                    # print(s_val)
                    # print(temp_ex_list)
                    if abs(s_val) > s_thresh:
                        dominant_s_v_val.append(s_val)
                        dominant_s_v_ex_list.append(temp_ex_list[0])
                    # if abs(s_val) > s_thresh:
                    #     dominant_s_ex_list.append(temp_ex_list[0])
                    #     dominant_s_val.append(s_val)




dominant_s_v_twc_ex_list = []
dominant_s_v_twc_val = []
for j in occ_idx:
    for f in virt_idx:
        for e in virt_idx:
            for c in virt_idx:
                # if j > i and j < m:
                # if b > a:
                temp_twc_ex_list = [((j, f), (e, c))]

#                    denom_s = denominator(temp_ex_list)[0]
#                    s_val = np.divide(two_body_ints[i, j, a, m], denom_s)
                s_twc_val = two_body_ints[j, f, e, c]
                # print('---------------------------------------------')
                # print(s_val)
                # print(temp_ex_list)
                if abs(s_twc_val) > s_thresh:
                    dominant_s_v_twc_val.append(s_twc_val)
                    dominant_s_v_twc_ex_list.append(temp_twc_ex_list[0])
                # if abs(s_val) > s_thresh:
                #     dominant_s_ex_list.append(temp_ex_list[0])
                #     dominant_s_val.append(s_val)


print(len(dominant_s_v_val))
print(len(dominant_s_v_twc_val))



# print(len(dominant_s_o_val))
# print(len(dominant_s_v_val))

# dominant_triples = []
# dominant_triples_ex_list = []
#
# print(dominant_t2)
# print(dominant_t2_excitations_list)
# print(len(dominant_t2_excitations_list))
# print('-----------------------------------')
#
# print(dominant_s_o_ex_list)
# print(dominant_s_o_val)
#
# print(len(dominant_s_o_val))
#
# print('***********************')
#
# st_thresh = 1e-10
# # for tt in range(len(dominant_t2)):
# #     for ss in range(len(dominant_s_val)):
# #         st = dominant_t2[tt] * dominant_s_val[ss]
# #
# #         if abs(st) > st_thresh:
# #             print('-------------------------------------------------')
# #             print(st)
# #             print(dominant_t2_excitations_list[tt])
# #             print(dominant_s_ex_list[ss])
# #
#
#
# # cso = []
# # for ex_t2 in dominant_t2_excitations_list:
# #     for ex_s in dominant_s_ex_list:
# #         print(ex_t2)
# #         print(ex_s)
# #         temp_cso = set(ex_s[1]) & set(ex_t2[0])
# #         print(temp_cso)
# #         if len(temp_cso) >= 1:
# #             cso.append(temp_cso)
# #         print('-------------------------')
# # print(cso)
#
#
# dominant_t2_tensor = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
#
# counter = 0
# for ex_t2 in dominant_t2_excitations_list:
#     i = ex_t2[0][0]
#     j = ex_t2[0][1]
#     a = ex_t2[1][0]
#     b = ex_t2[1][1]
#     dominant_t2_tensor[i, j, a, b] = dominant_t2[counter]
#     counter += 1
#
# dominant_so_tensor = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
# counter_s = 0
# for ex_s in dominant_s_o_ex_list:
#     i = ex_s[0][0]
#     j = ex_s[0][1]
#     a = ex_s[1][0]
#     m = ex_s[1][1]
#     dominant_so_tensor[i, j, a, m] = dominant_s_o_val[counter_s]
#     counter_s += 1
#
# dominant_sv_tensor = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
# counter_s = 0
# for ex_s in dominant_s_v_ex_list:
#     i = ex_s[0][0]
#     j = ex_s[0][1]
#     a = ex_s[1][0]
#     m = ex_s[1][1]
#     dominant_sv_tensor[i, j, a, m] = dominant_s_v_val[counter_s]
#     counter_s += 1
#
#
#
# dominant_s_o_val = [float(x) for x in dominant_s_o_val]
# print(dominant_s_o_val)
#
# dominant_s_v_val = [float(x) for x in dominant_s_v_val]
# print(dominant_s_v_val)
#
# print(dominant_s_o_ex_list)
#
# large_s_o_val = []
# large_s_o_ex_list = []
# for i in range(len(dominant_s_o_val)):
#     if abs(dominant_s_o_val[i]) > 1e-10:
#         large_s_o_val.append(dominant_s_o_val[i])
#         large_s_o_ex_list.append(dominant_s_o_ex_list[i])
#
#
# large_s_v_val = []
# large_s_v_ex_list = []
# for i in range(len(dominant_s_v_val)):
#     if abs(dominant_s_v_val[i]) > 1e-10:
#         large_s_v_val.append(dominant_s_v_val[i])
#         large_s_v_ex_list.append(dominant_s_v_ex_list[i])
#
#
# print(large_s_o_val)
# print(large_s_o_ex_list)
# print(len(large_s_o_val))
#
#
# large_s_o_ex_list_array = np.array(large_s_o_ex_list)
#
# large_s_v_ex_list_array = np.array(large_s_v_ex_list)
#
# print(large_s_o_ex_list_array)
# # print(array_flat)
# # print("\nShape of the flattened array:", array_flat.shape)
# # print("Data type of the flattened array:", array_flat.dtype)
#
# large_t2_ex_list_array = np.array(dominant_t2_excitations_list)
# #print(large_t2_ex_list_array)
#
#
large_t2_val = []
large_t2_ex_list = []
for i in range(len(dominant_t2)):
    if abs(dominant_t2[i]) > t2_thresh:
        large_t2_val.append(dominant_t2[i])
        large_t2_ex_list.append(dominant_t2_excitations_list[i])

large_t2_val = [float(i) for i in large_t2_val]
# print(large_t2_val)
# print(large_t2_ex_list)
# print(len(large_t2_ex_list))



#---------------------


dominant_two_body_ints_val, dominant_two_body_ints_ex = find_large_elements_from_numpy(two_body_ints, two_body_int_thresh)

# print(dominant_two_body_ints_val)
# print(len(dominant_two_body_ints_val))
#
#
# print(dominant_two_body_ints_ex)
# print(len(dominant_two_body_ints_ex))

#original_list = [[3, 7, 3, 7], [3, 3, 3, 3], [7, 7, 7, 7], [7, 3, 7, 3]]

# For each sublist, create a tuple of two tuples
dominant_two_body_ints_ex_list = [((sublist[0], sublist[1]), (sublist[2], sublist[3])) for sublist in dominant_two_body_ints_ex]



# Use zip to pair the elements and convert the result to a list
combined_so = list(zip(dominant_s_o_ex_list, dominant_s_o_val))
# print(combined_so)
# print(combined_so[0][0][0])

combined_so_twc = list(zip(dominant_s_o_twc_ex_list, dominant_s_o_twc_val))

combined_sv = list(zip(dominant_s_v_ex_list, dominant_s_v_val))

combined_sv_twc = list(zip(dominant_s_v_twc_ex_list, dominant_s_v_twc_val))
# print(combined_sv)
# print(combined_sv[0][0][0])


combined_t2 = list(zip(large_t2_ex_list, large_t2_val))
print(combined_t2)
print(len(combined_t2))

# print(combined_t2[0][0][0])

combined_two_body_ints = list(zip(dominant_two_body_ints_ex_list, dominant_two_body_ints_val))
# print(combined_two_body_ints)
# print(combined_two_body_ints[0][0][0])
# print(len(combined_two_body_ints))


print('---------Optimized Einsum-like Implementation Starts-------')
#--------------------------------------------------

from collections import defaultdict
import pprint

def _flatten_tuple(nested_tuple):
    """A helper function to flatten a nested tuple structure."""
    for item in nested_tuple:
        if isinstance(item, tuple):
            yield from _flatten_tuple(item)
        else:
            yield item

def group_by_relaxed_index(data_list, relaxed_index):
    """
    Groups items by their position, excluding one "relaxed" index.

    Args:
        data_list: The list of (position, value) tuples.
        relaxed_index: The integer index in the flattened position tuple
                       that should be allowed to vary.

    Returns:
        A list of lists, where each inner list is a group of items. Output looks like [[(((0, 1), (2, 1)), -0.010374291003186724)], [(((0, 1), (3, 0)), -0.03462545930279119)], [(((0, 4), (2, 4)), 0.08890047334127417)], [(((0, 4), (3, 5)), -0.02496417943457847)], [(((0, 5), (2, 5)), -0.010374291003186724)], [(((0, 5), (3, 4)), -0.03462545930279119)], [(((1, 4), (2, 5)), -0.10660039029159216)], [(((1, 4), (3, 4)), -0.06787975147535666)], [(((1, 5), (2, 4)), -0.22378591426017683)], [(((1, 5), (3, 5)), -0.0033283741423969443)], [(((4, 5), (6, 5)), -0.010374291003186724)], [(((4, 5), (7, 4)), -0.03462545930279119)]]
    """
    groups = defaultdict(list)

    for item in data_list:
        position_tuple, value = item

        # Flatten the nested position tuple into a single-level tuple
        flat_pos = tuple(_flatten_tuple(position_tuple))

        if not 0 <= relaxed_index < len(flat_pos):
            # Handle cases where the index is out of bounds
            # For simplicity, we'll just skip such items
            continue

        # Create the key by excluding the element at the relaxed_index
        key = tuple(el for i, el in enumerate(flat_pos) if i != relaxed_index)

        groups[key].append(item)

    return list(groups.values())


# print(combined_sv)
import pprint
final_list_so = group_by_relaxed_index(combined_so,3)
pprint.pprint(final_list_so)

final_list_sv = group_by_relaxed_index(combined_sv,1)
#final_list_two_body_ints = group_by_relaxed_index(combined_two_body_ints,3)
final_list_t2 = group_by_relaxed_index(combined_t2, 0)
print('-----------')
pprint.pprint(combined_t2)
print(len(combined_t2))
print('---------------------')
pprint.pprint(final_list_t2)
print(len(final_list_t2))
# exit()
final_list_t2v = group_by_relaxed_index(combined_t2, 2)
#print(final_list_so)


import numpy as np



# einsummed_list = []
# for fixed_so in final_list_so:
#     for fixed_t2 in final_list_t2:
#         local_sum = 0.0
#         for pos1, val1 in fixed_so:
#             for pos2, val2 in fixed_t2:
#                 (i, j), (a, m) = pos1
#                 (p, k), (b, c) = pos2
#                 if m == p:
#                     if k > j > i and c > b > a:
#                         # 4. Construct the new element and add it to the result list
#                         new_pos = ((i, j, k), (a, b, c))
#                         new_val = val1 * val2
#                         local_sum += new_val
#         if abs(local_sum) > 1e-10:
#             einsummed_list.append((new_pos, local_sum))
# triples_so_t2o = einsummed_list.copy()
# print(einsummed_list)
# print(len(einsummed_list))
#
#
#
# einsummed_list = []
# for fixed_sv in final_list_sv:
#     for fixed_t2v in final_list_t2v:
#         local_sum = 0.0
#         for pos1, val1 in fixed_sv:
#             for pos2, val2 in fixed_t2v:
#                 (i, e), (a, b) = pos1
#                 (j, k), (f, c) = pos2
#                 if e == f:
#                     if k > j > i and c > b > a:
#                         # 4. Construct the new element and add it to the result list
#                         new_pos = ((i, j, k), (a, b, c))
#                         new_val = val1 * val2
#                         local_sum += new_val
#         if abs(local_sum) > 1e-10:
#             einsummed_list.append((new_pos, local_sum))
# triples_sv_t2v = einsummed_list.copy()
# print(einsummed_list)
# print(len(einsummed_list))



#--------------------

def _is_strictly_increasing(tup):
    """Helper function to check if elements in a tuple are strictly increasing."""
    return all(tup[i] < tup[i + 1] for i in range(len(tup) - 1))








#-------------------


def merge_and_add_tuples(list1, list2):
    """
    Merges two lists of (position, value) tuples.
    If positions match, values are multiplied. Otherwise, items are kept.
    """
    # Use the position as a key and the value as the dictionary's value
    merged_data = {}

    # 1. Process the first list, adding all its items to the dictionary
    for position, value in list1:
        merged_data[position] = value

    # 2. Process the second list
    for position, value in list2:
        # Check if the position from list2 already exists in our dictionary
        if position in merged_data:
            # If it exists, add the existing value by the new value
            merged_data[position] += value
        else:
            # If it's a new position, add it to the dictionary
            merged_data[position] = value

    # 3. Convert the final dictionary back to the desired list format
    result_list = [(position, value) for position, value in merged_data.items()]

    return result_list



import numpy as np
import pprint

s_t2_thresh = inp.s_t2_thresh

def s_t2_contracted_via_v(grouped_list1v, grouped_list2v):
    einsummed_listv = []
    for fixed_sv in grouped_list1v:
        for fixed_t2v in grouped_list2v:
            local_sumv = 0.0
            for pos1v, val1v in fixed_sv:
                for pos2v, val2v in fixed_t2v:
                    tuple1_Av, tuple1_Bv = pos1v  # (i,e), (a,b)
                    tuple2_Av, tuple2_Bv = pos2v  # (j,k), (e,c)
                    # print(tuple1_A)
                    # print('tuple1_B', tuple1_Bv[-1])
                    # print('tuple2_A', tuple2_Av[0])
                    # print('----------------')
                    new_tuple_Av = tuple1_Av[:-1] + tuple2_Av
                    new_tuple_Bv = tuple1_Bv + tuple2_Bv[1:]
                    # new_pos = (new_tuple_A, new_tuple_B)
                    # print(tuple2_B)
                    if tuple1_Bv and tuple2_Av and tuple1_Av[-1] == tuple2_Bv[0]:
                        if set(new_tuple_Av).issubset(occ_idx) and set(new_tuple_Bv).issubset(virt_idx) and set((tuple1_Av[-1], tuple2_Bv[0])).issubset(virt_idx):
                            if _is_strictly_increasing(new_tuple_Av) and _is_strictly_increasing(new_tuple_Bv):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_posv = (new_tuple_Av, new_tuple_Bv)
                                new_valv = val1v * val2v
                                local_sumv += new_valv
            if abs(local_sumv) > s_t2_thresh:
                einsummed_listv.append((new_posv, local_sumv))
    return einsummed_listv


def s_t2_contracted_via_v_relaxed_v(grouped_list1v, grouped_list2v):
    einsummed_listv = []
    for fixed_sv in grouped_list1v:
        for fixed_t2v in grouped_list2v:
            local_sumv = 0.0
            for pos1v, val1v in fixed_sv:
                for pos2v, val2v in fixed_t2v:
                    tuple1_Av, tuple1_Bv = pos1v  # (i,e), (a,b)
                    tuple2_Av, tuple2_Bv = pos2v  # (j,k), (f,g)
                    # print(tuple1_A)
                    # print('tuple1_B', tuple1_Bv[-1])
                    # print('tuple2_A', tuple2_Av[0])
                    # print('----------------')
                    new_tuple_Av = tuple1_Av[:-1] + tuple2_Av
                    new_tuple_Bv = tuple1_Bv + tuple2_Bv[1:]
                    # new_pos = (new_tuple_A, new_tuple_B)
                    # print(tuple2_B)
                    if tuple1_Bv and tuple2_Av and tuple1_Av[-1] == tuple2_Bv[0]:
                        if set(new_tuple_Av).issubset(occ_idx) and set(new_tuple_Bv).issubset(virt_idx) and set((tuple1_Av[-1] ,tuple2_Bv[0])).issubset(virt_idx):
                            if _is_strictly_increasing(new_tuple_Av) and _is_strictly_increasing(new_tuple_Bv[:-1]):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_posv = (new_tuple_Av, new_tuple_Bv)
                                new_valv = val1v * val2v
                                local_sumv += new_valv
            if abs(local_sumv) > s_t2_thresh:
                einsummed_listv.append((new_posv, local_sumv))
    return einsummed_listv


def s_t2_contracted_via_o(grouped_list1o, grouped_list2o):
    einsummed_listo = []
    for fixed_so in grouped_list1o:
        for fixed_t2 in grouped_list2o:
            local_sumo = 0.0
            for pos1o, val1o in fixed_so:
                for pos2o, val2o in fixed_t2:
                    tuple1_Ao, tuple1_Bo = pos1o  # (i,j), (a,m)----(i,j,k), (a,b,n)
                    tuple2_Ao, tuple2_Bo = pos2o  # (m,k), (b,c)----(n,l),(c,d)
                    # print(tuple1_A)
                    # print('tuple1_B', tuple1_Bo[-1])
                    # print('tuple2_A', tuple2_Ao[0])
                    # print('----------------')
                    new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:]
                    new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo
                    # new_pos = (new_tuple_A, new_tuple_B)
                    # print(tuple2_B)
                    if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                        if set(new_tuple_Ao).issubset(occ_idx) and set(new_tuple_Bo).issubset(virt_idx) and set((tuple1_Bo[-1],tuple2_Ao[0])).issubset(occ_idx):
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_poso = (new_tuple_Ao, new_tuple_Bo)
                                new_valo = val1o * val2o
                                local_sumo += new_valo
            if abs(local_sumo) > s_t2_thresh:
                einsummed_listo.append((new_poso, local_sumo))
    return einsummed_listo



def s_t2_contracted_via_o_relaxed_o(grouped_list1o, grouped_list2o):
    einsummed_listo = []
    for fixed_so in grouped_list1o:
        for fixed_t2 in grouped_list2o:
            local_sumo = 0.0
            for pos1o, val1o in fixed_so:
                for pos2o, val2o in fixed_t2:
                    tuple1_Ao, tuple1_Bo = pos1o  # (i,j), (a,m)----(i,j,k), (a,b,n)
                    tuple2_Ao, tuple2_Bo = pos2o  # (m,n), (b,c)----(n,l),(c,d)
                    # print(tuple1_A)
                    # print('tuple1_B', tuple1_Bo[-1])
                    # print('tuple2_A', tuple2_Ao[0])
                    # print('----------------')
                    new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:]
                    new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo
                    # new_pos = (new_tuple_A, new_tuple_B)
                    # print(tuple2_B)
                    if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                        # if set(tuple1_Ao).issubset(occ_idx) and set(tuple2_Ao).issubset(occ_idx) and set(tuple1_Bo[:-1]).issubset(virt_idx) and set(tuple2_Bo).issubset(virt_idx) and tuple1_Bo[-1] in occ_idx:
                        if set(new_tuple_Ao).issubset(occ_idx) and set(new_tuple_Bo).issubset(virt_idx) and set((tuple1_Bo[-1],tuple2_Ao[0])).issubset(occ_idx):
                            if _is_strictly_increasing(new_tuple_Ao[:-1]) and _is_strictly_increasing(new_tuple_Bo):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_poso = (new_tuple_Ao, new_tuple_Bo)
                                new_valo = val1o * val2o
                                local_sumo += new_valo
            if abs(local_sumo) > s_t2_thresh:
                einsummed_listo.append((new_poso, local_sumo))
    return einsummed_listo



def s_t2_contracted_via_o_relaxed_v(grouped_list1o, grouped_list2o):
    einsummed_listo = []
    for fixed_so in grouped_list1o:
        for fixed_t2 in grouped_list2o:
            local_sumo = 0.0
            for pos1o, val1o in fixed_so:
                for pos2o, val2o in fixed_t2:
                    # tuple1_Ao, tuple1_Bo = pos1o  # (i,j), (a,m)----(i,j,k), (a,b,n)
                    # tuple2_Ao, tuple2_Bo = pos2o  # (m,k), (b,c)----(n,l),(c,d)

                    (i,j) ,(a,m) = pos1o
                    (p,k), (b,f) = pos2o

                    # new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:]
                    # new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo
                    new_tuple_Ao = (i,j,k)
                    new_tuple_Bo = (a,b,f)

                    # if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                    if m == p:
                        # if set((i,j)).issubset(occ_idx) and set((p,k)).issubset(occ_idx) and set((a)).issubset(virt_idx) and set((b,f)).issubset(virt_idx) and m in occ_idx:
                        if set((i,j,k)).issubset(occ_idx) and set((a,b,f)).issubset(virt_idx) and set((m,p)).issubset(occ_idx):
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo[:-1]):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_poso = (new_tuple_Ao, new_tuple_Bo)
                                new_valo = val1o * val2o
                                local_sumo += new_valo
            if abs(local_sumo) > s_t2_thresh:
                einsummed_listo.append((new_poso, local_sumo))
    return einsummed_listo






print('Triples Generation using function')
def pt_triples_generator_via_st2(ex_val_combined_so, ex_val_combined_sv, ex_val_combined_t2):
    grouped_list_so = group_by_relaxed_index(ex_val_combined_so,3)
    grouped_list_sv = group_by_relaxed_index(ex_val_combined_sv,1)
    grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2,0)
    grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2,2)

    so_t2o_contracted = s_t2_contracted_via_o(grouped_list_so,grouped_list_t2o)
    sv_t2v_contracted = s_t2_contracted_via_v(grouped_list_sv, grouped_list_t2v)

    t3 = merge_and_add_tuples(so_t2o_contracted, sv_t2v_contracted)

    return t3


# dominant_triples_ex_val_list = pt_triples_generator_via_st2(combined_so, combined_sv, combined_t2)
# dominant_triples_ex_list = [item[0] for item in dominant_triples_ex_val_list]

# print(dominant_triples_ex_val_list)
# print(len(dominant_triples_ex_val_list))
#
# print('==========================')
# print(final_list_so)

final_list_so_inter = group_by_relaxed_index(combined_so,0)
final_list_two_body = group_by_relaxed_index(combined_two_body_ints,0)
#pprint.pprint(final_list_so)


my_tuple = (9, 1, 2)
#allowed_elements = [0, 1, 2, 6, 7, 8]



s_s_thresh = inp.s_s_thresh
def so_so_contraction(so_list,v_list):
    '''

    :param so_list: Grouped s_o list
    :param v_list: Grouped v list
    :return:
    '''
    einsummed_list = []
    for fixed_so1 in so_list:
        for fixed_so2 in v_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
                    tuple1_Ao, tuple1_Bo = pos1  # (i,j), (a,m)
                    tuple2_Ao, tuple2_Bo = pos2  # (p,k), (b,n)
                    # (i, j), (a, m) = pos1
                    # (p, k), (b, n) = pos2
                    new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:] #i,j,k
                    new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo   #a,b,n
                    if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                        # if set(tuple2_Ao).issubset(occ_idx) and set(tuple2_Bo[:-1]).issubset(virt_idx) and tuple2_Bo[-1] in occ_idx:
                        if set(new_tuple_Ao).issubset(occ_idx) and set(new_tuple_Bo[:-1]).issubset(virt_idx) and set((tuple1_Bo[-1],tuple2_Ao[0], new_tuple_Bo[-1])).issubset(occ_idx):
                        # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo[:-1]):
#                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                #new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))
    return einsummed_list



def so_t2oso_contraction(so_list,v_list):
    '''

    :param so_list: Grouped s_o list
    :param v_list: so_t2o
    :return:

    '''
    einsummed_list = []
    for fixed_so1 in so_list:
        for fixed_so2 in v_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
                    # tuple1_Ao, tuple1_Bo = pos1  # (k,l), (c,n)
                    # tuple2_Ao, tuple2_Bo = pos2  # (i,j,n), (a,b,d)
                    (k,l), (c,m) = pos1  # (k,l), (c,n)
                    #(k, l), (m, d) = pos1
                    (i,j,p), (a,b,d) = pos2  # (i,j,n), (a,b,d)
                    #(i, j, p), (a, b, c) = pos2

                    new_tuple_Ao = (i,j,k,l) # tuple2_Ao[:-1] + tuple1_Ao #
                    new_tuple_Bo = (a,b,c,d) #tuple2_Bo + tuple1_Bo[1:] #
                    # print(m)
                    # print(p)

                    # if tuple1_Bo and tuple2_Ao and tuple1_Bo[0] == tuple2_Ao[-1]:
                    if m == p:
                        # print('here')
                        # print('**')
                        # print([i,j,k,l])
                        # print(occ_idx)
                        # print([a,b,c,d])
                        # print(virt_idx)
                        # print('**')
                        # if set(tuple1_Ao).issubset(occ_idx) and set(tuple2_Ao).issubset(occ_idx) and set(tuple2_Bo).issubset(virt_idx) and set(tuple1_Bo[:-1]).issubset(occ_idx) and tuple1_Bo[-1] in virt_idx:
                        if set((i,j,k,l)).issubset(occ_idx) and set((a,b,c,d)).issubset(virt_idx) and set((m,p)).issubset(occ_idx):
                        # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):

#                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                #new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2

                                local_sum += new_val
            #print('-----------------------')
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))
    return einsummed_list



# def so_so_contraction(so_list,v_list):
#     # trying so so contraction
#     einsummed_list = []
#     for fixed_so1 in so_list:
#         for fixed_so2 in v_list:
#             local_sum = 0.0
#             for pos1, val1 in fixed_so1:
#                 for pos2, val2 in fixed_so2:
#                     (i, j), (a, m) = pos1
#                     (p, k), (b, n) = pos2
#                     if m == p:
#                         if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
#                             if k > j > i and b > a:
#                                 # 4. Construct the new element and add it to the result list
#                                 new_pos = ((i, j, k), (a, b, n))
#                                 new_val = val1 * val2
#                                 local_sum += new_val
#             if abs(local_sum) > 1e-10:
#                 einsummed_list.append((new_pos, local_sum))
#     return einsummed_list


def sv_sv_contraction(sv_list,v_list):
    # trying so so contraction
    einsummed_list = []
    for fixed_so1 in sv_list:
        for fixed_so2 in v_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
                    tuple1_Ao, tuple1_Bo = pos1  # (i,e), (a,b)
                    tuple2_Ao, tuple2_Bo = pos2  # (j,g), (f,c)
                    # (i, j), (a, m) = pos1
                    # (p, k), (b, n) = pos2
                    new_tuple_Ao = tuple1_Ao[:-1] + tuple2_Ao
                    new_tuple_Bo = tuple1_Bo + tuple2_Bo[1:]
                    if tuple1_Bo and tuple2_Ao and tuple1_Ao[-1] == tuple2_Bo[0]:
                        if set(tuple2_Bo).issubset(virt_idx) and set(tuple2_Ao[:-1]).issubset(occ_idx) and tuple2_Ao[-1] in virt_idx:
                            # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao[:-1]) and _is_strictly_increasing(new_tuple_Bo):
                                #                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                # new_pos = ((i, j, g), (a, b, c))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))
    return einsummed_list

# print('jh')
# exit()


def sv_t2vsv_contraction(svt2_list,v_list):
    '''

    :param sv_list:
    :param v_list:
    :return:
    '''
    einsummed_list = []
    for fixed_so1 in svt2_list:
        for fixed_so2 in v_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
                    # tuple1_Ao, tuple1_Bo = pos1  # (k,f), (c,d)
                    # tuple2_Ao, tuple2_Bo = pos2  # (i,j,k), (a,b,f)
                    (k,f), (c,d) = pos1  # (k,f), (c,d)
                    (i,j,l), (a,b,g) = pos2  # (i,j,k), (a,b,f)
                    # (i, j), (a, m) = pos1
                    # (p, k), (b, n) = pos2
                    new_tuple_Ao = (i,j,k,l) #tuple2_Ao + tuple1_Ao[1:]
                    new_tuple_Bo = (a,b,c,d) #tuple2_Bo[:-1] + tuple1_Bo
                    # if tuple1_Ao and tuple2_Bo and tuple1_Ao[0] == tuple2_Bo[-1]:
                    if f == g:
                        if set((i,j,k,l)).issubset(occ_idx) and set((a,b,c,d)).issubset(virt_idx) and set((f,g)).issubset(virt_idx):
                            # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):
                                #                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                # new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))
    return einsummed_list




def sv_sot2_contraction(sot2_list,v_list):
    einsummed_list = []
    for fixed_so1 in sot2_list:
        for fixed_so2 in v_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
#                    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                    # tuple1_Ao, tuple1_Bo = pos1  # (f,l), (c,d)
                    # tuple2_Ao, tuple2_Bo = pos2  # (i,j,k), (a,b,f)
                    (k, f), (c, d) = pos1
                    (i,j,l), (a, b, g) = pos2
                    # print("===========")
                    # print(f)
                    # print(g)
                    # print("==============")
                    new_tuple_Ao = (i,j,k,l) #tuple2_Ao + tuple1_Ao[1:]
                    new_tuple_Bo = (a,b,c,d) #tuple2_Bo[:-1] + tuple1_Bo
                    # if tuple1_Ao and tuple2_Bo and tuple1_Ao[0] == tuple2_Bo[-1]:
                    if f == g:
                        if set(new_tuple_Ao).issubset(occ_idx) and set(new_tuple_Bo).issubset(virt_idx) and set((f,g)).issubset(virt_idx):
                            # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):
                                #                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                # new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))
    return einsummed_list














#------------------- New Quadruples diagrams


def s_s_contracted_via_o_relaxed_v(so_list,s_twc_list):
    einsummed_list = []
    for fixed_so1 in so_list:
        for fixed_so2 in s_twc_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
    #                    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                    # tuple1_Ao, tuple1_Bo = pos1  # (f,l), (c,d)
                    # tuple2_Ao, tuple2_Bo = pos2  # (i,j,k), (a,b,f)
                    (i,j), (a,m) = pos1
                    (n,e), (b,c) = pos2
                    # print("===========")
                    # print(f)
                    # print(g)
                    # print("==============")
                    new_tuple_Ao = (i,j,e) #tuple2_Ao + tuple1_Ao[1:]
                    new_tuple_Bo = (a,b,c) #tuple2_Bo[:-1] + tuple1_Bo
                    # if tuple1_Ao and tuple2_Bo and tuple1_Ao[0] == tuple2_Bo[-1]:
                    if m == n:
                        if set(new_tuple_Ao[:-1]).issubset(occ_idx) and set(new_tuple_Bo).issubset(virt_idx) and set((m,n)).issubset(occ_idx):
                            # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao[:-1]) and _is_strictly_increasing(new_tuple_Bo):
                                #                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                # new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))

    return einsummed_list



def s_s_contracted_via_v_relaxed_o(sv_list,s_twc_list):
    einsummed_list = []
    for fixed_so1 in sv_list:
        for fixed_so2 in s_twc_list:
            local_sum = 0.0
            for pos1, val1 in fixed_so1:
                for pos2, val2 in fixed_so2:
    #                    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                    # tuple1_Ao, tuple1_Bo = pos1  # (f,l), (c,d)
                    # tuple2_Ao, tuple2_Bo = pos2  # (i,j,k), (a,b,f)
                    (i,e), (a,b) = pos1
                    (j,k), (f,m) = pos2
                    # print("===========")
                    # print(f)
                    # print(g)
                    # print("==============")
                    new_tuple_Ao = (i,j,k) #tuple2_Ao + tuple1_Ao[1:]
                    new_tuple_Bo = (a,b,m) #tuple2_Bo[:-1] + tuple1_Bo
                    # if tuple1_Ao and tuple2_Bo and tuple1_Ao[0] == tuple2_Bo[-1]:
                    if e == f:
                        if set(new_tuple_Ao).issubset(occ_idx) and set(new_tuple_Bo[:-1]).issubset(virt_idx) and set((e,f)).issubset(virt_idx):
                            # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo[:-1]):
                                #                            if k > j > i and b > a:
                                # 4. Construct the new element and add it to the result list
                                # new_pos = ((i, j, k), (a, b, n))
                                new_pos = (new_tuple_Ao, new_tuple_Bo)
                                new_val = val1 * val2
                                local_sum += new_val
            if abs(local_sum) > s_s_thresh:
                einsummed_list.append((new_pos, local_sum))

    return einsummed_list



def s_t2_contracted_via_o_relaxed_v1(grouped_list1o, grouped_list2o):

    '''

    :param grouped_list1o: s_o
    :param grouped_list2o: t2
    :return:
    '''
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.',grouped_list2o)
    einsummed_listo = []
    for fixed_so in grouped_list1o:
        for fixed_t2 in grouped_list2o:
            local_sumo = 0.0
            for pos1o, val1o in fixed_so:
                for pos2o, val2o in fixed_t2:
                    # tuple1_Ao, tuple1_Bo = pos1o  # (i,j), (a,m)----(i,j,k), (a,b,n)
                    # tuple2_Ao, tuple2_Bo = pos2o  # (m,k), (b,c)----(n,l),(c,d)

                    (i,k) ,(a,m) = pos1o
                    (j,n), (b,e) = pos2o

                    # new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:]
                    # new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo
                    new_tuple_Ao = (i,j,k)
                    new_tuple_Bo = (a,b,e)

                    # if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                    if m == n:
                        # if set((i,j)).issubset(occ_idx) and set((p,k)).issubset(occ_idx) and set((a)).issubset(virt_idx) and set((b,f)).issubset(virt_idx) and m in occ_idx:
                        if set((i,j,k)).issubset(occ_idx) and set((a,b,e)).issubset(virt_idx) and set((m,n)).issubset(occ_idx):
                            if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):# _is_strictly_increasing(new_tuple_Bo[:-1]):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_poso = (new_tuple_Ao, new_tuple_Bo)
                                new_valo = val1o * val2o
                                local_sumo += new_valo
                                # print('-----------------------')
                                # print(new_poso)
                                # print(new_valo)
                                # print('-------------------------')
            if abs(local_sumo) > s_t2_thresh:
                einsummed_listo.append((new_poso, local_sumo))
    return einsummed_listo


def s_st_contraction_via_o(grouped_list1o, grouped_list2o):
    einsummed_listo = []
    for fixed_so in grouped_list1o:
        for fixed_t2 in grouped_list2o:
            local_sumo = 0.0
            for pos1o, val1o in fixed_so:
                for pos2o, val2o in fixed_t2:
                    # tuple1_Ao, tuple1_Bo = pos1o  # (i,j), (a,m)----(i,j,k), (a,b,n)
                    # tuple2_Ao, tuple2_Bo = pos2o  # (m,k), (b,c)----(n,l),(c,d)

                    (i,j,k) ,(a,b,e) = pos1o
                    (l,f), (d,c) = pos2o
                    # print('****************')
                    # print(e)
                    # print(f)
                    # print('****************')
                    # new_tuple_Ao = tuple1_Ao + tuple2_Ao[1:]
                    # new_tuple_Bo = tuple1_Bo[:-1] + tuple2_Bo
                    new_tuple_Ao = (i,j,k,l)
                    new_tuple_Bo = (a,b,c,d)

                    # if tuple1_Bo and tuple2_Ao and tuple1_Bo[-1] == tuple2_Ao[0]:
                    if e == f:
                        # print('here')
                        # print(new_tuple_Ao)
                        # print(new_tuple_Bo)
                        # if set((i,j)).issubset(occ_idx) and set((p,k)).issubset(occ_idx) and set((a)).issubset(virt_idx) and set((b,f)).issubset(virt_idx) and m in occ_idx:
                        if set((i,j,k,l)).issubset(occ_idx) and set((a,b,c,d)).issubset(virt_idx) and set((e,f)).issubset(virt_idx):
                           if _is_strictly_increasing(new_tuple_Ao) and _is_strictly_increasing(new_tuple_Bo):
                                #print('here')
                                # 4. Construct the new element and add it to the result list
                                #                        new_pos = ((i, j, k), (a, b, c))
                                new_poso = (new_tuple_Ao, new_tuple_Bo)
                                new_valo = val1o * val2o
                                local_sumo += new_valo
                                print(local_sumo)
            if abs(local_sumo) > s_t2_thresh:
                einsummed_listo.append((new_poso, local_sumo))
    return einsummed_listo




















# einsummed_list = []
# for fixed_so1 in final_list_sv:
#     for fixed_so2 in final_list_two_body:
#         local_sum = 0.0
#         for pos1, val1 in fixed_so1:
#             for pos2, val2 in fixed_so2:
#                 # tuple1_Ao, tuple1_Bo = pos1  # (i,e), (a,b)
#                 # tuple2_Ao, tuple2_Bo = pos2  # (j,g), (f,c)
#                 (i, e), (a, b) = pos1
#                 (j, g), (f, c) = pos2
#                 # new_tuple_Ao = tuple1_Ao[:-1] + tuple2_Ao
#                 # new_tuple_Bo = tuple1_Bo + tuple2_Bo[1:]
#                 if tuple1_Bo and tuple2_Ao and tuple1_Ao[-1] == tuple2_Bo[0]:
#                     if set(tuple2_Bo).issubset(virt_idx) and set(tuple2_Ao[:-1]).issubset(occ_idx) and tuple2_Ao[-1] in virt_idx:
#                         # if p in occ_idx and k in occ_idx and n in occ_idx and b in virt_idx:
#                         if _is_strictly_increasing(new_tuple_Ao[:-1]) and _is_strictly_increasing(new_tuple_Bo):
#                             #                            if k > j > i and b > a:
#                             # 4. Construct the new element and add it to the result list
#                             # new_pos = ((i, j, g), (a, b, c))
#                             new_pos = (new_tuple_Ao, new_tuple_Bo)
#                             new_val = val1 * val2
#                             local_sum += new_val
#         if abs(local_sum) > 1e-10:
#             einsummed_list.append((new_pos, local_sum))



# triples_so_so = so_so_contraction(final_list_so, final_list_two_body)
# pprint.pprint(triples_so_so)
# print(len(triples_so_so))
# # print(len(einsummed_list))
#
#
# xxx = group_by_relaxed_index(triples_so_so,5)
# t4 = s_t2_contracted_via_o(xxx, final_list_t2)
# print(t4)
# print(len(t4))

#------------------------------------------* Quadruples Generation *-------------------------------------------------------------------------------------------------------------------------

# def pt_quadruples_generator_via_st2(ex_val_combined_so, ex_val_combined_sv,v_two_body,ex_val_combined_t2):
#     grouped_list_so = group_by_relaxed_index(ex_val_combined_so,3)
#     grouped_list_sv = group_by_relaxed_index(ex_val_combined_sv,1)
#     grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2,0)
#     grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2,2)
#     grouped_list_two_body_o = group_by_relaxed_index(v_two_body,0)
#     grouped_list_two_body_v = group_by_relaxed_index(v_two_body,2)
#
#     t3_so_so = so_so_contraction(grouped_list_so, grouped_list_two_body_o)
#     grouped_list_t3_so_so = group_by_relaxed_index(t3_so_so, 5)
#
#     t3_sv_sv = sv_sv_contraction(grouped_list_sv, grouped_list_two_body_v)
#     grouped_list_t3_sv_sv = group_by_relaxed_index(t3_sv_sv, 2)
# #    print(grouped_list_t3_sv_sv)
#
#
#     t3_t2o_contracted = s_t2_contracted_via_o(grouped_list_t3_so_so, grouped_list_t2o)
#     #print('t3_t2o_contracted',t3_t2o_contracted)
#     t3_t2v_contracted = s_t2_contracted_via_v(grouped_list_t3_sv_sv, grouped_list_t2v)
# #    print('t3_t2v_contracted', t3_t2v_contracted)
#     t4 = merge_and_add_tuples(t3_t2o_contracted, t3_t2v_contracted)
#
#     return t4


# def pt_quadruples_generator_via_st2(ex_val_combined_so, ex_val_combined_sv,v_two_body_o, v_two_body_v,ex_val_combined_t2):
def pt_quadruples_generator_via_st2(ex_val_combined_so, ex_val_combined_sv, v_two_body, ex_val_combined_t2):
    grouped_list_so = group_by_relaxed_index(ex_val_combined_so,3)
    grouped_list_sv = group_by_relaxed_index(ex_val_combined_sv,1)
    grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2,0)
    grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2,2)
    # grouped_list_two_body_o = group_by_relaxed_index(v_two_body_o,0)
    # grouped_list_two_body_v = group_by_relaxed_index(v_two_body_v,2)
    grouped_list_two_body_o = group_by_relaxed_index(v_two_body,0)
    grouped_list_two_body_v = group_by_relaxed_index(v_two_body,2)

    t3_so_so = so_so_contraction(grouped_list_so, grouped_list_two_body_o)
    grouped_list_t3_so_so = group_by_relaxed_index(t3_so_so, 5)

    t3_sv_sv = sv_sv_contraction(grouped_list_sv, grouped_list_two_body_v)
    grouped_list_t3_sv_sv = group_by_relaxed_index(t3_sv_sv, 2)
#    print(grouped_list_t3_sv_sv)


    t3_t2o_contracted = s_t2_contracted_via_o(grouped_list_t3_so_so, grouped_list_t2o)
    #print('t3_t2o_contracted',t3_t2o_contracted)
    t3_t2v_contracted = s_t2_contracted_via_v(grouped_list_t3_sv_sv, grouped_list_t2v)
#    print('t3_t2v_contracted', t3_t2v_contracted)
    t4 = merge_and_add_tuples(t3_t2o_contracted, t3_t2v_contracted)



    s_s_co_rv = s_s_contracted_via_o_relaxed_v(grouped_list_so,grouped_list_two_body_o)
    s_s_co_rv1 = group_by_relaxed_index(s_s_co_rv,2)

    s_s_cv_ro = s_s_contracted_via_v_relaxed_o(grouped_list_sv,grouped_list_two_body_v)
    s_s_cv_ro1 = group_by_relaxed_index(s_s_cv_ro,5)

    s_s_co_rv_t2_cv = s_t2_contracted_via_v(s_s_co_rv1, grouped_list_t2v)
    s_s_cv_ro_t2_co = s_t2_contracted_via_o(s_s_cv_ro1, grouped_list_t2o)

    t4_0 = merge_and_add_tuples(s_s_co_rv_t2_cv, s_s_cv_ro_t2_co)

    #print('t4_0********************===============================', t4_0)

    t4_final = merge_and_add_tuples(t4, t4_0)

    so_t2o_relaxed_o = s_t2_contracted_via_o_relaxed_o(grouped_list_so,grouped_list_t2o)
    #print('----------------------',so_t2o)
    grouped_list_so_t2o = group_by_relaxed_index(so_t2o_relaxed_o,2)
    #print('groupeddddddddddddddddd****************************************************',grouped_list_so_t2o)
#    grouped_list_so1 = group_by_relaxed_index(ex_val_combined_so, 2)
    t4_1 = so_t2oso_contraction(grouped_list_so,grouped_list_so_t2o)
    #print('t41===============', t4_1)


    t4_final = merge_and_add_tuples(t4_final,t4_1)


    sv_t2v = s_t2_contracted_via_v_relaxed_v(grouped_list_sv,grouped_list_t2v)
    grouped_list_sv_t2v = group_by_relaxed_index(sv_t2v,5)
    #grouped_list_sv1 = group_by_relaxed_index(ex_val_combined_sv,0)
    t4_2 = sv_t2vsv_contraction(grouped_list_sv,grouped_list_sv_t2v)
    #print('t42==================', t4_2)

    t4_final = merge_and_add_tuples(t4_final,t4_2)

    #
    so_t2o_relaxed_v = s_t2_contracted_via_o_relaxed_v(grouped_list_so,grouped_list_t2o)
    grouped_list_so_t2o1 = group_by_relaxed_index(so_t2o_relaxed_v,5)
#    grouped_list_sv1 = group_by_relaxed_index(ex_val_combined_sv, 0)
    #t4_3 = sv_t2vsv_contraction(grouped_list_sv1, grouped_list_so_t2o1)
    #print('===========================',grouped_list_so_t2o1)
    t4_3 = sv_sot2_contraction(grouped_list_sv,grouped_list_so_t2o1)
    #print('t43=========================', t4_3, len(t4_3))
    t4_final = merge_and_add_tuples(t4_final,t4_3)



    grouped_list_t2o_10 = group_by_relaxed_index(ex_val_combined_t2, 1)
    st10 = s_t2_contracted_via_o_relaxed_v1(grouped_list_so,grouped_list_t2o_10)
    st_10_grouped = group_by_relaxed_index(st10,5)
#    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',st_10_grouped)
    t4_4 = s_st_contraction_via_o(st_10_grouped,grouped_list_sv)
    print('t4_4--------------',t4_4)

    t4_final = merge_and_add_tuples(t4_final,t4_4)

    # print(merge_and_add_tuples(t4_1,t4_2))
    # print(merge_and_add_tuples(t4))

    return t4_final




# dominant_quadruples_ex_val_list = pt_quadruples_generator_via_st2(combined_so, combined_sv, combined_two_body_ints, combined_t2)
# #dominant_quadruples_ex_val_list = pt_quadruples_generator_via_st2(combined_so, combined_sv, combined_so_twc, combined_sv_twc, combined_t2)
# print(dominant_quadruples_ex_val_list)
# print(len(dominant_quadruples_ex_val_list))
#
# dominant_quadruples_ex_list = [item[0] for item in dominant_quadruples_ex_val_list]







#--------------------------------------------------------------------------------------------

#------------------------------------------* Pentuples Generation *-------------------------------------------------------------------------------------------------------------------------

def pt_pentuples_generator_via_st2(ex_val_combined_so, ex_val_combined_sv,v_two_body,ex_val_combined_t2):
    grouped_list_so = group_by_relaxed_index(ex_val_combined_so,3)
    grouped_list_sv = group_by_relaxed_index(ex_val_combined_sv,1)
    grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2,0)
    grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2,2)
    grouped_list_two_body_o = group_by_relaxed_index(v_two_body,0)
    grouped_list_two_body_v = group_by_relaxed_index(v_two_body,2)

    t3_so_so = so_so_contraction(grouped_list_so, grouped_list_two_body_o)
    grouped_list_t3_so_so = group_by_relaxed_index(t3_so_so, 5)

    t3_sv_sv = sv_sv_contraction(grouped_list_sv, grouped_list_two_body_v)
    grouped_list_t3_sv_sv = group_by_relaxed_index(t3_sv_sv, 2)
#    print(grouped_list_t3_sv_sv)

    t4_so_so_so = so_so_contraction(grouped_list_t3_so_so, grouped_list_two_body_o)
    grouped_list_t4_so_so_so = group_by_relaxed_index(t4_so_so_so, 7)


    t4_sv_sv_sv = sv_sv_contraction(grouped_list_t3_sv_sv, grouped_list_two_body_v)
    grouped_list_t4_sv_sv_sv = group_by_relaxed_index(t4_sv_sv_sv, 3)

    t4_t2o_contracted = s_t2_contracted_via_o(grouped_list_t4_so_so_so, grouped_list_t2o)
    #print('t3_t2o_contracted',t3_t2o_contracted)
    t4_t2v_contracted = s_t2_contracted_via_v(grouped_list_t4_sv_sv_sv, grouped_list_t2v)
#    print('t3_t2v_contracted', t3_t2v_contracted)
    t5 = merge_and_add_tuples(t4_t2o_contracted, t4_t2v_contracted)

    return t5







# dominant_pentuples_ex_val_list = pt_pentuples_generator_via_st2(combined_so, combined_sv, combined_two_body_ints, combined_t2)
# pprint.pprint(dominant_pentuples_ex_val_list)
# print(len(dominant_pentuples_ex_val_list))
#
# dominant_pentuples_ex_list = [item[0] for item in dominant_pentuples_ex_val_list]





def excited_det_list(ex_list):
    ex_det_list = list()
    for i in ex_list:
        ex_det = hf_rev.copy()
        for j in i[0]:
            ex_det[j] = 0
        for k in i[1]:
            ex_det[k] = 1
        ex_det_list.append(ex_det)
    return ex_det_list







hf_det = [hf_rev]
dominant_singles_dets = excited_det_list(dominant_singles_ex_list)
dominant_doubles_dets = excited_det_list(dominant_doubles_ex_list)
# dominant_triples_dets = excited_det_list(dominant_triples_ex_list)
# dominant_quadruples_dets = excited_det_list(dominant_quadruples_ex_list)
# dominant_pentuples_dets = excited_det_list(dominant_pentuples_ex_list)
# print('***********************************************')
# print('dominant_singles_dets')
# # print(dominant_singles_dets)
# print(len(dominant_singles_dets))
#
# print('dominant_doubles_dets')
# # print(dominant_doubles_dets)
# print(len(dominant_doubles_dets))
#
# print('dominant_triples_dets')
# # print(dominant_triples_dets)
# print(len(dominant_triples_dets))
#
# print('dominant_quadruples_dets')
# # print(dominant_quadruples_dets)
# print(len(dominant_quadruples_dets))
#
# print('dominant_pentuples_dets')
# # print(dominant_pentuples_dets)
# print(len(dominant_pentuples_dets))
#
# print('***********************************************')
#
# num_dominant_dets = len([hf_rev]) +len(dominant_singles_dets) + len(dominant_doubles_dets) + len(dominant_triples_dets) + len(dominant_quadruples_dets) + len(dominant_pentuples_dets)
# print('num_dominant_dets',num_dominant_dets)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # #Dont delete
# # dominant_triples_tensor = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
# # for i in occ_idx:
# #     for j in occ_idx:
# #         for k in occ_idx:
# #             for a in virt_idx:
# #                 for b in virt_idx:
# #                     for c in virt_idx:
# #                         if k > j > i and c > b > a:
# #                             print(i, j, k, a, b, c)
# #                             #                            dominant_triples_tensor[i,j,k,a,b,c] = np.einsum('m,m', dominant_so_tensor[i,j,a,:], dominant_t2_tensor[:,k,b,c]) + np.einsum('e,e', dominant_sv_tensor[i,:,a,b], dominant_t2_tensor[j,k,:,c])
# #                             temp_ex_list = [((i, j, k), (a, b, c))]
# #
# #                             denom_t = denominator(temp_ex_list)[0]
# #                             dominant_triples_tensor[i, j, k, a, b, c] = np.einsum('m,m', two_body_ints[i, j, a, :],
# #                                                                                   dominant_t2_tensor[:, k, b,
# #                                                                                   c]) + np.einsum('e,e',
# #                                                                                                   two_body_ints[i,
# #                                                                                                   :, a, b],
# #                                                                                                   dominant_t2_tensor[j,
# #                                                                                                   k, :, c])
# #
# #                             dominant_triples_tensor[i, j, k, a, b, c] = np.divide(
# #                                 dominant_triples_tensor[i, j, k, a, b, c], denom_t)
# #                             print(dominant_triples_tensor[i, j, k, a, b, c])
# #
# # trip, trip_ex = find_large_elements_from_numpy(dominant_triples_tensor, 1e-10)
# # # print(len(trip))
# # print('---------------------')
# # print(trip_ex)
# # print(trip)
# # print(len(trip_ex))
# #
# # dominant_triples_ex_list = []
# # dominant_triples_val = []
# # for jj in range(len(trip_ex)):
# #     if len(trip_ex[jj]) == len(set(trip_ex[jj])):
# #         dominant_triples_ex_list.append(trip_ex[jj])
# #         dominant_triples_val.append(trip[jj])
# #
# # print(dominant_triples_val)
# # print(dominant_triples_ex_list)
# # print(len(dominant_triples_ex_list))
# # exit()
# # dominant_triples_ex_list = convert_coords_to_halved_tuples(dominant_triples_ex_list)
# #
# #
# # def excited_det_list(ex_list):
# #     ex_det_list = list()
# #     for i in ex_list:
# #         ex_det = hf_rev.copy()
# #         for j in i[0]:
# #             ex_det[j] = 0
# #         for k in i[1]:
# #             ex_det[k] = 1
# #         ex_det_list.append(ex_det)
# #     return ex_det_list
# #
# #
# # print(excited_det_list(doubles_ex_list))
# # print('---------------------------------------')
# # print(dominant_t2_excitations_list)
# # print(len(dominant_t2_excitations_list))
# # print(len(dominant_triples_ex_list))
# # exit()
# # print(dominant_triples_ex_list)
# # print('---------------------------------------')
# #
# # # dominant_quadruples_ex_list = [((0,1,4,5),(2,3,6,7))]
# hf_det = [hf_rev]
# # hf_det = [[1, 1, 0, 0, 1, 1, 0, 0]]
# # #H6
# # hf_det = [[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
# # #H2O
# # hf_det = [[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]]
# # N2
# # hf_det = [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]
# # singles_det_list = excited_det_list(dominant_t1_ex_list)
# # doubles_det_list = excited_det_list(dominant_t2_excitations_list)
# # triples_det_list = excited_det_list(dominant_triples_ex_list)
# # # symmetry allowed triples for H6
# # # triples_det_list =[[0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0]]
# # # quadruples_det_list = excited_det_list(dominant_quadruples_ex_list)
# #
# # print(singles_det_list)
# # print(singles_ex_list)
# # print(len(triples_det_list))
# # print(triples_det_list)
#
# #det_list = hf_det + singles_det_list + doubles_det_list + triples_det_list  # + quadruples_det_list
#
# det_list = hf_det + dominant_singles_dets + dominant_doubles_dets + dominant_triples_dets + dominant_quadruples_dets + dominant_pentuples_dets #+ [[0,0,0,1,1,1,0,0,0,1,1,1]]
#
# # exact symmetry allowed dets (SDTQPH) for H6 in qubit convention
# #det_list =[[0,0,0,1,1,1,0,0,0,1,1,1]]+ [[0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
# # SDTQ all for H6 in qubit convention
# #det_list = [[0,0,0,1,1,1,0,0,0,1,1,1]]+ [[0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]]
# # det_list = [x[::-1] for x in det_list]
# # print(det_list)
# # # exit()
# # #det_list = [[0,0,0,1,1,1,0,0,0,1,1,1]]+[[0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]]
# #
# # #------------------------------------
# # all_singles = excited_det_list(singles_ex_list)
# # all_doubles = excited_det_list(doubles_ex_list)
# # all_triples = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
# # det_list = hf_det + all_singles + all_doubles + all_triples
# # print(len(det_list))
# #
#
#
#
# #---------- dont comment out in general
# det_list = [x[::-1] for x in det_list]
#
# from qiskit_addon_sqd.qubit import solve_qubit
# from qiskit_addon_sqd.qubit import sort_and_remove_duplicates
#
#
# #print(det_list)
# #exit()
# det_list = sort_and_remove_duplicates(np.asarray(det_list))
# b = solve_qubit(np.asarray(det_list), qubit_op)
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy + frozen_energy_shift
# # eigen_values.sort()
# print()
# perturbative_diag_energy = eigen_values[0]
# print('Perturbative diag energy', perturbative_diag_energy)
# print(len(det_list))
# subspace_ci_coeffs = b[1]
# subspace_ci_coeffs = subspace_ci_coeffs[:, 0].real
# print('Perturbative eigen vector', subspace_ci_coeffs)
# num_mbpt_dominant_dets = np.where(np.abs(subspace_ci_coeffs) > 1e-8)[0].size
#
#
# # # # NumPyMinimumEigensolver
# # numpy_solver = NumPyMinimumEigensolver()
# # result = numpy_solver.compute_minimum_eigenvalue(operator=qubit_op)
# # ref_value = result.eigenvalue.real
# #
# # exact_energy = ref_value + nuclear_repulsion_energy + frozen_energy_shift
#
# # print('Exact energy', exact_energy)
# #
# # print('Difference with exact energy', abs(exact_energy - eigen_values[0]))
#
# print('Difference with symmetry space energy', abs(symmetry_space_diag_energy - eigen_values[0]))
# print('num_symmetry_space_dominant_dets',num_symmetry_space_dominant_dets)
# print('num_mbpt_dominant_dets',num_mbpt_dominant_dets)





#-----------------





#
# var_form_pps = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles,
#                    excitations='sdt',
#                    qubit_mapper=mapper, initial_state=init_state)
# final_excitation_list_pps = var_form_pps._get_excitation_list()
# detss = hf_det + excited_det_list(final_excitation_list_pps)
# print(len(final_excitation_list_pps))
# # exit()
#
#
# detss = [x[::-1] for x in detss]
# # print(detss)
# # print(detss)
#
#
# b = solve_qubit(np.asarray(detss), qubit_op)
# # b = solve_qubit(np.asarray(sampled_dets_binary), jw_mapped_hamiltonian)
# eigen_values = b[0] + nuclear_repulsion_energy + frozen_energy_shift
# # eigen_values.sort()
# print(eigen_values)
# print(eigen_values[0])
# print(len(detss))
# sdt_en = eigen_values[0]
# print('Perturbative diag energy and sdt energy difference', abs(perturbative_diag_energy - sdt_en))
# subspace_ci_coeffs = b[1]
# subspace_ci_coeffs = subspace_ci_coeffs[:, 0].real
#
# print('sdt eigen vector', subspace_ci_coeffs)
# print(np.where(np.abs(subspace_ci_coeffs) > 1e-12)[0].size)



#---------------------- Up to N-th rank determinants generation -----------------


def nth_rank_mbpt_generator(ex_val_combined_so, ex_val_combined_sv,v_two_body,ex_val_combined_t2, r):
    '''

    :param ex_val_combined_so:
    :param ex_val_combined_sv:
    :param v_two_body:
    :param ex_val_combined_t2:
    :param r: maximum rank of excitations
    :return:
    '''

    mbpt_selected_ex_val_list = []
        #t3
    grouped_list_so_n_1 = group_by_relaxed_index(ex_val_combined_so, 3)
    grouped_list_sv_n_1 = group_by_relaxed_index(ex_val_combined_sv, 1)
    grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2, 0)
    grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2, 2)

    so_t2o_contracted = s_t2_contracted_via_o(grouped_list_so_n_1, grouped_list_t2o)
    sv_t2v_contracted = s_t2_contracted_via_v(grouped_list_sv_n_1, grouped_list_t2v)

    t3 = merge_and_add_tuples(so_t2o_contracted, sv_t2v_contracted)

    grouped_list_two_body_o = group_by_relaxed_index(v_two_body, 0)
    grouped_list_two_body_v = group_by_relaxed_index(v_two_body, 2)

    mbpt_selected_ex_val_list.append(t3)
    for nn in range(3, r):
        print ('calculating rank..', nn+1)

        # t4
        #grouped_list_so = group_by_relaxed_index(ex_val_combined_so,3)
        #grouped_list_sv = group_by_relaxed_index(ex_val_combined_sv,1)
        #grouped_list_t2o = group_by_relaxed_index(ex_val_combined_t2,0)
        #grouped_list_t2v = group_by_relaxed_index(ex_val_combined_t2,2)
        # grouped_list_two_body_o = group_by_relaxed_index(v_two_body,0)
        # grouped_list_two_body_v = group_by_relaxed_index(v_two_body,2)

        tn_so_so = so_so_contraction(grouped_list_so_n_1, grouped_list_two_body_o)
        grouped_list_tn_so_so = group_by_relaxed_index(tn_so_so, 2*nn-1)

        tn_sv_sv = sv_sv_contraction(grouped_list_sv_n_1, grouped_list_two_body_v)
        grouped_list_tn_sv_sv = group_by_relaxed_index(tn_sv_sv, nn-1)
        #print(grouped_list_t3_sv_sv)


        tn_t2o_contracted = s_t2_contracted_via_o(grouped_list_tn_so_so, grouped_list_t2o)
        #print('t3_t2o_contracted',t3_t2o_contracted)
        tn_t2v_contracted = s_t2_contracted_via_v(grouped_list_tn_sv_sv, grouped_list_t2v)
        #print('t3_t2v_contracted', t3_t2v_contracted)
        tnn_plus_1 = merge_and_add_tuples(tn_t2o_contracted, tn_t2v_contracted)
        if nn == 3:
            t4 = pt_quadruples_generator_via_st2(ex_val_combined_so,ex_val_combined_sv,v_two_body,ex_val_combined_t2)
            tnn_plus_1 = merge_and_add_tuples(tnn_plus_1,t4)

        mbpt_selected_ex_val_list.append(tnn_plus_1)

        grouped_list_so_n_1 = grouped_list_tn_so_so
        grouped_list_sv_n_1 = grouped_list_tn_sv_sv

    return mbpt_selected_ex_val_list



    #
    # return t4

print('-------------------------------')
print('1',len(dominant_singles_dets))
print('-------------------------------')
print('2',len(dominant_doubles_dets))

num_total_dets = len([hf_rev])+len(dominant_singles_dets) + len(dominant_doubles_dets)

# dominant_all_ex_val_list = nth_rank_mbpt_generator(combined_so, combined_sv, combined_two_body_ints, combined_t2,num_total_particles)

dominant_all_ex_val_list = nth_rank_mbpt_generator(combined_so, combined_sv, combined_two_body_ints, combined_t2, mbpt_max_rank)
for i in range(len(dominant_all_ex_val_list)):
    print(i+3)
    print(len(dominant_all_ex_val_list[i]))
    print('-----------------------------------')

dominant_all_ex_list = []
for i in range(len(dominant_all_ex_val_list)):
    print('-------------------------------')
    dominant_ex = [item[0] for item in dominant_all_ex_val_list[i]]
#    print(dominant_ex)
    dominant_all_ex_list.append(dominant_ex)
    #dominant_all_ex_list.append(dominant_ex)
    #print(dominant_all_ex_list)
    #print(dominant_all_ex_val_list[i])
    #for item in dominant_all_ex_val_list[i]:
    #    print(item[0])
print('*****')
#print(dominant_all_ex_list)





all_det_list_segmented = []
all_det_list_segmented.append(hf_det)
all_det_list_segmented.append(dominant_singles_dets)
all_det_list_segmented.append(dominant_doubles_dets)

all_det_list = []
all_det_list += hf_det
all_det_list += dominant_singles_dets
all_det_list += dominant_doubles_dets
for ex_list in dominant_all_ex_list:
    dets = excited_det_list(ex_list)
    all_det_list_segmented.append(dets)
    all_det_list += dets



num_total_dominant_dets = len(all_det_list)
print('num_total_dominant_dets',num_total_dominant_dets)









#---------- dont comment out in general
det_list = [x[::-1] for x in all_det_list]

#----------------------

import sys

# Create a large Python list of lists
# 1,000 rows, 100 columns
#my_list = [[i for i in range(100)] for _ in range(1000)]

# Create the NumPy array from the list
det_array = np.array(det_list, dtype=np.uint8) # Using uint8 for efficiency

# --- Memory Calculation ---

# Memory of the Python list (this is a rough estimate)
# It's the size of the outer list's pointers + the size of all inner lists + the size of all integer objects
list_pointers_size = sys.getsizeof(det_list)
inner_lists_size = sum(sys.getsizeof(row) for row in det_list)
# Python caches small integers, so this part is small, but the pointer overhead is huge
total_list_size_estimate = list_pointers_size + inner_lists_size
print(f"Estimated size of Python list of lists: ~{total_list_size_estimate / 1024:.2f} KB")


# Memory of the NumPy array
# It's the raw data size + a small, fixed overhead
array_size = det_array.nbytes + sys.getsizeof(det_array) - det_array.nbytes
print(f"Size of NumPy array: ~{array_size / 1024:.2f} KB")
#---------------------
print ('det_array',det_array)
from qiskit_addon_sqd.qubit import solve_qubit
from qiskit_addon_sqd.qubit import sort_and_remove_duplicates

del det_list
with open('det_list_mbpt_rank_'+str(mbpt_max_rank)+'_' + str(molecule) + '_' + str(basis) + '_' + str(
        bond_dist) + '.pkl', 'wb') as file:  # 'wb' means write binary mode
    pickle.dump(det_array, file)




