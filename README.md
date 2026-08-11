# PIGen-SQD
Physics-Informed Generative Machine Learning Driven Sample Based Quantum Diagonalization (PIGen-SQD) uses perturbative theoretic measures along with generative machine learning model (in this case a restricted boltzmann machine or RBM) to reconstruct dominant fermionic configurations for better configuration recovery.

Follow the steps to run the entire PIGen-SQD workflow:
-  1. The new_rbm.py is the main file for the workflow.
- 2. The inp.py file contains the control to run the workflow. Change the molecule names (check if the associated coordinate exists in the new_rbm.py file) and other parameters.
- 3. The optimized_mbpt2.py file contains the generation of configurations with the perturbative theoretic measures and sparse-tensor operations. If the "mbpt_max_rank" is set to "n" in the inp.py file, the augmented configurations to the hardware samples have a maximum excitation rank of n. If n > 4, only a subset of the algebraic terms are used to generate the "n"-th rank configurations. Till n = 4, the algebraic terms used in the calculation can be found in the Appendix section of PIGen-SQD paper: https://iopscience.iop.org/article/10.1088/2058-9565/ae917f 
