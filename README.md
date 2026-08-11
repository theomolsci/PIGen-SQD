# PIGen-SQD
Physics-Informed Generative Machine Learning Driven Sample Based Quantum Diagonalization (PIGen-SQD) uses perturbative theoretic measures along with generative machine learning model (in this case a restricted boltzmann machine or RBM) to reconstruct dominant fermionic configurations for better configuration recovery.

The steps and details to run the entire PIGen-SQD workflow:
-  1. The new_rbm.py is the main file for the workflow.
- 2. The inp.py file contains the control to run the workflow. Change the molecule names (check if the associated coordinate exists in the new_rbm.py file) and other parameters.
- 3. The optimized_mbpt2.py file contains the generation of configurations with the perturbative theoretic measures and sparse-tensor operations. If the "mbpt_max_rank" is set to "n" in the inp.py file, the augmented configurations to the hardware samples have a maximum excitation rank of n. If n > 4, only a subset of the algebraic terms are used to generate the "n"-th rank configurations. Till n = 4, the algebraic terms used in the calculation can be found in the Appendix section of PIGen-SQD paper: https://iopscience.iop.org/article/10.1088/2058-9565/ae917f
  4. The thresholds in the inp.py file can be tuned to optimize the efficiency. The variables: "t2_thresh = 1e-10, t1_thresh = 1e-10, s_thresh = 1e-10, two_body_int_thresh = 1e-10, s_t2_thresh = 1e-10, s_s_thresh = 1e-10" are set to these values by default and they are required for the sparse tensor operations. They can be made much tighter for more efficient perturbative selection, though with much smaller dimensional configuration subspace will be generated.
  5. For RBM-driven generation, the following parameters can be tuned "en_conv_thresh =1e-5, rbm_training_iter = 3, n_gibbs_sampling = 1".
  6. The workflow given here is for H2O in 6-31G basis with bond-stretch factor 1.0.
 
- NOTE: The code in this branch is not the optimized variant of the workflow. This can be used to re-generate the results of PIGen-SQD paper (https://iopscience.iop.org/article/10.1088/2058-9565/ae917f). A more optimized variant of the workflow would be uploaded soon. If you use the code or any of the data, please cite the paper: **Chayan Patra et al 2026 Quantum Sci. Technol. 11 035075**, **DOI: 10.1088/2058-9565/ae917f**

