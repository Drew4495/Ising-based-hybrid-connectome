import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from scipy.stats import pearsonr
from scipy.stats import zscore
import pandas as pd





##############################
##############################
#region Helper functions
"""
Helper functions
"""
##############################
##############################
def corr2(a, b):
    """
    Compute the Pearson correlation coefficient between two 2-dimensional arrays.

    Parameters:
    - a (numpy.ndarray): The first 2D input array.
    - b (numpy.ndarray): The second 2D input array.

    Returns:
    - float: The Pearson correlation coefficient between the flattened arrays.

    Note:
    This function requires the input arrays to have the same shape and the scipy.stats.pearsonr function for computation.
    """
    return pearsonr(a.flatten(), b.flatten())[0]



def binarize_time_series(all_time_series):
    """
    This function preprocesses the time series data by removing time points with zero activity across all ROIs,
    normalizes the data using z-scores, binarizes the normalized data around zero,

    Parameters
    - time_series (numpy.ndarray): (num_ROIs x num_timepoints x num_subjects) functional time series data

    Returns:
    - binarized time_series
    """
    ##Initialize binarized array in same shape as time_series
    binarized_ts_ndarray = np.full_like(all_time_series, np.nan)

    ## Binarize time series for each subject
    for subj in range(all_time_series.shape[2]):
        time_series = all_time_series[:,:,subj]
        time_series = time_series[:, ~np.all(time_series == 0, axis=0)]  # Remove time points that are zero for all ROIs
        ts_zscored = zscore(time_series, axis=1, nan_policy='omit')  # Z-score normalization (across time for each ROI)
        binarized_data = np.sign(ts_zscored)
        binarized_ts_ndarray[:,:,subj] = binarized_data
    return binarized_ts_ndarray


def get_EIB_from_rssc(rssc):
    sum_positives = np.sum(rssc[rssc > 0])
    sum_negatives = np.abs(np.sum(rssc[rssc < 0]))
    if sum_negatives != 0:
        EIB = sum_positives / sum_negatives
    else:
        EIB = float('inf')  # To handle division by zero if there are no negative numbers
    return EIB


##############################
##############################
#endregion
##############################
##############################








##############################
##############################
#region Ising EIB Model Class
"""
Ising EIB Model Class
"""
##############################
##############################
class Ising_EIB_Model:

    def __init__(self, SC, rsfMRI_time_series, random_state=143):
        self.SC = SC
        self.rsfMRI_time_series = rsfMRI_time_series
        self.rsfMRI_time_series_bin = binarize_time_series(self.rsfMRI_time_series) if rsfMRI_time_series is not None else None
        self.num_regions = SC.shape[0] if SC is not None else None
        self.rssc_opt_all_subjects = None
        self.rs_num = random_state
        self.rs = np.random.RandomState(seed=random_state)
        self.spins = self.rs.choice([-1, 1], size=self.num_regions)



    def optimum(self, lambd_beta, num_ising_recon_sims=2000, num_ising_burn_in_steps=1000, verbose=True):
        """
        Main function to find optimal rssc for several subjects given their rsfMI_time_series and SC. For each subject,
            1. Generate rssc for several beta and lambda values (pseudo function)
            2. Simulate rsfMRI_time_series (ising function) to compute FC correlation value
            3. Choose rssc and associated params (lambda and beta) that optimize SC and FC correlation sum (FC_corr + SC_corr)
        Values that are stored:
            1. opt_combined_corr: FC_corr + SC_corr
            2. opt_FC_corr
            3. opt_SC_corr
            4. opt_lambda_FSE
            5. opt_beta_FSE
            6. opt_beta_ising_sim

        NOTE: I should store all FC_corr and SC_corr values for all parameter values to see the line plots for these

        Parameters
        - SC (ndarray): Structural connectivity, a 3D numpy array (n x n x subject_count).
        - FC_timeseries (list): Time series data, a list of length num_patients where each entry is a FC_timeseries (n x t)
        - lambd_beta (ndarray): Lambda/Beta combinations to optimize on, a 2D numpy array (n x 2) where lambdas are in 1st column and betas in the 2nd.

        Returns:
        - optimal_params_corrs (dict): All the param (lamb-beta) and correlation (FC and SC) values
        - rsscs_opt_subject (ndarray): rsscs for each subject
        - SC_corr (ndarray): All SC_corr values for all subjects AND all lambd-beta combinations
        - FC_corr (ndarray): All FC_corr values for all subjects AND all lambd-beta combinations
        - optimal_indices_params_all (???): ???
        """

        ## Check if # ROIs in SC matches in rsfMRI_timeseries
        if self.SC.shape[2] != self.rsfMRI_time_series_bin.shape[2]:
            raise ValueError("Number of subjects in 'SC' and 'rsfMRI_time_series_bin' do not match")

        ## Instantiate return variables and set variables
        num_subj = self.SC.shape[2]
        num_param_combs = lambd_beta.shape[0]
        SC_corr = np.full((num_param_combs, num_subj), np.nan)  # Similarity with SC for all param combinations
        FC_corr = np.full((num_param_combs, num_subj), np.nan)  # FC reconstruction quality for all param combinations
        rsscs_opt_subject = np.full(num_subj, np.nan).tolist()  # list to store optimal rssc for each subject
        optimal_params_corrs = pd.DataFrame(data=np.full((num_subj, 6), np.nan),
                                            columns=["opt_lambda_FSE", "opt_beta_FSE", "opt_beta_ising_sim",
                                                     "opt_FC_corr", "opt_SC_corr",
                                                     "opt_combined_corr"])  # Stores optimal beta-lambda combination and its respective beta_sim_opt for each subject
        optimal_indices_params_all = []

        ## Generate FC and Extract SC for each subject
        for subj_idx in range(num_subj):
            ## Start time recording
            start_time_subj = time.time()

            ##instantiate SC, FC, and storing variables for loop
            subj_SC = self.SC[:, :, subj_idx] - np.diag(np.diag(self.SC[:, :, subj_idx]))  # Remove diagonal
            subj_binarized_rsfMRI_timeseries = self.rsfMRI_time_series_bin[:, :, subj_idx]
            FC = np.corrcoef(subj_binarized_rsfMRI_timeseries)
            np.fill_diagonal(FC, 0)
            betas_sim_opt = np.full(num_param_combs,
                                    np.nan)  # Will store optimal beta from ising simulations for a given lambda-beta combination when constructing rssc
            rsscs_param_combinations = []  # Store rssc for every lambd-beta combination to return later

            ## Generate rssc for every lambda/beta combination for each subject
            for i in range(lambd_beta.shape[0]):
                start_time_lambd_beta = time.time()
                lambd, beta = lambd_beta[i, 0], lambd_beta[i, 1]
                ## Generate rssc based on pseudolikelihood function for each lambd-beta pair
                rssc = self.pseudo(binarizedData=subj_binarized_rsfMRI_timeseries, SC=subj_SC, lambda_val=lambd, B=beta)
                rsscs_param_combinations.append(rssc) # Append each rssc to list

                ## Store SC and FC reconstruction accuracy with optimal simulated beta (lambda not stored for some reason)
                SC_corr[i, subj_idx] = corr2(subj_SC, (np.sign(subj_SC)) * (np.abs(rssc)))
                beta_sim_opt, FC_recon, max_corr, _, _ = self.ising(J=rssc, FC=FC, num_sims=num_ising_recon_sims,
                                                               burn_in_steps=num_ising_burn_in_steps,
                                                               betas=np.arange(0.4, 2.8, 0.2))
                FC_corr[i, subj_idx] = max_corr
                betas_sim_opt[i] = beta_sim_opt
                end_time_lambd_beta = time.time()

                ## Print out end of each parameter combination's time
                print(f"lambda-beta combination {i + 1}/{lambd_beta.shape[0]} finished: {end_time_lambd_beta - start_time_lambd_beta}")

            ## Find optimal param values
            combined_subj_corr = FC_corr[:, subj_idx] + SC_corr[:, subj_idx]
            optimal_indices = np.flatnonzero(combined_subj_corr == combined_subj_corr.max())
            optimal_indices_params_all.append(optimal_indices)
            if len(optimal_indices) > 1 and verbose:
                print(f"WARNING: Subject {subj_idx}: >1 optimal lambda-beta combination. Choosing first optimal value in lambd_beta.")
            optimal_comb_idx = optimal_indices[0]

            ## Store optimal rssc, params and correlations values for each subject
            rsscs_opt_subject[subj_idx] = rsscs_param_combinations[optimal_comb_idx]
            optimal_params_corrs.loc[subj_idx, "opt_combined_corr"] = combined_subj_corr[optimal_comb_idx]
            optimal_params_corrs.loc[subj_idx, "opt_FC_corr"] = FC_corr[optimal_comb_idx, subj_idx]
            optimal_params_corrs.loc[subj_idx, "opt_SC_corr"] = SC_corr[optimal_comb_idx, subj_idx]
            optimal_params_corrs.loc[subj_idx, "opt_lambda_FSE"] = lambd_beta[optimal_comb_idx, 0]
            optimal_params_corrs.loc[subj_idx, "opt_beta_FSE"] = lambd_beta[optimal_comb_idx, 1]
            optimal_params_corrs.loc[subj_idx, "opt_beta_ising_sim"] = betas_sim_opt[optimal_comb_idx]

            ## Print completion of subject
            end_time_subj = time.time()
            print(f"subject {subj_idx} finished: {end_time_subj - start_time_subj}")

        self.rssc_opt_all_subjects = rsscs_opt_subject
        return optimal_params_corrs, rsscs_opt_subject, SC_corr, FC_corr, optimal_indices_params_all



    def pseudo(self, binarizedData, SC, lambda_val, B, iterationMax = 50000, learning_rate = 0.1):
        ## Convert warnings to errors for debugging. Remove later
        warnings.filterwarnings('error')

        ## Preprocess input and define variables
        np.fill_diagonal(SC, 0)  # Ensure diagonal of SC is 0. No self-connections
        num_nodes, num_data = binarizedData.shape
        iterationMax = iterationMax  # max iterations for gradient ascent
        dt = learning_rate  # learning rate for gradient ascent
        corr = (binarizedData @ binarizedData.T) / num_data
        J = np.zeros((num_nodes, num_nodes))

        ## Gradient ascent
        for t in range(iterationMax):
            ## Store current state of J for convergence check
            J_prev = np.copy(J)
            ## Compute Gradient of pseudolikelihood w.r.t J
            try:
                dl_dJ = (-corr) + \
                        ((0.5 * binarizedData @ np.tanh(B * (J @ binarizedData)).T) / num_data) + \
                        ((0.5 * (binarizedData @ np.tanh(B * (J @ binarizedData)).T).T) / num_data)
                dl_dJ = dl_dJ - np.diag(np.diag(dl_dJ))
            except Warning as e:
                error_info = {
                    "error": str(e),
                    "iteration num": t,
                    "J": J,
                    "dl_dJ": dl_dJ,
                    "binarizedData": binarizedData
                }
                print(str(e))
                return error_info

            ## Update J matrix
            J = J - (dt * dl_dJ) - (dt * lambda_val * (J - np.sign(J) * SC))

            ## Break if convergence is achieved
            if t > 0 and np.allclose(J, J_prev):
                break

        return J



    def ising(self, J, FC, num_sims=2000, burn_in_steps=1000, betas=np.arange(0.4, 2.8, 0.2)):
        """
        Simulates the Ising model to reconstruct a functional connectome (FC) and find the temperature
        at which the reconstructed FC best matches the observed FC.

        Parameters:
        - J (numpy.ndarray): A square matrix representing the coupling strengths between regions.
        - FC (numpy.ndarray): A square matrix representing the observed functional connectivity between regions.
        - sims (int): Number of simulations or time points for the Monte Carlo Markov Chain (MCMC) simulation.

        Returns:
        - max_index (int): The index of the temperature at which the correlation between the observed
                           and reconstructed FC is maximized.
        - FC_recon (numpy.ndarray): The reconstructed functional connectivity matrix at the optimal temperature.
        - inter_corr (numpy.ndarray): An array containing the correlation coefficients between the observed
                                      and reconstructed FC at each temperature.
        - S_recon (numpy.ndarray): The matrix of spin states at the optimal temperature.
        """

        ## Replace NaN values with 0 and ensure the diagonal is zero
        FC = np.nan_to_num(FC, nan=0.0)
        np.fill_diagonal(FC, 0)
        J = np.nan_to_num(J, nan=0.0)  #### DO I NEED TO FILL DIAGONALS WITH 0s
        num_ROIs = FC.shape[0]

        ## Initialize variables for ising simulation
        num_betas = len(betas)
        num_timesteps = burn_in_steps + num_sims
        spins_history = np.zeros((num_ROIs, num_sims, num_betas))  # Stores spin states at each temp and simulation step. Doesn't store burn_in steps
        corr_vec = np.zeros(len(betas))  # Stores FC/FC-recon correlation at each temp

        ## Perform ising simulation and computer correlation for each temp
        for beta_idx, beta in enumerate(betas):
            spin_vec = np.random.choice([-1, 1], size=num_ROIs)  # Initialize spin vector
            for time in range(num_timesteps):
                rand_spin_permut = np.random.permutation(
                    num_ROIs)  # Creates random ordering of spins to propose flips to more effectively search the space
                for idx in range(num_ROIs):
                    flip_idx = rand_spin_permut[
                        idx % num_ROIs]  # modulo is redundant but prevents no indexing past num_ROIs
                    delta_E = 2 * (J[flip_idx, :] @ (spin_vec * spin_vec[flip_idx]))

                    # Flip spin based on acceptance criteria (transition probability)
                    if delta_E <= 0 or (np.random.rand() <= np.exp(-delta_E * beta)):
                        spin_vec[flip_idx] = -spin_vec[flip_idx]

                # Store spin_vec for respective temp and time step
                if time >= burn_in_steps:
                    sim_time = time - burn_in_steps
                    if sim_time == num_sims:
                        print(
                            f"DEBUG: sim_time = 20, time = {time}, num_steps = {num_timesteps}, burn_in_steps = {burn_in_steps}")
                    spins_history[:, sim_time, beta_idx] = spin_vec

            ## Find temperature with maximum correlation
            FC_recon = spins_history[:, :, beta_idx] @ spins_history[:, :,
                                                       beta_idx].T / num_sims  ###MAY NEED TO CHANGE THIS TO PEARSON CORRELATION
            np.fill_diagonal(FC_recon, 0)
            corr_vec[beta_idx] = np.corrcoef(FC_recon.flatten(), FC.flatten())[
                0, 1]  # [0,1] extracts the correlation coefficient

        ## Store beta that maximized correlation between FC and FC_recon
        max_beta_idx = np.argmax(corr_vec)
        max_beta = betas[max_beta_idx]
        max_corr = corr_vec[max_beta_idx]
        FC_recon = spins_history[:, :, max_beta_idx] @ spins_history[:, :,
                                                       max_beta_idx].T / num_sims  ###MAY NEED TO CHANGE THIS TO PEARSON CORRELATION
        np.fill_diagonal(FC_recon, 0)
        spins_history_bestbeta = spins_history[:, :, max_beta_idx]

        return max_beta, FC_recon, max_corr, corr_vec, spins_history_bestbeta



    def calc_magnetization(self, spins):
        magnetization = np.mean(spins)
        return magnetization



    def calc_energy(self, spins, J):
        """I need to double check this is correct. I am worried I am overcounting the spins"""
        H = -0.5 * np.sum(J * np.outer(spins, spins))
        return H



    def calc_specific_heat(self, energies, beta):
        energies = np.array(energies)
        energy_mean = np.mean(energies)
        energy_sq_mean = np.mean(energies**2)
        specific_heat = (energy_sq_mean - energy_mean**2) * (beta**2)
        return specific_heat



    def calc_magnetic_susceptibility(self, magnetizations, beta):
        magnetizations = np.array(magnetizations)
        mag_mean = np.mean(magnetizations)
        mag_sq_mean = np.mean(magnetizations ** 2)
        susceptibility = (mag_sq_mean - mag_mean ** 2) * beta
        return susceptibility


    def sim_and_calc_thermodynamics_onebeta(self, beta, num_steps, num_iters_per_step, num_burn_in_steps, J, keep_spins=False):
        """Simulates the Ising Model using Metropolis-Hastings algorithm"""
        # Reset spins to random state
        self.spins = self.rs.choice([-1, 1], size=self.num_regions)
        energy_vals = []
        magnetization_vals = []
        if keep_spins:
            self.simulated_spins = np.zeros((self.num_regions, num_steps))

        # Burn-in Phase
        for _ in range(num_burn_in_steps):
            self.make_step(J=J, beta=beta)

        # Simulation Phase
        for i in range(num_steps):
            for _ in range(num_iters_per_step):
                self.make_step(J=J, beta=beta)
            #Calculate thermodynamic properties
            current_energy = self.calc_energy(spins=self.spins, J=J)
            current_magnetization = self.calc_magnetization(spins=self.spins)
            energy_vals.append(current_energy)
            magnetization_vals.append(current_magnetization)
            # Keep spins if set to true
            if keep_spins:
                self.simulated_spins[:,i] = self.spins

        # Calculate thermodynamic properties and save
        specific_heat = self.calc_specific_heat(energies=energy_vals, beta=beta)
        magnetic_susceptibility = self.calc_magnetic_susceptibility(magnetizations=magnetization_vals, beta=beta)
        thermodynamic_properties_dict = {
            'energy': energy_vals,
            'magnetization': magnetization_vals,
            'specific_heat': specific_heat,
            'magnetic_susceptibility': magnetic_susceptibility
        }

        return thermodynamic_properties_dict



    def sim_and_calc_thermodynamics_allbeta(self, betas, num_steps, num_iters_per_step, num_burn_in_steps, J,
                                            keep_spins=False):
        """
        Simulates the Ising Model using Metropolis-Hastings algorithm for a range of beta values and calculates
        thermodynamic properties for each beta.

        Parameters:
        - betas (array): Array of beta (inverse temperature) values to simulate.
        - num_steps (int): Number of steps to simulate after burn-in for each beta.
        - num_iters_per_step (int): Number of iterations per each step.
        - num_burn_in_steps (int): Number of burn-in steps to reach equilibrium before measurement.
        - J (ndarray): Interaction matrix.
        - keep_spins (bool): If True, keeps track of spins states across simulation steps.

        Returns:
        - dict: A dictionary containing lists of thermodynamic properties for each beta.
        """
        results = {
            'beta': [],
            'energy': [],
            'magnetization': [],
            'specific_heat': [],
            'magnetic_susceptibility': [],
            'mean_energy': [],
            'mean_magnetization': []
        }

        for beta in betas:
            print(f"Simulating for beta = {beta}")
            # Run the simulation and calculate thermodynamics for the current beta
            thermodynamics = self.sim_and_calc_thermodynamics_onebeta(
                beta=beta,
                num_steps=num_steps,
                num_iters_per_step=num_iters_per_step,
                num_burn_in_steps=num_burn_in_steps,
                J=J,
                keep_spins=keep_spins
            )

            # Collect the results for each beta
            results['beta'].append(beta)
            results['energy'].append(thermodynamics['energy'])
            results['magnetization'].append(thermodynamics['magnetization'])
            results['specific_heat'].append(thermodynamics['specific_heat'])
            results['magnetic_susceptibility'].append(thermodynamics['magnetic_susceptibility'])
            results['mean_energy'].append(np.mean(thermodynamics['energy']))
            results['mean_magnetization'].append(np.mean(thermodynamics['magnetization']))

        return results



    def make_step(self, J, beta):
        """Performs one Metropolis-Hastings step"""
        for idx in range(self.num_regions):
            # Propose spin flip
            spin_to_flip = self.rs.randint(self.num_regions)

            #Calculate energy pre and post spin slip and calculate difference
            current_energy = self.calc_energy(spins = self.spins, J=J)
            self.spins[spin_to_flip] *= -1
            proposed_energy = self.calc_energy(spins = self.spins, J=J)
            energy_diff = proposed_energy - current_energy

            # Accept or reject spin flip based on Boltzman probability
            if energy_diff > 0 and np.exp(-beta * energy_diff) < self.rs.rand():
                self.spins[spin_to_flip] *= -1  # Revert the flip
            else:
                self.H = proposed_energy  # Accept the new configuration












##############################
##############################
#endregion
##############################
##############################


