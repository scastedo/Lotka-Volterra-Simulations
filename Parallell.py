import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
from scipy.optimize import root_scalar 
from matplotlib.ticker import FormatStrFormatter
#import numba as nb
from scipy.special import erf
from tqdm import tqdm
import multiprocessing as mp
import csv
import joblib
import pandas as pd


def W_calc(delta):
    output = np.zeros((delta.size,3), dtype = float)
    for i in range(0,delta.size):
        output[i,0] = 0.5*(1 + erf(delta[i]/np.sqrt(2)))
        output[i,1] = 0.5*((np.sqrt(2/np.pi)*np.exp(-(delta[i]**2)/2))+delta[i]*(1 + erf(delta[i]/np.sqrt(2))))
        output[i,2] = output[i,0] + delta[i]*output[i,1]
    return output

def Variance(w,big_gamma,delta,r): #This function calculates the variance of the set of the data in terms of the w variables
    var_array = np.zeros((delta.size), dtype = float)
    for i in range(0,delta.size):
        var_array[i] = w[i,2]/((w[i,2] + ((big_gamma * w[i,0])/(1 + r*(w[i,1]**2/w[i,2]))))**2 * (1 + r*(w[i,1]**2/w[i,2])))
    return var_array


def R_theory(w,bg,delta, sigma):
    r_array = np.zeros((delta.size),dtype=float)
    for i in range(0,delta.size):
        #b = 2*bg*w[i,0]-np.power(sigma,-2)
        #c = np.power(bg*w[i,0],2)
        #func = lambda x:x**2 +b*x + c
        #result = root_scalar(func,  bracket=[-float('inf'), float('inf')])
        #if result.converged:
        #  row_val = (result.root-w[i,2])/np.power(w[i,1],2)
        #    r_array[i] = row_val
        r2 = (np.power(sigma,-2)-w[i,2])*np.power(w[i,1],-2)
        if r2>0:
            r_array[i] = r2
    return r_array


def Biomass(w,big_gamma,delta,r,gamma, MU):
    bio_array = np.zeros((delta.size), dtype = float)
    for i in range(0,delta.size):
        b = (w[i,2] * delta[i])*(1 + r*(w[i,1]**2/w[i,2])) - gamma*w[i,0]*w[i,1]
        b/= (w[i,1] * (w[i,2]*(1 + r*(w[i,1]**2/w[i,2])) + (big_gamma * w[i,0])))
        b-= MU
        bio_array[i] = 1/b
    return bio_array

def test_correlations(N, alpha, SIGMA, GAMMA, BIG_GAMMA, ROW, COLUMN, MU):
    rows_sum=0
    pp_sum = 0
    columns_sum = 0
    gamma_sum = 0
    matrix = alpha-MU/N
    #Calculating Row correlations
    squared_matrix = 0 #(np.multiply(matrix,matrix)) #element-wise
    matrix_mean  = 0 # matrix.mean()
    matrix_variance = 0#(squared_matrix.mean() - matrix.mean()**2)
    theoretical_variance = SIGMA*SIGMA/N
    if SIGMA != 0:
        for i in range(N):
            fixed_index = 0
            fixed_row_sum = 0
            
            fixed_index_column = 0
            fixed_column_sum = 0
            
            fixed_index_gamma = 0
            fixed_gamma_sum = 0
            for k in [x for x in range(i) if x != i]:
                pp_sum += matrix[i,k]*matrix[k,i]/theoretical_variance
            for j in [x for x in range(N) if x != i]:
                temp_row = (matrix[i,j]*matrix[i,:])
                temp_row = np.delete(temp_row, (j))/theoretical_variance
                fixed_index = temp_row.mean() # averages for fixed element, and changes to a correlation
                fixed_row_sum += fixed_index #averaging for each pair in fixed row
                
                temp_column = (matrix[j,i]*matrix[:,i])
                temp_column = np.delete(temp_column, (j))/theoretical_variance
                fixed_index_column = temp_column.mean() # averages for fixed element, and changes to a correlation
                fixed_column_sum += fixed_index_column #averaging for each pair in fixed column

                temp_gamma = (matrix[i,j]*matrix[:,i])
                temp_gamma = np.delete(temp_gamma, (j))/theoretical_variance
                if i>j:
                    temp_gamma = np.delete(temp_gamma, (i-1))
                else: 
                    temp_gamma = np.delete(temp_gamma, (i))
                fixed_index_gamma = temp_gamma.mean() # averages for fixed element, and changes to a correlation
                fixed_gamma_sum += fixed_index_gamma #averaging for each pair in fixed column

            fixed_row_sum/= N # average correlations in a row... should be equal to R/N
            fixed_column_sum/= N
            fixed_gamma_sum/=N
            rows_sum += fixed_row_sum #averaging for each row
            columns_sum += fixed_column_sum
            gamma_sum +=fixed_gamma_sum
        pp_sum /= (N*(N-1)*0.5)
    return [matrix_mean, matrix_variance, pp_sum, rows_sum, columns_sum, gamma_sum]

def beta_kappa_correlations(N,row,column,gamma, sigma):
    correlation_values_b_k = np.zeros((N,2), dtype = float)
    
    for i in range(N):
        #Generate random normal values
        rand_val_1 = np.random.normal(0, 1)
        rand_val_2 = np.random.normal(0, 1)
    
        beta_i = rand_val_1*np.sqrt(row/N)
        beta_i *= sigma/np.sqrt(N)
        #Different cases for when r=0 as wd usually do DIV by 0
        if (row ==0 and gamma==0):
            kappa_i = np.sqrt(column/N)*rand_val_2
            kappa_i *= sigma/np.sqrt(N)
        else: 
            kappa_i = gamma*rand_val_1/(np.sqrt(row*N)) + (np.sqrt(column-gamma*gamma/row)*rand_val_2)/np.sqrt(N)
            kappa_i *= sigma/np.sqrt(N)

        
        correlation_values_b_k[i,0] = beta_i
        correlation_values_b_k[i,1] = kappa_i

    return correlation_values_b_k

def omega_correlations(N,row,column,gamma,big_gamma, sigma):
    correlation_values_omega = np.zeros((N,N), dtype = float)
    #temp variables to make below calulation clearer
    temp1 = 1.0-(row+column)/N
    temp2 = big_gamma - 2.0*gamma/N
    for i in range(N):
        for j in range(i):
            rand_val_1 = np.random.normal(0, 1)
            rand_val_2 = np.random.normal(0, 1)
                
            correlation_values_omega[i,j] = rand_val_1*np.sqrt(temp1)
            correlation_values_omega[i,j] *= sigma/np.sqrt(N)

            correlation_values_omega[j,i] = rand_val_1*(temp2)/(np.sqrt(temp1)) + rand_val_2*np.sqrt(temp1 - (temp2*temp2)/temp1)
            correlation_values_omega[j,i] *= sigma/np.sqrt(N)
    return correlation_values_omega

# @nb.jit(nopython = True) # delete this line if you dont have numba, its just for efficiency
def may_matrix(N, sigma, mu, gamma, b_gamma, r, c, bk_array, omega_matrix): 
    result = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z_ij = bk_array[i,0] + bk_array[j,1] + omega_matrix[i,j]
            result[i,j] = mu/N + z_ij    
    np.fill_diagonal(result,0) # Forces diagonal components to = 0 due to LV eqn below
    return result

def lotka_volterra(X, t, alpha):
    dxdt = X*(1 -X + alpha @ X)
    if np.any(dxdt>10**4):
        X = np.zeros((X.shape))
        dxdt = np.zeros((dxdt.shape))
        raise Exception
    return dxdt

def run_lv(args):
    SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
    overall_x = np.zeros((N,iterations))
    values = np.zeros((8,))
    # average_correlations = 0
    for i in range(iterations):
        alpha = may_matrix(N, SIGMA, MU, GAMMA, BIG_GAMMA, ROW, COLUMN, beta_kappa_correlations(N,ROW,COLUMN,GAMMA, SIGMA), omega_correlations(N,ROW,COLUMN,GAMMA,BIG_GAMMA, SIGMA))
        x_0 = np.full((N),1) #start at x=1, doesn't matter
        t = np.arange(T_MIN, T_MAX+DT, DT)
        integrate = int.odeint(lotka_volterra, x_0, t, args = (alpha,))
        overall_x[:,i] = integrate[-1,:]
        # average_correlations += test_correlations(N, alpha, SIGMA, GAMMA, BIG_GAMMA, ROW, COLUMN, MU)
    # average_correlations /= iterations
    # print('Average PP correlations: ', average_correlations[2], 'should be: ', BIG_GAMMA)
    # print('Average row correlations:' , average_correlations[3], 'should be: ' ,ROW) 
    # print('Average column correlations: ', average_correlations[4], 'should be: ', COLUMN)
    # print('Average gamma correlations:' , average_correlations[5], 'should be: ' ,GAMMA) 
    # print("")
    values[0] = overall_x.mean()
    values[1] = (np.count_nonzero(overall_x > 0.01)/(N*iterations))
    values[2] = SIGMA
    values[3] = BIG_GAMMA
    values[4] = ROW
    values[5] = GAMMA
    values[6] = COLUMN
    values[7] = MU
    return values
def distance_measure(x_1, x_2):
    diff = np.power((x_1-x_2),2)
    N_average = np.mean(diff,axis = 0)
    T_average = np.mean (N_average)
    diff_norm = np.power(x_1,2)
    N_average_norm = np.mean(diff_norm,axis = 0)
    T_average_norm = np.mean (N_average_norm)
    return T_average/T_average_norm

def run_lv_distance(args):
    SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
    distances = []
    diverge = 0
    for i in range(iterations): 
        alpha = may_matrix(N, SIGMA, MU, GAMMA, BIG_GAMMA, ROW, COLUMN, beta_kappa_correlations(N,ROW,COLUMN,GAMMA, SIGMA), omega_correlations(N,ROW,COLUMN,GAMMA,BIG_GAMMA, SIGMA))
        x_0 = np.random.uniform(0.7,1.3,N) #start at x=1, doesn't matter
        x_1 = np.random.uniform(0.7,1.3,N) #start at x=1, doesn't matter
        t = np.arange(T_MIN, T_MAX+DT, DT)
        try:
            integrate_1 = int.odeint(lotka_volterra, x_0, t, args = (alpha,))
            integrate_2 = int.odeint(lotka_volterra, x_1, t, args = (alpha,))
            distances.append(distance_measure(integrate_1, integrate_2))
        except:
            diverge += 1   
    if len(distances) == 0:
        return -1
    else:
        return sum(distances)/len(distances)
def run_lv_instabilites(args):
    SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
    diverge = 0
    for i in range(iterations): 
        alpha = may_matrix(N, SIGMA, MU, GAMMA, BIG_GAMMA, ROW, COLUMN, beta_kappa_correlations(N,ROW,COLUMN,GAMMA, SIGMA), omega_correlations(N,ROW,COLUMN,GAMMA,BIG_GAMMA, SIGMA))
        x_0 = np.random.uniform(0.7,1.3,N) #start at x=1, doesn't matter
        t = np.arange(T_MIN, T_MAX+DT, DT)
        try:
            integrate_1 = int.odeint(lotka_volterra, x_0, t, args = (alpha,))
        except:
            diverge += 1   
    return diverge/iterations

def calculate_distances_vary_bg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, gamma_arr, gamma_sigma_dict, w_array, delta, filename):
    args_list = []
    for BIG_GAMMA in gamma_sigma_dict:
        for SIGMA in gamma_sigma_dict[BIG_GAMMA]:
            args_list.append((SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N))
    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_lv_distance, args_list)
        pool.close()
        pool.join()
        # process the results
        values = [np.zeros((len(gamma_sigma_dict[BIG_GAMMA]))) for BIG_GAMMA in gamma_arr]
        opper_line = np.zeros(len(gamma_arr))
        mifty_line = np.zeros(len(gamma_arr))

        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(np.array(gamma_sigma_dict[BIG_GAMMA]) == SIGMA)[0][0]
            ii = np.where(np.array(gamma_arr) == BIG_GAMMA)[0][0]
            values[ii][jj] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'BIG_GAMMA', 'DISTANCE'] 
        with open(filename, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['ROW', ROW])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for ii, BIG_GAMMA in enumerate(gamma_arr):
                mifty = True
                opper = True
                for jj, SIGMA in enumerate(gamma_sigma_dict[BIG_GAMMA]):
                    writer.writerow([SIGMA**2, BIG_GAMMA, values[ii][jj]])
                    if (values[ii][jj] == -1) and mifty:
                        mifty_line[ii] = SIGMA**2
                        mifty = False
                    if (values[ii][jj] > 0.1) and opper:
                        opper_line[ii] = SIGMA**2
                        opper = False
        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        for i, BIG_GAMMA in enumerate(gamma_arr): 
            ax_1.scatter(np.array(gamma_sigma_dict[BIG_GAMMA])**2, values[i][:], label= f'$\\Gamma$ = {gamma_arr[i]:.2g}')
            ax_1.vlines(mifty_line[i], 0, 1)
            ax_1.vlines(opper_line[i], 0, 1)
        ax_1.set_title('distance measure')
        ax_1.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Distance, d')
        plt.show()
def calculate_distances_vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename):
    args_list = []
    for ROW in row_sigma_dict:
        for SIGMA in row_sigma_dict[BIG_GAMMA]:
            args_list.append((SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N))
    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(run_lv_distance, args_list), total=len(args_list)))
        pool.close()
        pool.join()
        # process the results
        values = [np.zeros((len(row_sigma_dict[ROW]))) for ROW in row_arr]
        opper_line = np.zeros(len(row_arr))
        mifty_line = np.zeros(len(row_arr))

        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(np.array(row_sigma_dict[ROW]) == SIGMA)[0][0]
            ii = np.where(np.array(row_arr) == ROW)[0][0]
            values[ii][jj] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'ROW', 'DISTANCE'] 
        with open(filename, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['BG', BIG_GAMMA])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for ii, ROW in enumerate(row_arr):
                mifty = True
                opper = True
                for jj, SIGMA in enumerate(row_sigma_dict[ROW]):
                    writer.writerow([SIGMA**2, ROW, values[ii][jj]])
                    if (values[ii][jj] == -1) and mifty:
                        mifty_line[ii] = SIGMA**2
                        mifty = False
                    if (values[ii][jj] > 0.1) and opper:
                        opper_line[ii] = SIGMA**2
                        opper = False
        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        for i, ROW in enumerate(row_arr): 
            ax_1.scatter(np.array(row_sigma_dict[BIG_GAMMA])**2, values[i][:], label= f'$\\Gamma$ = {row_arr[i]:.2g}')
            # ax_1.vlines(mifty_line[i], 0, 1)
            # ax_1.vlines(opper_line[i], 0, 1)
        ax_1.set_title('distance measure')
        ax_1.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Distance, d')
        plt.show()
def instabilities_vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename):
    args_list = []
    for ROW in row_sigma_dict:
        for SIGMA in row_sigma_dict[BIG_GAMMA]:
            args_list.append((SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N))
    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_lv_instabilites, args_list)
        pool.close()
        pool.join()
        # process the results
        values = [np.zeros((len(row_sigma_dict[ROW]))) for ROW in row_arr]

        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(np.array(row_sigma_dict[ROW]) == SIGMA)[0][0]
            ii = np.where(np.array(row_arr) == ROW)[0][0]
            values[ii][jj] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'ROW', 'Divergence Percentage'] 
        with open(filename, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['BG', BIG_GAMMA])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for ii, ROW in enumerate(row_arr):

                for jj, SIGMA in enumerate(row_sigma_dict[ROW]):
                    writer.writerow([SIGMA**2, ROW, values[ii][jj]])
        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        for i, ROW in enumerate(row_arr): 
            ax_1.scatter(np.array(row_sigma_dict[BIG_GAMMA])**2, values[i][:], label= f'$\\Gamma$ = {row_arr[i]:.2g}')
            # ax_1.vlines(mifty_line[i], 0, 1)
            # ax_1.vlines(opper_line[i], 0, 1)
        ax_1.set_title('Divergence Instability')
        ax_1.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Onset of M->Infty Instability')
        plt.show()   
def vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename):
    args_list = []
    for ROW in row_sigma_dict:
        for SIGMA in row_sigma_dict[ROW]:
            args_list.append((SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N))
    if __name__ == '__main__':
        # Set the number of parallel processes
        num_processes = joblib.cpu_count()

        # Use Parallel to parallelize the execution
        results = joblib.Parallel(n_jobs=num_processes)(
            joblib.delayed(run_lv)(args) for args in args_list)
        
        # process the results
        values = [np.zeros((len(row_sigma_dict[ROW]), 8)) for ROW in row_arr]
        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(np.array(row_sigma_dict[ROW]) == SIGMA)[0][0]
            ii = np.where(np.array(row_arr) == ROW)[0][0]
            values[ii][jj, :] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'ROW', 'BIOMASS', 'FRACTION'] 
        with open(filename, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['BIG_GAMMA', BIG_GAMMA])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for ii, ROW in enumerate(row_arr):
                for jj, SIGMA in enumerate(row_sigma_dict[ROW]):
                    writer.writerow([SIGMA**2, ROW, values[ii][jj, 0], values[ii][jj, 1]])

        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        fig_2, ax_2 = plt.subplots(1, 1, figsize = (15, 7))
        for i, ROW in enumerate(row_arr): 
            ax_1.scatter(np.array(row_sigma_dict[ROW])**2, values[i][:,0], label= f'r = {row_arr[i]:.2g}')
            ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 
            ax_2.scatter(np.array(row_sigma_dict[ROW])**2, values[i][:,1], label= f'r = {row_arr[i]:.2g}') 
            ax_2.plot(Variance(w_array,BIG_GAMMA,delta,ROW), w_array[:,0])
        ax_2.set_title('FRACTION')
        ax_1.set_title('AVERAGE ABUNDANCE')
        ax_1.legend()
        ax_2.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Abundance, M')
        ax_2.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_2.set_ylabel('Fraction of Surviving Species, $\\phi$')

        plt.show()
    
def vary_bg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, bg_arr, sigma_arr, w_array, delta):
    args_list = [(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N) for BIG_GAMMA in bg_arr for SIGMA in sigma_arr] 

    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_lv, args_list)
        pool.close()
        pool.join()
        
        # process the results
        values = np.zeros((sigma_arr.size, 8, bg_arr.size)) 
        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(sigma_arr == SIGMA)[0][0] 
            ii = np.where(bg_arr == BIG_GAMMA)[0][0]
            values[jj, :, ii] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'BIG_GAMMA', 'BIOMASS', 'FRACTION'] 
        with open('results_bg.csv', 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['ROW', ROW])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for jj, SIGMA in enumerate(sigma_arr):
                for ii, BIG_GAMMA in enumerate(bg_arr):
                    writer.writerow([SIGMA**2, BIG_GAMMA, values[jj, 0, ii], values[jj, 1, ii]])

        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        fig_2, ax_2 = plt.subplots(1, 1, figsize = (15, 7))
        for i, BIG_GAMMA in enumerate(bg_arr): #######################################
            ax_1.scatter(sigma_arr**2, values[:,0,i], label= f'$\\Gamma$ = {bg_arr[i]:.2g}')
            ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 
            ax_2.scatter(sigma_arr**2, values[:,1,i], label= f'$\\Gamma$ = {bg_arr[i]:.2g}') 
            ax_2.plot(Variance(w_array,BIG_GAMMA,delta,ROW), w_array[:,0])
        ax_2.set_title('FRACTION')
        ax_1.set_title('AVERAGE ABUNDANCE')
        ax_1.legend()
        ax_2.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Abundance, M')
        ax_2.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_2.set_ylabel('Fraction of Surviving Species, $\\phi$')

        plt.show()

def vary_lg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, lg_arr, sigma_arr, w_array, delta):
    args_list = [(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N) for GAMMA in lg_arr for SIGMA in sigma_arr] #############################

    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_lv, args_list)
        pool.close()
        pool.join()
        
        # process the results
        values = np.zeros((sigma_arr.size, 8, lg_arr.size)) ###############################################
        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(sigma_arr == SIGMA)[0][0] 
            ii = np.where(lg_arr == GAMMA)[0][0]
            values[jj, :, ii] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'GAMMA', 'BIOMASS', 'FRACTION'] ###############################################
        with open('results_lg.csv', 'w', newline='') as f: ###############################################
            writer = csv.writer(f)
            writer.writerow(['BIG_GAMMA', BIG_GAMMA])
            writer.writerow(['ROW', ROW])
            writer.writerow(['COLUMN', COLUMN])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for jj, SIGMA in enumerate(sigma_arr):
                for ii, GAMMA in enumerate(lg_arr):
                    writer.writerow([SIGMA**2, GAMMA, values[jj, 0, ii], values[jj, 1, ii]])

        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        fig_2, ax_2 = plt.subplots(1, 1, figsize = (15, 7))
        for i, GAMMA in enumerate(lg_arr): #######################################
            ax_1.scatter(sigma_arr**2, values[:,0,i], label= f'$\\gamma$ = {lg_arr[i]:.2g}')
            ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 
            ax_2.scatter(sigma_arr**2, values[:,1,i], label= f'$\\gamma$ = {lg_arr[i]:.2g}') 
            ax_2.plot(Variance(w_array,BIG_GAMMA,delta,ROW), w_array[:,0])
        ax_2.set_title('FRACTION')
        ax_1.set_title('AVERAGE ABUNDANCE')
        ax_1.legend()
        ax_2.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Abundance, M')
        ax_2.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_2.set_ylabel('Fraction of Surviving Species, $\\phi$')

        plt.show()
def vary_c(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, col_arr, sigma_arr, w_array, delta, filename):
    args_list = [(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N) for COLUMN in col_arr for SIGMA in sigma_arr] #############################

    if __name__ == '__main__':
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_lv, args_list)
        pool.close()
        pool.join()
        
        # process the results
        values = np.zeros((sigma_arr.size, 8, col_arr.size)) ###############################################
        for i, args in enumerate(args_list):
            SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
            jj = np.where(sigma_arr == SIGMA)[0][0] 
            ii = np.where(col_arr == COLUMN)[0][0]
            values[jj, :, ii] = results[i]

        # save the results to a CSV file
        header = ['VARIANCE', 'COLUMN', 'BIOMASS', 'FRACTION'] 
        with open(filename, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(['BIG_GAMMA', BIG_GAMMA])
            writer.writerow(['ROW', ROW])
            writer.writerow(['GAMMA', GAMMA])
            writer.writerow(['MU', MU])
            writer.writerow(header)
            for jj, SIGMA in enumerate(sigma_arr):
                for ii, COLUMN in enumerate(col_arr):
                    writer.writerow([SIGMA**2, COLUMN, values[jj, 0, ii], values[jj, 1, ii]])

        # make the figures
        fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
        fig_2, ax_2 = plt.subplots(1, 1, figsize = (15, 7))
        for i, COLUMN in enumerate(col_arr): 
            ax_1.scatter(sigma_arr**2, values[:,0,i], label= f'c = {col_arr[i]:.2g}')
            ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 
            ax_2.scatter(sigma_arr**2, values[:,1,i], label= f'c = {col_arr[i]:.2g}') 
            ax_2.plot(Variance(w_array,BIG_GAMMA,delta,ROW), w_array[:,0])
        ax_2.set_title('FRACTION')
        ax_1.set_title('AVERAGE ABUNDANCE')
        ax_1.legend()
        ax_2.legend()
        ax_1.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_1.set_ylabel('Average Abundance, M')
        ax_2.set_xlabel('Variance of Interaction Strength, $\\sigma^2$')
        ax_2.set_ylabel('Fraction of Surviving Species, $\\phi$')

        plt.show()
def r_phi(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, w_array, delta, filename):
    args_list = []
    plt.figure(figsize=(10, 6))
    plt.plot(R_theory(w_array,BIG_GAMMA,delta, SIGMA), w_array[:,0])
    for ROW in row_arr:
        args_list.append((SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N))
    if __name__ == '__main__':
        # Set the number of parallel processes
        num_processes = joblib.cpu_count()

        # Use Parallel to parallelize the execution
        results = joblib.Parallel(n_jobs=num_processes)(
            joblib.delayed(run_lv)(args) for args in args_list)
        
        # process the results
        fraction_values = [result[1] for result in results]  # Extract values[1] from results
        # Create a graph
        # plt.figure(figsize=(10, 6))
        plt.scatter(row_array, fraction_values, marker='x')
        # Create a DataFrame with the data
        data = {'Row': row_array, 'Fraction': fraction_values}
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('row_phi.csv', index=False)

        # plt.plot(R_theory(w_array,BIG_GAMMA,delta), w_array[:,0])
        plt.xlabel('$r$', fontsize=17)
        plt.ylabel('$\\phi$',fontsize=17)
        plt.xlim(0,6.25)
        plt.ylim(.5,.9)
        plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust the labelsize as needed
        plt.show()

#################################################################
#################################################################
#Run-time constants
T_MIN = 0.0
T_MAX = 100.0
DT = 0.01
iterations = 100
N = 500

# Parameter Constants
delta = np.linspace(0,5, num = 100000)
w_array = W_calc(delta)

MU = -1.0
COLUMN = 10.0 # GREATER THAM GAMMA^2/R
BIG_GAMMA = 0.0 # Predator Prey
GAMMA  = 0.0
ROW = 0.0
SIGMA = 0.0 
#######################################################################
#######################################################################
# row_sigma_dict = {0: np.arange(0,1.8,1.8/6),
#                    0.5: np.arange(0, 1.5, 1.5/6),
#                    1.0: np.arange(0, 1.25, 1.25/6),
#                    1.5: np.arange(0, 1.1, 1.1/6),
#                    2.0: np.arange(0, 1.0, 1/6),
#                    2.5: np.arange(0, 0.8, .8/6)}
# row_arr = list(row_sigma_dict.keys())
# filename = 'row_test.csv'
# vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename)
#######################################################################
#######################################################################
# bg_arr = np.array([-1,0,1])
# sigma_arr=  np.sqrt(np.arange(0,2,0.25)) # SIGMA
# vary_bg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, bg_arr, sigma_arr, w_array, delta)
#######################################################################
#######################################################################
# lg_arr = np.arange(0,1.0,0.2)
# sigma_arr=  np.sqrt(np.arange(0,2,0.25)) # SIGMA
# vary_lg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, lg_arr, sigma_arr, w_array, delta)
#######################################################################
#######################################################################
# col_array = np.arange(0,1.0,0.2)
# sigma_arr=  np.sqrt(np.arange(0,2,0.25)) # SIGMA
# filename = 'col_test.csv'
# vary_c(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, col_array, sigma_arr, w_array, delta, filename)
########################################
# MU = -3.0
# gamma_sigma_dict = {0: np.sqrt(np.arange(2.25,7.3,0.05))}
# gamma_arr = list(gamma_sigma_dict.keys())
# filename = 'opper_numeric.csv'
# calculate_distances_vary_bg(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, gamma_arr, gamma_sigma_dict, w_array, delta, filename)
######################################
# MU = -1.0
# row_sigma_dict = {0: np.sqrt(np.arange(0,5,0.05)),
#                  1: np.sqrt(np.arange(0,5,0.05)),
#                  5: np.sqrt(np.arange(0,5,0.05))}
# row_arr = list(row_sigma_dict.keys())
# filename = 'fraction_r.csv'
# calculate_distances_vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename)  




# vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename)
# row_sigma_dict = {0: np.sqrt(np.arange(0,5,0.05)),
#                   1: np.sqrt(np.arange(0,5,0.05)),
#                   5: np.sqrt(np.arange(0,5,0.05))}
# row_arr = list(row_sigma_dict.keys())
# filename = 'opper_onset_vary_r.csv'
# instabilities_vary_r(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_arr, row_sigma_dict, w_array, delta, filename)












##FIGURE 4(WORKS)
T_MIN = 0.0
T_MAX = 100.0
DT = 0.01
iterations = 100
N = 500

# Parameter Constants
delta = np.linspace(0,5, num = 100000)
w_array = W_calc(delta)

MU = -1.0
COLUMN = 10.0 # GREATER THAM GAMMA^2/R
BIG_GAMMA = 0.0 # Predator Prey
GAMMA  = 0.0
ROW = 0.0
SIGMA = 0.0 



row_array = np.arange(0,6.5,0.25)
SIGMA = 0.75
filename = 'row_test.csv'
r_phi(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N, row_array, w_array, delta, filename)