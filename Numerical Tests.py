import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import numba as nb
import multiprocessing as mp
import csv

def beta_kappa_correlations(N,row,column,gamma, sigma):
    correlation_values_b_k = np.zeros((N,2), dtype = float)
    
    for i in range(N):
        #Generate random normal values
        rand_val_1 = np.random.normal(0, 1)
        rand_val_2 = np.random.normal(0, 1)
    
        beta_i = rand_val_1*np.sqrt(row/N)
        beta_i *= sigma/np.sqrt(N)
        #Different cases for when r=0 as wd usually do DIV by 0
        if row ==0:
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

@nb.jit(nopython = True) # delete this line if you dont have numba, its just for efficiency
def may_matrix(N, sigma, mu, gamma, b_gamma, r, c, bk_array, omega_matrix): 
    result = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z_ij = bk_array[i,0] + bk_array[j,1] + omega_matrix[i,j]
            result[i,j] = mu/N + z_ij    
    np.fill_diagonal(result,0) # Forces diagonal components to = 0 due to LV eqn below
    return result

def lotka_volterra(X, t, alpha):
    if X.mean()>50:
        raise Exception
    else:
        dxdt = X*(1 -X + alpha @ X)

    return dxdt


#Run-time constants
T_MIN = 0.0
T_MAX = 100.0
DT = 0.01
iterations = 20 #number of different runs to average over
N = 100 #number of species

GAMMA = 0.0 #Correlation sharing single index
COLUMN = 0.0
BIG_GAMMA = 0.0 # Predator Prey
ROW = 5.0
sigma_arr=  np.sqrt(np.arange(0,10,.25))# np.sqrt(np.array([1]))
mu_arr = np.arange(-4, 1, 1)


# CREATION OF ARRAYS etc

def run_lv(args):
    SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
    num_divergences = 0
    values = np.zeros((7,))
    values[2] = MU
    values[3] = BIG_GAMMA
    values[4] = ROW
    values[5] = GAMMA
    values[6] = COLUMN
    for i in range(iterations): #looping over different runs of LV to average it out.
        alpha = may_matrix(N, SIGMA, MU, GAMMA, BIG_GAMMA, ROW, COLUMN, beta_kappa_correlations(N,ROW,COLUMN,GAMMA, SIGMA), omega_correlations(N,ROW,COLUMN,GAMMA,BIG_GAMMA, SIGMA)) #Generating may matrix
        try:
            t = np.arange(T_MIN, T_MAX+DT, DT)
            x_0 = np.full((N),1) #Start all at x=1...shouldn't matter 
            integrate = int.odeint(lotka_volterra, x_0, t, args = (alpha,)) # integration!            
        except:
            num_divergences += 1        
    if (num_divergences>0):
        values[0] = SIGMA
    if num_divergences == iterations:
        values[1] = SIGMA
    return values




args_list = [(SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N) for MU in mu_arr for SIGMA in sigma_arr]
if __name__ == '__main__':
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_lv, args_list)
    pool.close()
    pool.join()
    # process the results
    values = np.zeros((7, mu_arr.size)) ###############################################
    for i, args in enumerate(args_list):
        SIGMA, BIG_GAMMA, ROW, GAMMA, COLUMN, MU, iterations, N = args
        ii = np.where(mu_arr == MU)[0][0]
        values[:, ii] = results[i]

    # save the results to a CSV file
    header = ['MU', '1ST DIV', 'ALL DIV'] ###############################################
    with open('stability_check.csv', 'w', newline='') as f: ###############################################
        writer = csv.writer(f)
        writer.writerow(['BIG_GAMMA', BIG_GAMMA])
        writer.writerow(['ROW', ROW])
        writer.writerow(['COLUMN', COLUMN])
        writer.writerow(['GAMMA', GAMMA])
        writer.writerow(header)
        for ii, MU in enumerate(mu_arr):
            writer.writerow([MU, values[0, ii]**2, values[1, ii]**2])

    # make the figures
    fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
    ax_1.scatter(mu_arr, values[0, :]**2)
    ax_1.scatter(mu_arr, values[1, :]**2)
    # ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 

    ax_1.set_title('Stability')
    ax_1.set_xlabel('$\\mu$')
    ax_1.set_ylabel('Variance of Interaction Strength, $\\sigma^2$')

    plt.show()

output = np.zeros((2,mu_arr.size))

for ii, MU in enumerate(mu_arr):
    xx = True
    for SIGMA in sigma_arr: # looping over different values of sigma for fixed r (& gamma ... etc.)
        num_divergences = 0
        print("Mu: ", MU, "Var: ", SIGMA**2)
        for i in range(iterations): #looping over different runs of LV to average it out.
            alpha = may_matrix(N, SIGMA, MU, GAMMA, BIG_GAMMA, ROW, COLUMN, beta_kappa_correlations(N,ROW,COLUMN,GAMMA, SIGMA), omega_correlations(N,ROW,COLUMN,GAMMA,BIG_GAMMA, SIGMA)) #Generating may matrix
            try:
                t = np.arange(T_MIN, T_MAX+DT, DT)
                x_0 = np.full((N),1) #Start all at x=1...shouldn't matter 
                integrate = int.odeint(lotka_volterra, x_0, t, args = (alpha,)) # integration!            
            except:
                integrate = np.zeros((t.size, N))
                num_divergences += 1        
        # print("        numb of div: ", num_divergences, ". Out of: ", iterations)
        if xx:
            if (num_divergences>0):
                output[0,ii] = SIGMA
                xx = False
        if num_divergences == iterations:
            output[1,ii] = SIGMA
            break
# make the figures
fig_1, ax_1 = plt.subplots(1, 1, figsize = (15, 7))
ax_1.scatter(mu_arr, output[0, :]**2)
ax_1.scatter(mu_arr, output[1, :]**2)
# ax_1.plot(Variance(w_array,BIG_GAMMA,delta,ROW), Biomass(w_array,BIG_GAMMA,delta,ROW, GAMMA, MU)) 

ax_1.set_title('Stability')
ax_1.set_xlabel('$\\mu$')
ax_1.set_ylabel('Variance of Interaction Strength, $\\sigma^2$')
plt.show()
