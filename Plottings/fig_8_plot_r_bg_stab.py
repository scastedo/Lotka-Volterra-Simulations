# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as int
# from scipy.special import erf
# cmap = plt.get_cmap('jet_r')

# def W_calc(delta):
#     output = np.zeros((delta.size,3), dtype = float)
#     for i in range(0,delta.size):
#         output[i,0] = 0.5*(1 + erf(delta[i]/np.sqrt(2)))
#         output[i,1] = 0.5*((np.sqrt(2/np.pi)*np.exp(-(delta[i]**2)/2))+delta[i]*(1 + erf(delta[i]/np.sqrt(2))))
#         output[i,2] = output[i,0] + delta[i]*output[i,1]
#     return output

# def Variance(w,big_gamma,delta,r): #This function calculates the variance of the set of the data in terms of the w variables
#     var_array = np.zeros((delta.size), dtype = float)
#     for i in range(0,delta.size):
#         var_array[i] = w[i,2]/((w[i,2] + ((big_gamma * w[i,0])/(1 + r*(w[i,1]**2/w[i,2]))))**2 * (1 + r*(w[i,1]**2/w[i,2])))
#     return var_array

# def Biomass(w,big_gamma,delta,r,gamma, MU):
#     bio_array = np.zeros((delta.size), dtype = float)
#     for i in range(0,delta.size):
#         b = (w[i,2] * delta[i])*(1 + r*(w[i,1]**2/w[i,2])) - gamma*w[i,0]*w[i,1]
#         b/= (w[i,1] * (w[i,2]*(1 + r*(w[i,1]**2/w[i,2])) + (big_gamma * w[i,0])))
#         b-= MU
#         bio_array[i] = 1/b
#     return bio_array

# MU = -8.0
# COLUMN = 0.0 # GREATER THAM GAMMA^2/R
# BIG_GAMMA = 0.0 # Predator Prey
# GAMMA  = 0.0
# ROW = 1.0
# SIGMA = 1.0 
# # Parameter Constants
# delta = np.linspace(-5,5, num = 100000)
# w_array = W_calc(delta)

# # Load CSV data into a Pandas DataFrame
# df = pd.read_csv('Plottings/varylg_r_1.csv', skiprows=4)


# # Set up plot
# fig, ax = plt.subplots(figsize=(10, 7))
# sigma_array = np.sqrt(np.array([0.6,1,2]))
# #sigma_array = np.sqrt(np.arange(0.001,10,0.1))
# # Plot each GAMMA as a separate line
# for i,SIGMA in enumerate(sigma_array):
#     x = (1/(SIGMA*np.sqrt(w_array[:,0]))-1)
#     y = -delta/w_array[:,1]
#     ax.plot(x,y,label = r'%.1f' %SIGMA**2, color = cmap(float(i)/len(sigma_array))) 
#     ax.fill_betweenx(y,x, color=cmap(float(i)/len(sigma_array)), alpha=0.2)


# # i=0
# # for GAMMA in (df['GAMMA'].unique()):
# #     data = df[df['GAMMA'] == GAMMA]
# #     ax.scatter(data['VARIANCE'], data['FRACTION'], label=f'$\gamma$ = {GAMMA:.1f}', color = cmap(float(i)/5))
# #     i+=1
# # # Add labels and legend
# ax.set_xlabel(r'$\Gamma$', fontsize = 24)
# ax.set_ylabel(r'$r$', fontsize = 28)
# plt.tick_params(axis='both', which='major', labelsize=17)  # Adjust the labelsize as needed
# ax.legend(title = r'$\sigma^2$', title_fontsize = 20, fontsize = 17, loc='upper right')
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# plt.ylim(0,5)
# plt.xlim(0,1)
# # Display plot
# # plt.savefig('fig_8_plot_r_bg_stab.png', dpi=500)
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import erf
import matplotlib.ticker as ticker

# Define constants and parameters
gammavec = np.array([-0.5, -0.1, 0.5])
mu = -0.1
numr = 101

# Preallocate arrays
critgvec = np.zeros((len(gammavec), numr))
critrvec = np.zeros(len(gammavec))
rvecvec = np.zeros((len(gammavec), numr))

# Define functions
def w0funct(mydelta):
    return 0.5 * (1.0 + erf(mydelta / np.sqrt(2.0)))

def w1funct(mydelta):
    return 0.5 * (np.sqrt(2.0 / np.pi) * np.exp(-mydelta**2 / 2.0) + mydelta * (1.0 + erf(mydelta / np.sqrt(2))))

def w2funct(mydelta):
    return w0funct(mydelta) + mydelta * w1funct(mydelta)

# Calculation loop
start = -0.1
for i, gamma in enumerate(gammavec):
    sigma = np.sqrt(2) * 1.1 / (1 + gamma)
    
    sol1 = root(lambda mydelta: sigma**2 - 1 / (w0funct(mydelta) * (1 + gamma)**2), start)
    mydelta = sol1.x[0]
    critr = -mydelta / w1funct(mydelta)
    critrvec[i] = critr
    rmin = critr
    rmax = 10 * critr
    rvec = np.linspace(rmin, rmax, numr)
    start = mydelta
    
    for j, r in enumerate(rvec):
        rvecvec[i, j] = r
        sol = root(lambda mydelta: sigma**2 - w2funct(mydelta) * (1 + r * w1funct(mydelta)**2 / w2funct(mydelta)) / 
                    (w2funct(mydelta) * (1 + r * w1funct(mydelta)**2 / w2funct(mydelta)) + gamma * w0funct(mydelta))**2, start)
        mydelta = sol.x[0]
        critgvec[i, j] = (mydelta * w2funct(mydelta) * (1 + r * w1funct(mydelta)**2 / w2funct(mydelta)) - 
                          mu * w1funct(mydelta) * (w2funct(mydelta) * (1 + r * w1funct(mydelta)**2 / w2funct(mydelta)) + 
                          gamma * w0funct(mydelta))) / (w0funct(mydelta) * w1funct(mydelta))

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(len(gammavec)):
    ax.plot(critgvec[-1-i], rvecvec[-1-i], label=gammavec[-1-i])
    ax.fill_between(critgvec[-1-i], rvecvec[-1-i],critrvec[0], alpha=0.2 )

ax.plot(np.linspace(-10,0, 100), [critrvec[0]] * 100, '--', color = 'black')

ax.set_xlabel(r'$\gamma$', fontsize=24)
ax.set_ylabel(r'$r$', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=17)
ax.legend(fontsize=17, loc='upper right', title = r'$\Gamma$', title_fontsize = 20)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.ylim(0, 5)
plt.xlim(min(critgvec.flatten()), max(critgvec.flatten()))
plt.xlim(-3, max(critgvec.flatten()))
# plt.subplots_adjust(left=0.2, right=0.8, top=0.8)
plt.savefig('plot_lg_r_stab.png', dpi=500)

plt.show()
