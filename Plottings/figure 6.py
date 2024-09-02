import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
from scipy.special import erf
cmap = plt.get_cmap('jet_r')

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

def Biomass(w,big_gamma,delta,r,gamma, MU):
    bio_array = np.zeros((delta.size), dtype = float)
    for i in range(0,delta.size):
        b = (w[i,2] * delta[i])*(1 + r*(w[i,1]**2/w[i,2])) - gamma*w[i,0]*w[i,1]
        b/= (w[i,1] * (w[i,2]*(1 + r*(w[i,1]**2/w[i,2])) + (big_gamma * w[i,0])))
        b-= MU
        bio_array[i] = 1/b
    return bio_array

MU = -1.0
COLUMN = 0.0 # GREATER THAM GAMMA^2/R
BIG_GAMMA = 0.0 # Predator Prey
GAMMA  = 0.0
ROW = 1.0
SIGMA = 0.0 
# Parameter Constants
delta = np.linspace(-5,5, num = 100000)
w_array = W_calc(delta)

# Load CSV data into a Pandas DataFrame
df = pd.read_csv('varylg_r_1.csv', skiprows=4)


# Set up plot
fig, ax = plt.subplots()

# Plot each GAMMA as a separate line
i=0
ax.plot(Variance(w_array,BIG_GAMMA, delta,ROW),w_array[:,0]) 
for GAMMA in df['GAMMA'].unique():
    data = df[df['GAMMA'] == GAMMA]
    ax.scatter(data['VARIANCE'], data['FRACTION'], label=f'$\gamma$ = {GAMMA:.1f}', color = cmap(float(i)/5))
    i+=1
# Add labels and legend
ax.set_xlabel(r'$\sigma^2$', fontsize = 17)
ax.set_ylabel(r'$\phi$', fontsize = 17)
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjust the labelsize as needed
ax.legend()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.ylim(0.5,1.01)
plt.xlim(0,1)
# Display plot
plt.savefig('fig_6.png', dpi=500, bbox_inches='tight')

plt.show()