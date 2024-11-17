# We load the relevant packages
from scipy import integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import scipy.constants as scp_const

import pdb

import warnings
warnings.filterwarnings("ignore")


'#----------FUNCTIONS----------#'

# Defining the Lane-Emdem equation as a pair of 
# ODE to be solved by Runge-Kutta methods
def lane_emden(xi,th,n):
    return ([th[1],(-th[0]**n)-(2/xi)*th[1]])

# Stopping condition for the computation is to reach
# a value for theta=0 
def selfdestruct(xi,th,n):
    return(th[0])

selfdestruct.terminal = True

def th_n0(xi):
    return (1-xi**2/6)

def th_n1(xi):
    return np.sin(xi)/xi

def pad_list(len_to_fill,list_to_fill,fill_with=''):
    pad_arr = np.array([fill_with]*(len_to_fill-len(list_to_fill)))
    new_list = np.append(list_to_fill,pad_arr)
    return new_list

def Dn(xi_1,tau_xi_1):
    return ((-(3/xi_1)*tau_xi_1)**(-1))
    

def Mn(xi_1,tau_xi_1):
    return (-(xi_1**2)*tau_xi_1)


def Bn(n,Mn):
    return ( (1/(n+1)) * Mn**(-2/3))
    
    
def rho_rat(n,th):
    return th**n

def paren(xi,tau):
    return xi**2 * tau
    

def main():
    
    '#----------POLYTROPIC CONSTANTS----------#'
    
    # Range of vaulues to compute the derivative
    xi_0 = 0.00000000000000001
    xi_max = 100

    # Initial condicions
    theta_0 = 1 # Initial condition for theta
    tau_0 = 0.0000000001 # Initial condition for theta derivative

    # Points at whose the derivative is computed
    cdf = 5000 # Resolution of the computation
    cdf = 1000
    
    xi_points = xi_max*cdf

    # Defining the dictionaries to store the
    # polytropic constants for each index
    xi_dict = {} # Values for xi, independent variable
    theta_dict = {} # Values for theta, dependent value
    tau_dict = {} # Values for tau, theta derivative
    xi_1_dict = {} # Values of xi 1, at which theta is zero
    tau_xi_1_dict = {} # Values of theta derivative at xi 1
    rho_rat_dict = {} # Ratio of densities for each index

    # Defining with n values we want to compute
    nn_list = [i for i in np.arange(0,5.001,0.001)]

    # Dataframe to store the polytropic constants
    df_cons = pd.DataFrame(columns = ['n','Dn','Mn','Rn','Bn'])

    # Doing the computation for each index
    for nn in nn_list:
        
        nn = nn.round(3)
        
        # Avoiding repeated values between iterations
        Rn_values = [np.nan]
        xi_1 = np.nan
        Dn_values = np.nan
        Mn_values = np.nan
        Bn_values = np.nan
        
        # Solving the Lane-Emden differential equation
        solution = solve_ivp(lane_emden,[xi_0,xi_max],[theta_0,tau_0],
                            t_eval = np.linspace(xi_0,xi_max,xi_points),args=(nn,),
                            events=(selfdestruct)
                            )
        
        # Values of xi
        xi_values = solution.t
        #xi_values_pad = pad_list(xi_points,xi_values,np.nan)
        xi_dict[nn] = xi_values
        
        # Values of theta
        theta_values = solution.y[0]
        #theta_values_pad = pad_list(xi_points,theta_values,np.nan)
        theta_dict[nn] = theta_values
        
        if nn == 0:
            theta_values_n0 = th_n0(xi_values)
            
        elif nn == 1:
            theta_values_n1 = th_n1(xi_values)
        
        # Values of tau
        tau_values = solution.y[1]
        #tau_values_pad = pad_list(xi_points,tau_values,np.nan)
        tau_dict[nn] = tau_values
        
        # Values of tau at xi_1
        tau_xi_values = solution.y[1,-1]
        tau_xi_1_dict[nn] = tau_xi_values 
        
        # Rn are the xi_1 all the values
        # at witch theta = 0
        # Due to the terminal option, 
        # this should be a single value xi 1
        Rn_values = solution.t_events[0]
        if len(Rn_values) != 0:
            xi_1 = Rn_values[0]
        else:
            # In case theta never reaches zero
            # the last xi value is chosen
            xi_1 = xi_values[-1]
        
        # xi 1 would be the first value 
        xi_1_dict[nn] = xi_1
        
        # Computing the constants values
        Dn_values = Dn(xi_1,tau_xi_values)
        Mn_values = Mn(xi_1,tau_xi_values)
        Bn_values = Bn(nn,Mn_values)
        
        # Computing the ratios
        rho_rat_values = rho_rat(nn,theta_values)
        rho_rat_dict[nn] = rho_rat_values

        # Adding the constants to the dataframe
        df_cons.loc[len(df_cons)] =[nn,Dn_values.round(3),
                                    Mn_values.round(2),
                                    xi_1.round(2),
                                    Bn_values.round(3)]
        
        # In case user would like to print
        # al the information in the terminal
        if '-p' in sys.argv:
            print('\n#---------#')
            print(f'\nValue of n: {nn}')
            print(f'\nValues of xi:\n{xi_values}')
            print(f'\nValues of xi:\n{xi_values}')
            print(f'\nValues of theta:\n{theta_values}') 
            print(f'\nValues of tau:\n{tau_values}')
            print(f'\nValues of tau in xi_1:\n{tau_xi_values}') 
            print(f'\nValues of Rn: {Rn_values}')
            print(f'\nValues of xi_1: {xi_1}')
            print(f'\nValues of Dn: {Dn_values}')
            print(f'\nValues of Mn: {Mn_values}')         
            print(f'\nValues of Bn: {Bn_values}')

            
    # Showing the results for the constants
    #print(f'Polytropic constants\n{df_cons}') 
    df_cons.to_csv(f'./csv_constants.csv',sep=',',header=True)
    
    
    nnn_list = [i for i in np.arange(0,5.5,0.5)]
    
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  # Use a serif font for LaTeX rendering
    plt.rc('font', size=16)  # Adjust size to your preference
    # Define the LaTeX preamble with siunitx
    plt.rcParams['text.latex.preamble'] = r'''
                \usepackage{siunitx}
                \sisetup{
                  detect-family,
                  separate-uncertainty=true,
                  output-decimal-marker={.},
                  exponent-product=\cdot,
                  inter-unit-product=\cdot,
                }
                \DeclareSIUnit{\cts}{cts}
                '''

    # Plotting theta values for each n
    f1 = plt.figure(1)
    plt.xlim(xi_0,xi_max)
    plt.ylim(-0.1,1.1)
    
    for nn in nnn_list:    
        plt.plot(xi_dict[nn],theta_dict[nn],label=f'n={nn}')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta$')
    plt.legend()
    
    f2 = plt.figure(5)
    plt.plot(xi_dict[0],theta_dict[0],label=f'n=0')
    plt.plot(xi_dict[0],theta_values_n0,label=f'n=0 An')
    plt.plot(xi_dict[1],theta_dict[1],label=f'n=1')
    plt.plot(xi_dict[1],theta_values_n1,label=f'n=1 An')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta$')
    plt.legend()

    # Ploting Dn values 
    f3 = plt.figure(2)
    plt.plot(df_cons['n'],df_cons['Dn'],label='Dn')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$D_n$')
    plt.legend()

    # Plotting Mn and Bn values
    f4 = plt.figure(3)
    plt.plot(df_cons['n'],df_cons['Mn'],label='Mn')
    plt.plot(df_cons['n'],df_cons['Bn'],label='Bn')
    plt.xlabel(r'$n$')
    plt.legend()

    # Ploting the rate rho/rho_c
    f5 = plt.figure(4)
    for nn in nnn_list:
        plt.plot(xi_dict[nn]/xi_1_dict[nn],rho_rat_dict[nn],label=f'{nn}')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$\rho/\rho_c$')
    plt.legend()


    '#----------COMPARISSON OF THE MODEL----------#'

    # Some physics contants
    c_cons = scp_const.c * 100 # in cm
    R_cons = scp_const.R * 1000 * 100 * 100 # in cm^3
    sb_cons = scp_const.Stefan_Boltzmann * 1000 # in g

    # Solar values
    mu_const = 0.61 
    mass_s = 1.98847*10**30 * 1000 # grams
    r_s = 6.95700*10**8 * 100 # cm

    # Computed constants
    a_cons = 4*sb_cons / c_cons
    beta_cons = (0.003 * (mass_s/mass_s)**2 * mu_const**4 + 1)**(-1/4)
    
    # Defining the dictionaries to store the
    # polytropic constants for each index
    mass_dict = {} # Values for mass
    rad_dict = {} # Values for radius
    temp_dict = {} # Values for temperature
    rho_dict = {} # Values for density
    pre_dict = {} # Values pressure
    par_dict = {} # Values mass integral parenthesis
    der_dict = {} # Values derivative of the parenthesis
    int_dict = {} # Values integral of the derivative

    # Computing models for different n values
    n_model_list = [2.5,3.0,3.5]
    for n_in in n_model_list:
        
        # Creating a dataframe to store the values
        df_model = pd.DataFrame(columns = ['M/Msun','R/Rsun','T','Rho','P','Par','Der','Int'])
        
        model_index = df_cons[df_cons.eq(n_in).any(axis=1)].index.values[0]
        model_index = df_cons.loc[df_cons['n'] == n_in].index.values[0]
        
        #pdb.set_trace()
        
        dn_value = df_cons.loc[model_index]['Dn']
        mn_value = df_cons.loc[model_index]['Mn']
        rn_value = df_cons.loc[model_index]['Rn']
        bn_value = df_cons.loc[model_index]['Bn']

        xi_dict_mod = xi_dict[n_in][1:] # Avoid the close to edge values
        rad_max = np.nanmax(xi_dict[n_in])
        
        # Density in g/cm^3
        rho_c = dn_value * mass_s / ((4 * np.pi * r_s**3) / 3)
        #pdb.set_trace()
        
        print(f'n={n_in}')
        print(f'\nrho_c = {rho_c:.2f} g/cm^3\n')
        
        for pos_xi,xi in enumerate(xi_dict_mod):
            
            # Computing the radius ratio because we want to
            # obtain the results between 0 and 1 Rsun
            rad = xi/rad_max
            
            # Computing the mass
            
            # integrating until radius xi
            xi_int_range = xi_dict_mod[:pos_xi]
            
            # Computing the derivative
            tau_int_range = tau_dict[n_in][:pos_xi]
            par = paren(xi_int_range,tau_int_range)
            der = (paren(xi_int_range+(1/cdf),tau_int_range) - paren(xi_int_range-(1/cdf),tau_int_range)) / (2*(1/cdf))
            inte = integrate.trapezoid(der,x=xi_int_range)
            
            # Method 4
            tau = tau_dict[n_in][pos_xi]
            inte = xi**2 * tau
            
            # Mass at each radius would be
            alpha = r_s / rn_value
            mass_xi = -4*np.pi * (alpha)**3 * rho_c * inte 
            
            mass_rat = mass_xi/mass_s # Mass in units of solar mass
            
            # Computing the density using rho=rho_c * theta^n
            # where each theta corresponds with a xi        
            th = theta_dict[n_in][pos_xi]
            rho = rho_c * th**n_in # Units of g/cm^3
            
            # Computing the temperature
            #pdb.set_trace()
            temp = ( (3*R_cons*(1-beta_cons)) / (a_cons*mu_const*beta_cons) )**(1/3) * rho**(1/3)
            
            # Computing the pressure
            pre = (R_cons * rho * temp) / (beta_cons * mu_const)
            
            # Storing the values in a dataframe
            df_model.loc[len(df_model)] = [mass_rat,rad,temp,rho,pre,par,der,inte]
            
            
        print(df_model)
        df_model.to_csv(f'./csv_model_n{n_in}.csv',sep=',',header=True)
        
        par_dict[n_in] = df_model['Par'][len(df_model)-1]
        der_dict[n_in] = df_model['Der'][len(df_model)-1]
        mass_dict[n_in] = df_model['M/Msun'].to_numpy()
        rad_dict[n_in] = df_model['R/Rsun'].to_numpy()
        temp_dict[n_in] = df_model['T'].to_numpy()
        rho_dict[n_in] = df_model['Rho'].to_numpy()
        pre_dict[n_in] = df_model['P'].to_numpy()
        
        #pdb.set_trace()
        
    # Loading the more complex model to compare with
    df_model_compl = pd.read_csv('./modelo_estelar_complejo.csv', sep=',')
    #df_model_compl.columns = ['M/Msun','R/Rsun','T','Rho','P','L/Lsun','X','Y(He4)','He3','C12','N14','O16']
    
    #print(df_model_compl)

    # Mass as function of radius
    f6 = plt.figure(6)
    for n in n_model_list:
        plt.plot(rad_dict[n],mass_dict[n],label=f'{n}')
    plt.plot(df_model_compl['R/Rsun'],df_model_compl['M/Msun'],label=f'Model')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$M/M_{\odot}$')
    plt.legend()
    
    # Mass as function of radius
    f7 = plt.figure(7)
    for n in n_model_list:
        plt.plot(rad_dict[n],np.log10(rho_dict[n]),label=f'{n}')
    plt.plot(df_model_compl['R/Rsun'],np.log10(df_model_compl['Rho']),label=f'Model')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$log(\rho)$')
    plt.legend()
    
    # Temperature as function of radius
    f8 = plt.figure(8)
    for n in n_model_list:
        plt.plot(rad_dict[n],np.log10(temp_dict[n]),label=f'{n}')
    plt.plot(df_model_compl['R/Rsun'],np.log10(df_model_compl['T']),label=f'Model')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$log(T)$')
    plt.legend()
    
    # Pressure as function of radius
    f9 = plt.figure(9)
    for n in n_model_list:
        plt.plot(rad_dict[n],np.log10(pre_dict[n]),label=f'{n}')
        #print(f'\n{n}')
        #print(rad_dict[n])
    plt.plot(df_model_compl['R/Rsun'],np.log10(df_model_compl['P']),label=f'Model')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$log(P)$')
    plt.legend()
    
    # Pressure as function of radius
    f10 = plt.figure(10)
    for n in n_model_list:
        plt.plot(rad_dict[n][:-1],par_dict[n],label=f'{n}')
        plt.plot(rad_dict[n][:-1],der_dict[n],label=f'{n}')
    plt.xlabel(r'$R/R_{\odot}$')
    plt.ylabel(r'$Function and its derivative$')
    plt.legend()
    
    # Showing the plots
    if '-pl' in sys.argv:
        
        #plt.close(f1) # Theta values for each n
        #plt.close(f2) # Analitycal oomparisson
        #plt.close(f3) # Dn evolution with xi
        #plt.close(f4) # Mn and Bn evolution with xi
        #plt.close(f5) # Rho ratio
        #plt.close(f6) # Mass
        #plt.close(f7) # Density
        #plt.close(f8) # Temperature
        #plt.close(f9) # Pressure
        plt.close(f10) # Derivative     
        
        plt.show()
    
        


'#----------RUNNING THE CODE----------#'

if __name__ == "__main__":
    
    main()
