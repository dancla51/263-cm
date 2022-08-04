# ENGSCI263: Lab Exercise 2
# lab2.py

# PURPOSE:
# IMPLEMENT a lumped parameter model and CALIBRATE it to data.

# PREPARATION:
# Review the lumped parameter model notes and use provided data from the kettle experiment.

# SUBMISSION:
# - Show your calibrated LPM to the instructor (in Week 3).

# imports
import matplotlib.pyplot
import numpy
import numpy as np
from matplotlib import pyplot as plt

def ode_model(t, x, q, a, b, x0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        x : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    # Calculates derivative and returns it
    dxdt = a * q - b * (x - x0)
    return dxdt

def solve_ode(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f. (q, a, b, x0)

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    # Create time vector
    tvec = np.arange(t0, t1, dt)
    tvec = np.append(tvec, t1)
    # Create x vector
    xvec = np.zeros(len(tvec))
    xvec[0] = x0

    # Loop through for each step
    for i in range(len(tvec)-1):
        # Do Euler method
        f0 = f(tvec[i], xvec[i], *pars)
        xeuler = xvec[i] + dt * f0
        f1 = f(tvec[i+1], xeuler, *pars)
        xvec[i+1] = xvec[i] + dt * (f0 + f1) / 2

    return tvec, xvec


def plot_benchmark():
    ''' Compare analytical and numerical solutions.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.
        
    '''
    # Analytical solution x = e^(-t)-1

    # Initialise parameters
    pars = [-1, 1, 1, 0]
    # Calls the function to find numerical solution
    tnum, xnum = solve_ode(ode_model, 0, 10, 0.1, 0, pars)

    # Finds analytical results
    xanalytical = np.zeros(len(tnum))
    for i in range(len(tnum)):
        xanalytical[i] = np.exp(tnum[i] * -1) - 1

    # Plot Numerical
    plt.plot(tnum, xnum, 'kx')
    plt.ylabel("X value")
    plt.xlabel("time")
    plt.title("Benchmarking")

    # Plot Analytical
    plt.plot(tnum, xanalytical, '-r')
    plt.show()

    # Compute relative error between numerical and analytical
    errdiff = np.zeros(len(xnum))
    for i in range(len(xnum)):
        errdiff[i] = abs((xanalytical[i] - xnum[i])/xanalytical[i])

    # Plot relative error
    plt.plot(tnum, errdiff, 'b-')
    plt.yscale("log")
    plt.title("Error Analysis")
    plt.xlabel("time")
    plt.ylabel("relative error")
    plt.show()

    # Conduct timestep convergence analysis
    # Create arrays
    invtimesteparray = np.arange(1, 3.1, 0.1)
    xat10 = np.zeros(len(invtimesteparray))
    # Loop through each value and calculate x
    for i in range(len(invtimesteparray)):
        tt, xt = solve_ode(ode_model, 0, 10, 1/invtimesteparray[i], 0, pars)
        xat10[i] = xt[-1]

    plt.plot(invtimesteparray, xat10, 'bo')
    plt.xlabel("1 over timestep")
    plt.ylabel("X at t=10")
    plt.title("timestep convergence")
    plt.show()





def load_kettle_temperatures():
    ''' Returns time and temperature measurements from kettle experiment.

        Parameters:
        -----------
        none

        Returns:
        --------
        t : array-like
            Vector of times (seconds) at which measurements were taken.
        T : array-like
            Vector of Temperature measurements during kettle experiment.

        Notes:
        ------
        It is fine to hard code the file name inside this function.

        Forgotten how to load data from a file? Review datalab under Files/cm/
        engsci233 on the ENGSCI263 Canvas page.
    '''
    # Loads data from kettledata
    kettle_data = np.loadtxt("263_Kettle_Experiment_22-07-19.csv", delimiter=',', skiprows=7)

    # Pulls out time and temperature columns
    timevec = kettle_data[:, 0]
    tempvec = kettle_data[:, 3]

    #returns
    return timevec, tempvec



def interpolate_kettle_heatsource(t):
    ''' Return heat source parameter q for kettle experiment.

        Parameters:
        -----------
        t : array-like
            Vector of times at which to interpolate the heat source.

        Returns:
        --------
        q : array-like
            Heat source (Watts) interpolated at t.

        Notes:
        ------
        This doesn't *have* to be coded as an interpolation problem, although it 
        will help when it comes time to do your project if you treat it that way. 

        Linear interpolation is fine for this problem, no need to go overboard with 
        splines. 
        
        Forgotten how to interpolate in Python, review sdlab under Files/cm/
        engsci233 on the ENGSCI263 Canvas page.
    '''

    # Loads data from kettledata
    kettle_data = np.loadtxt("263_Kettle_Experiment_22-07-19.csv", delimiter=',', skiprows=7)

    # Pulls out time column
    tv = kettle_data[:, 0]

    # Pulls out voltage and current columns
    voltvec = kettle_data[:, 1]
    currvec = kettle_data[:, 2]

    # Initialise qv values to zero
    qv = [0] * len(kettle_data)

    # Calculates qv at given times
    for i in range(len(qv)):
        qv[i] = voltvec[i] * currvec[i]
    # print(qv)

    # Interpolates based on inputs t which is array of time (linear)
    q = np.interp(t, tv, qv)

    # suggested approach
    # hard code vectors tv and qv which define a piecewise heat source for your kettle 
    # experiment
    # use a built in Python interpolation function 
    return q

def plot_kettle_model():
    ''' Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and 
        plot the kettle LPM for hard coded parameters, and then either display the 
        plot to the screen or save it to the disk.

    '''
    # Loads kettle data
    tvec, tempvec = load_kettle_temperatures()

    # Calibrate parameters (initial a = 1/(997*4182*500*10**-6), b = a * 0.48 * 0.02 / 0.01
    a = 1.1/(997*4182*500*10**-6)
    b = a * 0.48 * 0.028 / 0.01
    print("a = ", a, " and b = ", b)
    ambtemp = 22
    pars = [1, a, b, ambtemp]
    timestep = 0.01
    t0 = 0
    t1 = 1200
    intltemp = 22
    tres, Tres = solve_ode_kettle(ode_model, t0, t1, timestep, intltemp, pars)

    plt.plot(tvec, tempvec, 'rx')
    plt.plot(tres, Tres, 'b-')

    plt.show()



def solve_ode_kettle(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f. (q, a, b, x0)

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method.

        Overwrite q(t) with interpolated values

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    # Create time vector
    tvec = np.arange(t0, t1, dt)
    tvec = np.append(tvec, t1)
    # Create q vector
    qvec = interpolate_kettle_heatsource(tvec)
    # Create x vector
    xvec = np.zeros(len(tvec))
    xvec[0] = x0

    # Loop through for each step
    for i in range(len(tvec)-1):
        # Do Euler method
        pars[0] = qvec[i]
        f0 = f(tvec[i], xvec[i], *pars)
        xeuler = xvec[i] + dt * f0
        f1 = f(tvec[i+1], xeuler, *pars)
        xvec[i+1] = xvec[i] + dt * (f0 + f1) / 2

    return tvec, xvec



if __name__ == "__main__":
    # Plots the kettle data
    plot_kettle_model()


