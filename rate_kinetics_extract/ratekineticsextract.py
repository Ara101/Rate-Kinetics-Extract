import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class KineticAnalysis:
    def __init__(self, time, response):
        self.time = time
        self.response = response
        
    
    def curve_fit(self, function, p0):
        """
        Function to fit the curve to the data
        
        Parameters
        ----------
        function : function
            The function to fit the data to
        p0 : list
            The initial guess for the parameters
        
        Returns
        -------
        list
            The optimal parameters
        """
        popt, pcov = curve_fit(function, self.time, self.response, p0=p0)
        return popt
    
    def plot_curve_fit(self, function, p0):
        """
        Function to plot the curve fit
        
        Parameters
        ----------
        function : function
            The function to fit the data to
        p0 : list
            The initial guess for the parameters
        """
        popt = self.curve_fit(function, p0)
        plt.plot(self.time, self.response, 'b-', label='data')
        plt.plot(self.time, function(self.time, *popt), 'r-', label='fit')
        plt.show()

    # Function 1: Baseline falling to steady state response
    def baseline_steadystate_response(self, t, y_initial, y_final, kon):
        """
        Function to find the kon value from the data. 
        Assuming we know the baseline and the steady state response, we can find the kon value.
        The equation is y(t) = y_final * (1 - exp(-kon * t)) + y_intial

        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_intial : float
            The baseline value of the response
        y_final : float
            The steady state value of the response
        t : float
            The time value

        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_final * (1 - np.exp(-kon * t)) + y_initial

    # Function 2: Response falling to zero
    def response_to_zero(self, t, C, y_initial, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming we know that the response goes to zero
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        C : float
            The initial rate of signaling
        y_initial : float
            The baseline value of the response
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) + y_initial

    # Function 3: Response falling to steady state response
    def response_to_steady_state(self, t, y_initial, y_final, D, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming we know the steady state response
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_initial : float
            The initial rate of signaling
        y_final : float
            The final rate of signaling
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_final * (1 - D * np.exp(-kon * t) + (D - 1) * np.exp(-koff * t)) + y_initial

    # Function 4: Typical association
    def typical_association(self, t, y_final, conc, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming it is a typical association function
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_final : float
            The final rate of signaling
        t : float
            The time value
        conc : float
            The concentration of the substance
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        kd = koff / kon
        return ((y_final * conc) / (kd + conc)) * (1 - np.exp((-1 * (kon * conc + koff)) * t))

    # Function 5: Typical dissociation
    def typical_dissociation(self, t, y_initial, koff):
        """
        Function to find the koff values from the data
        Assuming it is a typical dissociation function
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_initial : float
            The initial rate of signaling
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_initial * np.exp(-koff * t)
    

    def get_function(self):
        """Select the appropriate function based on the assumption."""
        if self.assumption == "baseline+steadystate":
            return lambda t, y_initial, y_final, kon: y_final * (1 - np.exp(-kon * t)) + y_initial

        elif self.assumption == "response to zero":
            return lambda t, C, y_initial, kon, koff: (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) + y_initial

        elif self.assumption == "response to steady state":
            return lambda t, y_initial, y_final, D, kon, koff: y_final * ((1 - D * np.exp(-kon * t)) + (D - 1) * np.exp(-koff * t)) + y_initial

        elif self.assumption == "typical_association":
            return lambda t, y_final, conc, kon, koff: ((y_final * conc) / (koff / kon + conc)) * (1 - np.exp((-1 * (kon * conc + koff)) * t))

        elif self.assumption == "typical_dissociation":
            return lambda t, y_initial, koff: y_initial * np.exp(-koff * t)

        else:
            raise ValueError("Invalid assumption provided")

    def fit_data(self, p0):
        """
        Fit the data using the selected function.

        Parameters
        ----------
        p0 : list
            Initial guess for the parameters.

        Returns
        -------
        tuple
            The optimized parameters and covariance.
        """
        time = self.data.iloc[:, 0]
        response = self.data.iloc[:, 1]

        param_k, pcov_k = curve_fit(self.function, time, response, p0=p0)
        self.plot_fitted_curve(param_k)
        return param_k, pcov_k

    def plot_fitted_curve(self, param_k):
        """
        Plot the fitted curve along with the experimental data.

        Parameters
        ----------
        param_k : list
            The fitted parameters.
        """
        time = self.data.iloc[:, 0]
        response = self.data.iloc[:, 1]

        plt.plot(time, response, 'o', label='Experimental Data')
        fitted_response = self.function(time, *param_k)
        plt.plot(time, fitted_response, '-', label='Fitted Curve')
        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')
        plt.legend()
        plt.show()

        print("Fitted parameters: ", param_k)

