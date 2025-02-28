import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class KineticAnalysis:
    def __init__(self, time, response):
        self.time = time
        self.response = response

    def curve_fit(self, function, p0):
        """
        Function to fit the curve to the data.

        Parameters
        ----------
        function : function
            The function to fit the data to.
        p0 : list
            The initial guess for the parameters.

        Returns
        -------
        list
            The optimal parameters.
        """
        popt, pcov = curve_fit(function, self.time, self.response, p0=p0)
        return popt

    def plot_curve_fit(self, function, p0):
        """
        Function to plot the curve fit.

        Parameters
        ----------
        function : function
            The function to fit the data to.
        p0 : list
            The initial guess for the parameters.
        """
        popt = self.curve_fit(function, p0)
        plt.plot(self.time, self.response, 'bo', label='Experimental Data')
        plt.plot(self.time, function(self.time, *popt), 'r-', label='Fitted Curve')
        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')
        plt.legend()
        plt.show()
        print("Fitted parameters:", popt)

    def get_function(self, assumption):
        """
        Selects the appropriate function based on the assumption.

        Parameters
        ----------
        assumption : string
            The assumption defining which function to use.

        Returns
        -------
        function
            The corresponding mathematical function.
        """
        if assumption == "baseline+steadystate":
            return self.baseline_steadystate_response
        elif assumption == "response to zero":
            return self.response_to_zero
        elif assumption == "response to steady state":
            return self.response_to_steady_state
        elif assumption == "typical_association":
            return self.typical_association
        elif assumption == "typical_dissociation":
            return self.typical_dissociation
        else:
            raise ValueError("Invalid assumption provided")

    def fit_data(self, p0, assumption):
        """
        Fit the data using the selected function.

        Parameters
        ----------
        p0 : list
            Initial guess for the parameters.
        assumption : string
            The assumption that defines which function to use.

        Returns
        -------
        param_k, pcov_k
            The optimal parameters and their covariance.
        """
        function = self.get_function(assumption)
        param_k, pcov_k = curve_fit(function, self.time, self.response, p0=p0)
        self.plot_curve_fit(function, p0)
        return param_k, pcov_k

    # Function 1: Baseline falling to steady-state response
    def baseline_steadystate_response(self, t, y_initial, y_final, kon):
        return y_final * (1 - np.exp(-kon * t)) + y_initial

    # Function 2: Response falling to zero
    def response_to_zero(self, t, C, y_initial, kon, koff):
        return (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) + y_initial

    # Function 3: Response falling to steady-state response
    def response_to_steady_state(self, t, y_initial, y_final, D, kon, koff):
        return y_final * (1 - D * np.exp(-kon * t) + (D - 1) * np.exp(-koff * t)) + y_initial

    # Function 4: Typical association
    def typical_association(self, t, y_final, conc, kon, koff):
        kd = koff / kon
        return ((y_final * conc) / (kd + conc)) * (1 - np.exp((-1 * (kon * conc + koff)) * t))

    # Function 5: Typical dissociation
    def typical_dissociation(self, t, y_initial, koff):
        return y_initial * np.exp(-koff * t)
