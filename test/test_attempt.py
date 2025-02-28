import unittest
import numpy as np
import pandas as pd
import os
import sys

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Get the root directory (assuming your script is two levels deep in the directory structure)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
print(root_dir)

# Add the root directory to the Python path
sys.path.append(root_dir)

from rate_kinetics_extract.ratekineticsextract import KineticAnalysis
import numpy as np
import pandas as pd
class TestRateKinetics(unittest.TestCase):
    
    # Dataset for each equation
    time_eq1 = np.array([0.2519927506851981, 0.41026887798956047, 0.5998321467378084, 0.8188421368450075, 1.0893838893303711,
                         1.4629891665720638, 2.0040726715427906, 2.570922058, 3.1377714438624094, 3.7046208300222188,
                         4.271470216182029, 4.838319602341838, 5.405168988501647, 5.972018374661457, 6.538867760821266,
                         7.105717146981076, 7.672566533140885, 8.239415919300694, 8.806265305460503, 9.373114691620312,
                         9.939964077780122, 10.506813463939931, 11.07366285009974, 11.64051223625955, 12.20736162241936,
                         12.490786315499264, 13.057635701659073, 13.624485087818883, 14.191334473978692, 14.474759167058597,
                         15.041608553218406, 15.325033246298311, 15.89188263245812, 16.742156711697834, 17.309006097857644,
                         17.59243079093755, 18.159280177097358, 18.442704870177263, 19.009554256337072, 19.57640364249688,
                         20.426677721736596, 20.71010241, 21.27695180097631, 21.560376494056214, 22.127225880216024,
                         22.694075266375833, 22.977499959455738, 23.544349345615547, 24.111198731775357, 24.39462342485526,
                         24.96147281101507, 25.244897504094975, 25.811746890254785, 26.378596276414594, 26.945445662574404,
                         27.22887035565431, 28.362569127973927, 28.645993821053832, 29.21284320721364, 29.77969259337345,
                         30.063117286453355, 30.34654197953326, 30.91339136569307, 31.48024075185288, 32.04709013801269,
                         32.61393952, 33.18078891033231, 33.74763829649212, 34.31448768265193, 34.88133706881174,
                         35.44818645497155, 35.73161114805145, 36.29846053421126, 36.86530992037107, 37.43215930653088,
                         38.282433385770595, 38.849282771930405, 39.132707465010306, 39.699556851170115, 40.266406237329925])
    response_eq1 = np.array([0.6477658834124753, 1.2563364055299537, 1.919260172752578, 2.4935463402652687, 3.0563424485562996,
                             3.6127792787875466, 4.132032726975005, 4.4465804888577924, 4.622774535927474, 4.724470428867022,
                             4.7883259895499934, 4.816706238742426, 4.833261384104677, 4.842721467168822, 4.846268998317876,
                             4.854546570999002, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.854546570999002, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.854546570999002, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.858094102148056, 4.8557290813820195,
                             4.856911591765037, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195,
                             4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195, 4.858094102148056])
    
    # More datasets for eq2, eq3, eq4, eq5 follow...
    # Repeat for each equation's corresponding time and response dataset

    # Test for Equation 1
    def test_fit_data_equation_1(self):
        p0 = [0, 5, 1]
        expected_params = np.array([0, 5, 1])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq1, 'RU 1nM': self.response_eq1})
        
        params, _ = fit_data(self.time_eq1, self.response_eq1, p0, data)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

    # Test for Equation 2
    def test_fit_data_equation_2(self):
        p0 = [1, 1, 0.1, 0.09]
        expected_params = np.array([1, 1, 0.1, 0.09])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq2, 'RU 1nM': self.response_eq2})
        
        params, _ = fit_data(self.time_eq2, self.response_eq2, p0, data)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

    # Test for Equation 3
    def test_fit_data_equation_3(self):
        p0 = [0.5, 3, 0.15]
        expected_params = np.array([0.5, 3, 0.15])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq3, 'RU 1nM': self.response_eq3})
        
        params, _ = fit_data(self.time_eq3, self.response_eq3, p0, data)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

    # Test for Equation 4
    def test_fit_data_equation_4(self):
        p0 = [2, 0.5, 1]
        expected_params = np.array([2, 0.5, 1])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq4, 'RU 1nM': self.response_eq4})
        
        params, _ = fit_data(self.time_eq4, self.response_eq4, p0, data)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

    # Test for Equation 5
    def test_fit_data_equation_5(self):
        p0 = [0, 0.2, 0.3]
        expected_params = np.array([0, 0.2, 0.3])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq5, 'RU 1nM': self.response_eq5})
        
        params, _ = fit_data(self.time_eq5, self.response_eq5, p0, data)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

if __name__ == '__main__':
    unittest.main()