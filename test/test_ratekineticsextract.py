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

from rate_kinetics_extract.RateKineticsExtract import KineticAnalysis
import numpy as np
import pandas as pd
class TestRateKinetics(unittest.TestCase):
    
    # Dataset for each equation
    time_eq1 = np.array([0.2519927506851981, 0.41026887798956047, 0.5998321467378084, 0.8188421368450075, 1.0893838893303711, 1.4629891665720638, 2.0040726715427906, 2.570922058, 3.1377714438624094, 3.7046208300222188, 4.271470216182029, 4.838319602341838, 5.405168988501647, 5.972018374661457, 6.538867760821266, 7.105717146981076, 7.672566533140885, 8.239415919300694, 8.806265305460503, 9.373114691620312, 9.939964077780122, 10.506813463939931, 11.07366285009974, 11.64051223625955, 12.20736162241936, 12.490786315499264, 13.057635701659073, 13.624485087818883, 14.191334473978692, 14.474759167058597, 15.041608553218406, 15.325033246298311, 15.89188263245812, 16.742156711697834, 17.309006097857644, 17.59243079093755, 18.159280177097358, 18.442704870177263, 19.009554256337072, 19.57640364249688, 20.426677721736596, 20.71010241, 21.27695180097631, 21.560376494056214, 22.127225880216024, 22.694075266375833, 22.977499959455738, 23.544349345615547, 24.111198731775357, 24.39462342485526, 24.96147281101507, 25.244897504094975, 25.811746890254785, 26.378596276414594, 26.945445662574404, 27.22887035565431, 28.362569127973927, 28.645993821053832, 29.21284320721364, 29.77969259337345, 30.063117286453355, 30.34654197953326, 30.91339136569307, 31.48024075185288, 32.04709013801269, 32.61393952, 33.18078891033231, 33.74763829649212, 34.31448768265193, 34.88133706881174, 35.44818645497155, 35.73161114805145, 36.29846053421126, 36.86530992037107, 37.43215930653088, 38.282433385770595, 38.849282771930405, 39.132707465010306, 39.699556851170115, 40.266406237329925])
    response_eq1 = np.array([0.6477658834124753, 1.2563364055299537, 1.919260172752578, 2.4935463402652687, 3.0563424485562996, 3.6127792787875466, 4.132032726975005, 4.4465804888577924, 4.622774535927474, 4.724470428867022, 4.7883259895499934, 4.816706238742426, 4.833261384104677, 4.842721467168822, 4.846268998317876, 4.854546570999002, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.854546570999002, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.854546570999002, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.858094102148056, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195, 4.8557290813820195, 4.856911591765037, 4.8557290813820195, 4.8557290813820195, 4.858094102148056])
    
    time_eq2 = np.array([0.395939572892549, 1.731947822943837, 3.066704400027353, 4.400423004405889, 5.733179957601867, 7.065219488487051, 8.396633182888348, 9.727466833719214, 11.057918876937954, 12.38792825532663, 13.717617083321121, 15.047153268270762, 16.376460488653127, 17.70561506599064, 19.034723850414696, 20.363649463184938, 21.692544547346206, 23.021363309985052, 24.2895522552094, 25.61869156824243, 26.947540859490243, 28.276405415042543, 29.605254706290356, 30.934119261842657, 32.263044874612895, 33.5920010159921, 35.5855039174311, 36.914536380332734, 38.24362990045231, 39.57272342057188, 42.2310478395514, 44.22488655568907, 46.21884738626261, 47.54821566386292, 48.8775992057677, 50.20692169045456, 52.20112674989986, 53.530571349022594, 55.52485272999033, 56.85432785772202, 58.84873135312563, 60.843073791311305, 62.83743149380147, 64.16707452888251, 66.16160013872201, 68.15600363412561, 69.48576878364253, 70.81548814024599, 71.87313653370846])
    response_eq2 = np.array([1.1242542878448951, 2.0680462341536163, 2.8207494407158826, 3.414988814317674, 3.8624161073825505, 4.20031692766592, 4.442673378076062, 4.596476510067113, 4.6920208799403404, 4.719985085756896, 4.69901193139448, 4.654735272184936, 4.575503355704697, 4.47296793437733, 4.363441461595823, 4.225950782997764, 4.083799403430273, 3.9299962714392223, 3.7482289336316192, 3.6433631618195363, 3.4942207307979096, 3.347408650261002, 3.1982662192393754, 3.0514541387024607, 2.913963460104398, 2.7811334824757665, 2.59237509321402, 2.471196868008949, 2.35934004474273, 2.247483221476511, 2.0447427293064884, 1.9072520507084256, 1.7884041759880702, 1.7184936614466828, 1.6509134973900075, 1.5740119313944838, 1.4924496644295324, 1.434190902311709, 1.3642803877703216, 1.3106823266219259, 1.2594146159582422, 1.1988255033557067, 1.1405667412378833, 1.1126025354213276, 1.0799776286353477, 1.0287099179716641, 1.0193885160328158, 1.0030760626398205, 0.9774422073079805])
    
    time_eq3 = np.array([0.13097449919546933, 0.6952998945780213, 1.075652980895228, 1.2022948132988212, 1.5187212321801258, 1.8140532551443544, 2.2141383211081584, 2.6558630203478613, 3.957841666658311, 4.859350874123724, 5.780995184013612, 7.162364093922856, 8.54290983763791, 9.923030278819295, 11.302949501597658, 12.682818419775264, 14.062696484243917, 16.362622830814832, 17.742738698850697, 19.122941456651503, 20.963367287333703, 23.724321580398108, 25.104922201859438, 26.945828212821585, 28.78683025983972, 29.707358722221926, 30.62793748920489, 31.548502536751283, 32.92940498581715, 34.31033944690168, 35.691301346859355, 37.07228611254464, 37.992951769292546, 38.91363114547703, 39.373991412724116, 40.75502648301016, 41.675733298067776, 42.59644011312539, 43.51715607447405, 44.89824602250638, 45.81896198385504, 46.739696237785786, 48.58115102621073, 49.962309571425905, 50.88303467906561, 52.26414749282555, 52.72453519894579, 54.105661732142295, 55.486811131066425, 56.407568250724786, 57.328311650946596, 58.24908706318706, 60.09063788766799, 60.44639202421544])
    response_eq3 = np.array([0.06301723609153242, 2.676262993772049, 4.204516764743712, 4.654642332309688, 5.705840806110585, 6.6872500304624545, 7.718527882363915, 8.660934272929522, 10.544325110679361, 11.236538977935155, 11.599476001717848, 11.687298688347028, 11.432684238192566, 11.00114393403322, 10.485896774215652, 9.949722900483527, 9.417353883826776, 8.583974331299986, 8.150531598602955, 7.7532350081219725, 7.288188791095353, 6.721887034655822, 6.4901017269536005, 6.224810506384108, 5.999470285106041, 5.898214745693126, 5.817885920194771, 5.731849809083352, 5.625624784868469, 5.5327167604173955, 5.451223307192441, 5.379241996655919, 5.335059313373616, 5.296583915704371, 5.285907145289338, 5.2348525486673765, 5.207791722224252, 5.18073089578113, 5.157474926413379, 5.129249472243659, 5.10599350287591, 5.090347247658908, 5.0533474516118435, 5.053658425507427, 5.034207313215051, 5.0154940017337655, 5.016231802544855, 5.003225776676629, 4.9997318934968416, 4.993597780968271, 4.981756382826642, 4.983231984448823, 4.986183187693182, 4.986753306501752])

    time_eq4 = np.array([0.021933583647965604, 0.03514739398022623, 0.06242016190457736, 0.10455678957323786, 0.12484032380915469, 0.1740064302738605, 0.22349449759945386, 0.2654298997300597, 0.3298623170151842, 0.4228955907758213, 0.5524848372830676, 0.6827582506197001, 0.907996702882295, 1.1850171936043068, 1.374786291025788, 1.5208894466893925, 1.6962373805502846, 1.9239154440352029, 2.2369640009996683, 2.5119334196864163, 2.7615633261736785, 3.0577576916598725, 3.269331749862247, 3.506245320264107, 3.7516209213943488, 3.9970663925030814, 4.187415503906012, 4.424352364300702, 4.768256447787936, 5.110420352696199, 5.399943656849345, 5.628004632659055, 6.014180587250648, 6.3561996097715125, 6.69841180880891, 7.075669447553917, 7.365241045836197, 7.698631517285274, 8.005749982485925, 8.36546075431256, 8.768893811047729, 9.067142254530928])
    response_eq4 = np.array([-0.014726728266523104, 0.24213245410194517, 0.6017210000462079, 0.9172486059541898, 1.2181329549299513, 1.5850252426258296, 1.8932014088768785, 2.2454265906878788, 2.49486278654144, 2.861683527379343, 3.2284446458356006, 3.4704340062215957, 3.7269473787765195, 3.8732845522880837, 3.9316906373483547, 3.953470693363582, 3.975203051473493, 3.978161646544941, 3.9861458011241364, 3.994192062350879, 3.9937849187725103, 3.989054462450956, 3.9844620209017303, 3.988322984157616, 3.9921701459362686, 3.9832750938597066, 3.9914593698587773, 3.991072928496258, 3.996614641170149, 3.9960565756779443, 3.995584366415309, 4.004019740970564, 3.97696787397021, 4.002832063128178, 3.9934665794192483, 3.992851276440663, 3.983571648961304, 3.9830278928406937, 3.973719646618144, 3.973132962382749, 3.998896985939552, 4.007217885522047])

    time_eq5 = np.array([-0.06024096385542166, 0.4819277108433736, 1.1746987951807224, 1.716867469879518, 2.4096385542168677, 3.192771084337349, 4.066265060240963, 4.879518072289156, 5.783132530120481, 6.626506024096386, 7.590361445783133, 8.493975903614457, 9.33734939759036, 10.72289156626506, 11.807228915662648, 12.740963855421686, 13.855421686746986, 15.03012048192771, 16.08433734939759, 16.837349397590362, 17.771084337349397, 18.795180722891565, 19.96987951807229, 20.813253012048193, 21.566265060240966, 22.228915662650603, 23.132530120481928, 24.30722891566265, 25.481927710843372, 26.506024096385545, 27.5, 28.46385542168675, 29.397590361445783, 30.180722891566266, 31.05421686746988, 31.927710843373497, 32.89156626506024, 33.674698795180724, 34.57831325301205, 35.5421686746988, 36.44578313253012])
    response_eq5 = np.array([5.0599317715687375, 4.790006905294402, 4.460072969379657, 4.2799684623867575, 3.979974646232491, 3.6200232924855964, 3.4096952394694267, 3.1395384789800778, 2.8393643006585805, 2.6290620136662994, 2.3887165428179795, 2.1783627237779193, 1.9680604367856382, 1.757294361363332, 1.5767260659404094, 1.3962866006369374, 1.2755727787110818, 1.1847475444979239, 1.0042050150988917, 0.9736207447411562, 0.8530615189586399, 0.7623651148649362, 0.7314201201727375, 0.6109381924618926, 0.6102940418646359, 0.5797870695785718, 0.5191338493409052, 0.4283086151277491, 0.4273037401960291, 0.4264276953837598, 0.30581693755346784, 0.275052305028499, 0.33413379780885855, 0.24364352190627514, 0.24289630721345823, 0.21220897276016082, 0.24132457975615118, 0.15083430385356777, 0.23988168241829833, 0.11929669061189507, 0.17840394941614512])
        
    # Test for Equation 1
    def test_fit_data_equation_1(self):
        p0 = [0, 5, 1]
        expected_params = np.array([0, 5, 1])  
        data = pd.DataFrame({'time': self.time_eq1, 'RU 1nM': self.response_eq1})
        
        kinetics = KineticAnalysis(data['time'], data['RU 1nM'])
        params = kinetics.curve_fit(kinetics.baseline_steadystate_response, p0)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

    # Test for Equation 2
    def test_fit_data_equation_2(self):
        p0 = [1, 1, 0.1, 0.09]
        expected_params = np.array([1, 1, 0.1, 0.09])  
        data = pd.DataFrame({'time': self.time_eq2, 'RU 1nM': self.response_eq2})
        
        kinetics = KineticAnalysis(data['time'], data['RU 1nM'])
        params = kinetics.curve_fit(kinetics.response_to_zero, p0)
        np.testing.assert_allclose(params, expected_params, atol=1)

    # Test for Equation 3
    def test_fit_data_equation_3(self):
        p0 = [0, 5, 5, 0.1, 0.2]
        expected_params = np.array([0, 5, 5, 0.1, 0.2])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq3, 'RU 1nM': self.response_eq3})
        
        kinetics = KineticAnalysis(data['time'], data['RU 1nM'])
        params = kinetics.curve_fit(kinetics.response_to_steady_state, p0)
        np.testing.assert_allclose(params, expected_params, atol=8e-1)

    # Test for Equation 4
    def test_fit_data_equation_4(self):
        p0 = [5, 2, 1, 2]
        expected_params = np.array([5, 2, 1, 2])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq4, 'RU 1nM': self.response_eq4})
        
        kinetics = KineticAnalysis(data['time'], data['RU 1nM'])
        params = kinetics.curve_fit(kinetics.typical_association, p0)
        np.testing.assert_allclose(params, expected_params, atol=3)
    
    # Test for Equation 5
    def test_fit_data_equation_5(self):
        p0 = [5, 0.1]
        expected_params = np.array([5, 0.1])  # Replace with actual expected parameters
        data = pd.DataFrame({'time': self.time_eq5, 'RU 1nM': self.response_eq5})
        
        kinetics = KineticAnalysis(data['time'], data['RU 1nM'])
        params = kinetics.curve_fit(kinetics.typical_dissociation, p0)
        np.testing.assert_allclose(params, expected_params, atol=6e-1)

if __name__ == '__main__':
    unittest.main()