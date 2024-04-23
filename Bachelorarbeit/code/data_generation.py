from FyeldGenerator import generate_field
import numpy as np
import scipy.special as sp
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor



#################################################################################################

################################# Random field generation #######################################


# defining the covariance functions in spectal form with length scale l
# cos(2*pi * x / l) covariance
def spectrum_cosine(k, l = 600): 
    return np.abs(
        -1j/2 *(np.exp(1j * (2*np.pi / l + k) * 2) / (2*np.pi / l + k) + np.exp(1j * (k - 2*np.pi / l) * 2) / (k - 2*np.pi / l)))**2

# power series covariance 
def spectrum_power_series(k, coeff_powerseries = 4):
    return np.power(k, -coeff_powerseries)

def spectrum_test(k, coeff_powerseries = 4):
    return k**(-4)


# defining the Matérn kernel and its power spectrum
def spectrum_matern(f,nu=2, n=2,sigma=1,rho=400):
    return sigma**2 * (2**(n) * np.pi**(n/2) * sp.gamma(nu + 0.5 * n) * (2*nu)**nu) / (sp.gamma(nu) * rho**(2 * nu)) * (2 * nu / rho**2 + 4 * np.pi**2 * f**2)**-(nu + n * 0.5)

def matern(x,nu=2,sigma=1,rho=1):
    return sigma**2 * 2**(1 - nu) / sp.gamma(nu) * (np.sqrt(2*nu) * x / rho)**nu * sp.kv(nu, np.sqrt(2*nu) * x / rho)

def sq_exponential(x,sigma=1,l=1):
    return sigma**2 * np.exp(- np.abs(x)**2 / ( 2 * l**2))

def exponential(x,sigma=1,l=1):
    return sigma**2 * np.exp(- np.abs(x) / l)

def spectrum_sq_exponential_test(x,sigma=1,l=1):
    return sigma**2 * np.sqrt(l)/(2*np.sqrt(np.pi)) * np.exp(- l * x**2 / 4)
    # return sigma**2 *2*np.pi* l**2 * np.exp(- 2 * np.pi**2 * l**2 * x**2)







# helper function that generates the function to be passed to generate_field()
def spectrum_generator(spectrum,spectrum_coeff):
    # spectrum_coeff = kwargs.get('spectrum_coeff')
    def helper(k):
        return spectrum(k,*spectrum_coeff) # Choose variance function here
    return helper

# draw samples from a normal distribution for generate_field()
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


def random_field(size,spectrum,spectrum_coeff):
    shape = (size, size)
    return generate_field(distrib, spectrum_generator(spectrum,spectrum_coeff), shape)



def random_field_anisotropic():
    '''
    I don't understand how this function works, but it does
    Source: https://dsp.stackexchange.com/questions/36902/calculate-1d-power-spectrum-from-2d-images
    '''
    shift = np.fft.fftshift
    nN, nE = 1024, 1024
    dE, dN = 112., 1132.  # Arbitrary values for sampling in dx and dy
    amplitude = 50.

    rfield = np.random.rand(nN, nE)
    spec = np.fft.fft2(rfield)

    regime = np.array([.15, .60, 1.])
    beta = np.array([5./3, 8./3, 2./3])
    beta += 1.  # Betas are defined for 1D PowerSpec, increasing dimension

    kE = np.fft.fftfreq(nE, dE)
    kN = np.fft.fftfreq(nN, dN)

    k = kN if kN.size < kE.size else kE
    k = k[k > 0]
    k_rad = np.sqrt(kN[:, np.newaxis]**2 + kE[np.newaxis, :]**2)

    k0 = 0
    k1 = regime[0] * k.max()
    k2 = regime[1] * k.max()

    r0 = np.logical_and(k_rad > k0, k_rad < k1)
    r1 = np.logical_and(k_rad >= k1, k_rad < k2)
    r2 = k_rad >= k2

    amp = np.empty_like(k_rad)
    amp[r0] = k_rad[r0] ** -beta[0]
    amp[r0] /= amp[r0].max()

    amp[r1] = k_rad[r1] ** -beta[1]
    amp[r1] /= amp[r1].max()/amp[r0].min()

    amp[r2] = k_rad[r2] ** -beta[2]
    amp[r2] /= amp[r2].max()/amp[r1].min()

    amp[k_rad == 0.] = amp.max()

    amp *= amplitude**2
    spec *= np.sqrt(amp)  # We come from powerspec!
    noise = np.abs(np.fft.ifft2(spec))

    return noise



#################################################################################################

############################## 1D Gaussian Process Regression ###################################



def gaussian_regression_1D(x, kernel, n_samples=4,eval_points=[np.array([])]):
    '''
    Takes a kernel, number of evaluation points, spacial sampling frequency, amount of functions to draw from process
    Returns y values for the mean of the process, 95% confidence interval and samples.
    '''
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=10)

    if eval_points[0].any(): #fit the model only if have data points
        gpr_model.fit(eval_points[0].reshape(-1, 1), eval_points[1].reshape(-1, 1))

    X = x.reshape(-1, 1)
    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)
    y_samples = np.transpose(y_samples)
    #y_conf should go here too: 95% confidence interval calculated from the standard deviation
    y_conf = 1.959964 * y_std
    return y_mean, y_conf, y_samples



#################################################################################################

############################## 2D Gaussian Process Regression ###################################



def gaussian_regression_2D(X, points, kernel, n_samples=0):
    '''
    Takes a kernel, number of evaluation points, spacial sampling frequency, amount of functions to draw from process
    Returns y values for the mean of the process, 95% confidence interval and samples. Samples are really really slow
    '''
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=10)
    gpr_model.fit(points[:2,:].transpose(), points[2,:])

    coords=np.stack(X , axis = -1).reshape(-1,2)
    y_mean, y_std = gpr_model.predict(coords, return_std=True)
    y_conf = 1.959964 * y_std
    return y_mean, y_conf



#################################################################################################

######################################## Function Data ##########################################


def gaussian_pdf(x,mean, std): 
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2)) 
    return y_out 



def sample_points(f,x,n):
    '''
    Samples n points randomly from f(x) uniformly distributed for an interval x
    '''
    rng = np.random.RandomState(4)
    x_eval = rng.uniform(np.min(x), np.max(x), n)
    y_eval=f(x_eval)
    #y_eval = np.sin((x_eval - 2.5) ** 2)
    return [x_eval, y_eval]



def sample_points_2D(n, field, size):
    '''
    Samples n coordinates randomly from a (size*size) grid, evaluetes given scalar field at those coordinates
    '''
    points = []
    for i in range(n):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        z = field[x,y]
        points.append((x,y,z))
    return np.array(points).transpose()

