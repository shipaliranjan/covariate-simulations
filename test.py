import os
import ctypes
import numpy as np
import math
import pandas as pd
import sys
import csv
from math import *
from ctypes import c_double, c_int, POINTER, c_char_p
from scipy.special import gamma, factorial as spfactorial
from abc import ABC, abstractmethod, abstractproperty
import logging as log
import time   # for testing
import numpy as np
import scipy.optimize
from scipy.special import factorial as npfactorial
import symengine
import math
import pandas as pd
import csv
import os


def hazard_symbolic(i, args):
    if model == "GM":
        f = args[0]
        return f
    elif model == "DW3":
        f = 1 - symengine.exp(-args[0] * i**args[1])
        return f
    elif model == "DW2":
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f
    elif model == "IFRGSB":
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
        return f
    elif model == "IFRSB":
        f = 1 - args[0] / i
        return f
    elif model == "NB2":
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f
    elif model == "S":
        f = args[0] * (1 - args[1]**i)
        return f
    elif model == "TL":
        try:
            f = (1 - symengine.exp(-1/args[1])) / (1 + symengine.exp(- (i - args[0])/args[1]))
        except OverflowError:
            f = float('inf')
        return f


def hazard_numerical(i, args):
    if model == "GM":
        f = args[0]
        return f
    elif model == "DW3":
        f = 1 - math.exp(-args[0] * i**args[1])
        return f
    elif model == "DW2":
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f
    elif model == "IFRGSB":
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
        return f
    elif model == "IFRSB":
        f = 1 - args[0] / i
        return f
    elif model == "NB2":
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f
    elif model == "S":
        f = args[0] * (1 - args[1]**i)
        return f
    elif model == "TL":
        try:
            f = (1 - math.exp(-1/args[1])) / (1 + math.exp(- (i - args[0])/args[1]))
        except OverflowError:
            f = float('inf')
        return f


def initialEstimates():
    # bEstimate = [self.b0]
    parameterEstimates = list(parameters[model])
    betaEstimate = [0.01 for i in range(numCovariates)]
    return np.array(parameterEstimates + betaEstimate)


def LLF_sym(hazard, covariate_data):
    # x = b, b1, b2, b2 = symengine.symbols('b b1 b2 b3')

    x = symengine.symbols(f'x:{numSymbols}')
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(num_hazard_params, numSymbols):
            TempTerm1 = TempTerm1 * symengine.exp(covariate_data[j - num_hazard_params][i] * x[j])
        sum1 = 1 - ((1 - (hazard(i + 1, x[:num_hazard_params]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(num_hazard_params, numSymbols):
                TempTerm2 = TempTerm2 * symengine.exp(covariate_data[j - num_hazard_params][k] * x[j])
            sum2 = sum2 * ((1 - (hazard(i + 1, x[:num_hazard_params])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1 * sum2)

    firstTerm = -sum(kVec)  # Verified
    secondTerm = sum(kVec) * symengine.log(sum(kVec) / sum(prodlist))
    logTerm = []  # Verified
    for i in range(n):
        logTerm.append(kVec[i] * symengine.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = []  # Verified
    for i in range(n):
        factTerm.append(symengine.log(math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)

    f = firstTerm + secondTerm + thirdTerm - fourthTerm
    return f, x


def convertSym(x, bh, target):
    """Converts the symbolic function to a lambda function

    Args:

    Returns:

    """
    return symengine.lambdify(x, bh, backend='lambda')


def RLL_fit(x, covariate_data):
    # want everything to be array of length n
    cov_data = np.array(covariate_data)

    # gives array with dimensions numCovariates x n, just want n
    exponent_all = np.array([cov_data[i] * x[i + num_hazard_params] for i in range(numCovariates)])

    # sum over numCovariates axis to get 1 x n array
    exponent_array = np.exp(np.sum(exponent_all, axis=0))

    h = np.array([hazard_numerical(i + 1, x[:num_hazard_params]) for i in range(n)])

    one_minus_hazard = (1 - h)
    one_minus_h_i = np.power(one_minus_hazard, exponent_array)

    one_minus_h_k = np.zeros(n)
    for i in range(n):
        k_term = np.array([one_minus_hazard[i] for k in range(i)])

        # exponent array is just 1 for 0 covariate case, cannot index
        # have separate case for 0 covariates
        if numCovariates == 0:
            one_minus_h_k[i] = np.prod(np.array([one_minus_hazard[i]] * len(k_term)))
        else:
            exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
            one_minus_h_k[i] = np.prod(exp_term)

    failure_sum = np.sum(kVec)
    product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

    first_term = -failure_sum

    second_num = failure_sum
    second_denom = np.sum(product_array)

    second_term = failure_sum * np.log(second_num / second_denom)

    third_term = np.sum(np.log(product_array) * np.array(kVec))

    fourth_term = np.sum(np.log(npfactorial(kVec)))

    f = first_term + second_term + third_term - fourth_term
    return f


def RLL_minimize(x, covariate_data):
    return -RLL_fit(x, covariate_data)


def optimizeSolution(fd, B):
    # log.info("Solving for MLEs...")

    sol_object = scipy.optimize.root(fd, x0=B)
    solution = sol_object.x
    converged = sol_object.success
    # log.info("/t" + sol_object.message)

    return solution, converged


def modelFitting(betas, hazard, mle, covariate_data):
    omega = calcOmega(hazard, betas, covariate_data)
    # print(omega)
    # log.info("Calculated omega: %s", omega)

    mvf_array = MVF_all(mle, omega, hazard, covariate_data)
    # log.info("MVF values: %s", mvf_array)
    intensityList = intensityFit(mvf_array)
    # log.info("Intensity values: %s", intensityList)


def calcOmega(h, betas, covariate_data):
    # can likely use fewer loops
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(numCovariates):
            TempTerm1 = TempTerm1 * np.exp(covariate_data[j][i] * betas[j])
        sum1 = 1-((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(numCovariates):
                TempTerm2 = TempTerm2 * \
                    np.exp(covariate_data[j][k] * betas[j])
            sum2 = sum2*((1 - h[i])**(TempTerm2))
        prodlist.append(sum1*sum2)
    denominator = sum(prodlist)
    numerator = totalFailures

    return numerator / denominator


def MVF_all(mle, omega, hazard_array, covariate_data):
    mvf_array = np.array([MVF(mle, omega, hazard_array, dataPoints, covariate_data) for dataPoints in range(n)])
    return mvf_array


def MVF(x, omega, hazard_array, stop, cov_data):
    # gives array with dimensions numCovariates x n, just want n
    # switched x[i + 1] to x[i + numParameters] to account for
    # more than 1 model parameter
    # ***** can probably change to just betas
    exponent_all = np.array([cov_data[i][:stop + 1] * x[i + num_hazard_params] for i in range(numCovariates)])

    # sum over numCovariates axis to get 1 x n array
    exponent_array = np.exp(np.sum(exponent_all, axis=0))

    h = hazard_array[:stop + 1]

    one_minus_hazard = (1 - h)
    one_minus_h_i = np.power(one_minus_hazard, exponent_array)
    one_minus_h_k = np.zeros(stop + 1)
    for i in range(stop + 1):
        k_term = np.array([one_minus_hazard[i] for k in range(i)])
        if numCovariates == 0:
            one_minus_h_k[i] = np.prod(
                np.array([one_minus_hazard[i]] * len(k_term)))
        else:
            exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
            one_minus_h_k[i] = np.prod(exp_term)

    product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

    result = omega * np.sum(product_array)
    return result


def intensityFit(mvf_array):
    difference = [mvf_array[i+1]-mvf_array[i] for i in range(len(mvf_array) - 1)]
    return [mvf_array[0]] + difference


def runEstimation(covariateData):
    # need class of specific model being used, lambda function stored as class variable

    # ex. (max covariates = 3) for 3 covariates, zero_array should be length 0
    # for no covariates, zero_array should be length 3
    # numZeros = Model.maxCovariates - self.numCovariates
    # zero_array = np.zeros(numZeros)   # create empty array, size of num covariates

    # create new lambda function that calls lambda function for all covariates
    # for no covariates, concatenating array a with zero element array
    optimize_start = time.time()    # record time
    initial = initialEstimates()

    # log.info("Initial estimates: %s", initial)
    # pass hazard rate function
    f, x = LLF_sym(hazard_symbolic, covariateData)

    bh = np.array([symengine.diff(f, x[i]) for i in range(numSymbols)])

    fd = convertSym(x, bh, "numpy")

    solution_object = scipy.optimize.minimize(RLL_minimize, x0=initial, args=(covariateData,), method='Nelder-Mead')
    mle_array, converged = optimizeSolution(fd, solution_object.x)
    # print(mle_array)
    optimize_stop = time.time()
    runtime = optimize_stop - optimize_start
    # print(runtime)

    modelParameters = mle_array[:num_hazard_params]
    # print(modelParameters)
    betas = mle_array[num_hazard_params:]
    # log.info("model parameters =", modelParameters)
    # log.info("betas =", betas)

    hazard = np.array([hazard_numerical(i + 1, modelParameters) for i in range(n)])
    hazard_array = hazard    # for MVF prediction, don't want to calculate again
    modelFitting(betas, hazard, mle_array, covariateData)
    return runtime, converged, mle_array


# rllcv_libpath = os.path.abspath("rllcv.so")
# if not rllcv_libpath:
#     print("Unable to find rllcv lib")
#     sys.exit()

# try:
#     rllcv_lib = ctypes.CDLL(rllcv_libpath)
# except OSError:
#     print("Unable to load rllcvlib")
#     sys.exit()

# dataset = pd.read_csv("GM_10cov_dataset.csv").transpose().values
# dataset = dataset[1:]
# covariates = dataset[1:]
# kVec = dataset[0]

# num_hazard_params = 1

# max_val = 2 ^ 64


def h0(model, params, ivl):
    model = model
    params = params
    i = ivl + 1
    if model == "GM":
        b = params[0]
        return b 		# eqn 22
    elif model == "NB2":
        b = params[0]
        return (i*b**2)/(1 + b*(i - 1))
    elif model == "DW2":
        b = params[0]
        return 1 - b**(i**2 - (i - 1)**2)
    elif model == "DW3":
        c, b = params
        return 1 - exp(-c * i**b)
    elif model == "S":
        p, pi = params
        return p * (1 - pi**i)
    elif model == "TL":
        c, d = params
        return (1 - exp(-1 / d)) / (1 + exp(- (i - c) / d))
    elif model == "IFRSB":
        c = params[0]
        return 1 - c / i
    elif model == "IFRGSB":
        c, alpha = params
        return 1 - c / ((i - 1) * alpha + 1)

# --- COST/OBJECTIVE FUNCTION ------------------------------------------------------------+


def cpp_RLLCV(model_name, x):  # Caroline's in C++
    # the vector x contains [b, beta1, beta2, beta3, beta4] so 1: is just the betas

    rllcv_lib.rllcv.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_char_p]
    rllcv_lib.rllcv.restype = c_double

    cov_data = dataset.ctypes.data_as(POINTER(c_double))
    hazard_array = np.array(x[:num_hazard_params]).ctypes.data_as(POINTER(c_double))
    beta_array = np.array(x[num_hazard_params:len(x)]).ctypes.data_as(POINTER(c_double))
    num_intervals = c_int(len(dataset[0]))
    num_covariates = c_int(len(dataset) - 1)
    hazard_name = c_char_p(model_name.encode('utf-8'))

    retval = rllcv_lib.rllcv(cov_data, hazard_array, beta_array,num_covariates, num_hazard_params, num_intervals, hazard_name)
    if np.isnan(retval) or retval > max_val:
        return float('inf')
    return retval


def RLL(model_name, x):  # Jacob's from C-SFRAT
    n = len(kVec)
    hazard_params = x[0:num_hazard_params]
    # want everything to be array of length n
    cov_data = np.array(covariates)

    # gives array with dimensions numCovariates x n, just want n
    exponent_all = np.array([cov_data[i] * x[i + num_hazard_params] for i in range(len(covariates))])

    # sum over numCovariates axis to get 1 x n array
    exponent_array = np.exp(np.sum(exponent_all, axis=0))

    h = np.array([h0(model_name, hazard_params, i) for i in range(n)])

    one_minus_hazard = (1 - h)
    one_minus_h_i = np.power(one_minus_hazard, exponent_array)

    one_minus_h_k = np.zeros(n)
    for i in range(n):
        k_term = np.array([one_minus_hazard[i] for k in range(i)])

        # exponent array is just 1 for 0 covariate case, cannot index
        # have separate case for 0 covariates
        if len(covariates[0]) == 0:
            one_minus_h_k[i] = np.prod(np.array([one_minus_hazard[i]] * len(k_term)))
        else:
            exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
            one_minus_h_k[i] = np.prod(exp_term)

    failure_sum = np.sum(kVec)
    product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

    first_term = -failure_sum

    second_num = failure_sum
    second_denom = np.sum(product_array)

    second_term = failure_sum * np.log(second_num / second_denom)
    third_term = np.sum(np.log(product_array) * np.array(kVec))
    fourth_term = np.sum(np.log(spfactorial(kVec)))

    f = -(first_term + second_term + third_term - fourth_term)
    return f





def RLLCV(model_name, x):  # Caroline's optimized version of Josh's
    kvec_sum = np.sum(kVec)
    fourthTerm = np.sum(np.log(spfactorial(kVec)))
    n = len(kVec)
    # the vector x contains [b, beta1, beta2, beta3, beta4] so 1: is just the betas
    betas = np.array(x[num_hazard_params:len(x)])
    hazard_params = x[0:num_hazard_params]
    prodlist = np.zeros(n)
    # store g calculations in an array, for easy retrieval
    glookups = np.zeros(n)
    for i in range(n):
        one_minus_hazard = (1 - h0(model_name, hazard_params, i))
        try:
            glookups[i] = exp(np.dot(betas, covariates[:, i]))
        except OverflowError:
            return float('inf')
        # calculate the sum of all gxib from 0 to i, then raise (1 - b) to that sum
        exponent = np.sum(glookups[:i])
        sum1 = 1 - (one_minus_hazard ** glookups[i])
        prodlist[i] = (sum1 * (one_minus_hazard ** exponent))

    firstTerm = -kvec_sum  # Verified
    secondTerm = kvec_sum * np.log(kvec_sum/np.sum(prodlist))
    thirdTerm = np.dot(kVec, np.log(prodlist))  # Verified

    cv = -(firstTerm + secondTerm + thirdTerm - fourthTerm)

    if np.isnan(cv):
        return float('inf')

    return cv


# Converges at 32.06889534800335
gm_x = [0.13301841, 0.08674405, -0.09466961,  0.14564956,  0.15383779, -
        0.1973276, -0.09984516, 0.07759801, 0.03415457,  0.00398881, -0.08664824]
# Converges at 28.960325639630938
nb2_x = [0.03869884, 0.00621316, 0.16050914,  0.0876674, 0.03264906,
         0.02655284, 0.19412494, -0.00096592,  0.0125871,  -0.04533476, -0.01065368]
# Converges at  35.9969682299797
dw2_x = [0.99345516,  0.04936018,  0.10393346, 0.15088888, 0.15918737, -
         0.17274806, -0.01578893,  0.05814326, -0.11132775, -0.20782257, -0.05848546]
# Converges at 28.809394418014605
dw3_x = [0.01154487,  0.26614919,  0.06652046, 0.22027618, -0.0845511, 0.15518881, -
         0.06962841,  0.01304973,  0.00096311, 0.08422386, -0.02240908, -0.10236005]
# Converges at 34.648509014704025
s_x = [0.03877921, 0.8634113,   0.09541224, -0.14026449, 0.09403284,  0.0555435, -
       0.12904187,  0.09481806, -0.05346137, 0.15582485, 0.04568661, 0.03284889]
# Converges at 26.790966544217554
tl_x = [1.39621118, 2.58515399,  0.27219623, -0.12004428,  0.0671122, 0.0365206,
        0.11427245, -0.26159111, -0.07167988, -0.15784051,  0.00417037, 0.04847583]
# Converges at 6.724709568634125
ifrsb_x = [0.99364344, -0.11138931, -0.05560922, 0.01931634, -0.00790432,
           0.15864315, 0.04270897,  0.17905343,  0.03117651,  0.17713567, -0.10641685]
# Converges at 30.108284917849332
ifrgsb_x = [0.51435809,  0.09365767, -0.0183567,  0.08190612,  0.11857083,  0.00743039,
            0.15372869, 0.06054129, -0.21726666, -0.30750791,  0.08731458, -0.28696934]

# for i in range(1000000):
# 	 #cpp_RLLCV("GM", gm_x)
# 	 #RLLCV("GM", gm_x)
# 	 RLL("GM", gm_x)
# 	#print("C++ gets ", cpp_RLLCV("NB2", gm_x))
# 	#print("Jacob gets ", RLL("NB2", gm_x))
# 	#print("Python gets ", RLLCV("NB2", gm_x))

parameters = {"IFRGSB": [0.1, 0.1], "GM": [0.01], "NB2": [0.01], "DW2": [0.994], "DW3": [0.1, 0.5], "S": [0.1, 0.1], "TL": [0.1, 0.1], "IFRSB": [0.1]}
hazard_names = ["IFRGSB", "GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB"]
num_covariates = 10
num_simulated_sets = 30

for hazard_name in ["GM"]:
    for model in ["GM"]:
        for numCov in range(10, num_covariates+1):
            runtimes = []
            for run in range(1, num_simulated_sets+1):
                input_file = f"datasets/cov_sims30/sim{run}/{hazard_name}/{hazard_name}_{numCov}cov_dataset.csv"
                metricNames = csv.reader(open(input_file, newline=''))
                metricNames = next(metricNames)[2:]
                data = pd.read_csv(input_file)
                t = data["T"].values     # failure times
                kVec = data["FC"].values     # number of failures
                totalFailures = sum(kVec)
                n = len(kVec)
                covariates = np.array([data[name].values for name in metricNames])
                numCovariates = len(covariates)
                num_hazard_params = len(parameters[model])
                numSymbols = numCovariates + num_hazard_params
                runtime, converged, mle_array = runEstimation(covariates)

                if converged:
                    objective_start = time.time()
                    jacob_LL = RLL(model, mle_array)
                    objective_stop = time.time()
                    time1 = objective_stop - objective_start

                    objective_start = time.time()
                    caroline_LL = RLLCV(model, mle_array)
                    objective_stop = time.time()
                    time2 = objective_stop - objective_start

                    # objective_start = time.time()
                    # cpp_LL = cpp_RLLCV(model, mle_array)
                    # objective_stop = time.time()
                    # time3 = objective_stop - objective_start

                    times = [time1, time2]
                    lls = [jacob_LL, caroline_LL]

                    print(f"Data:{hazard_name} | Model:{model} | {numCov} Covariate(s) | Run {run}")
                    print(f"{lls}\n{times}\n\n")
