import sys
from math import exp
import random
import math
import pandas as pd
from csv import writer

# 8 Different Hazard Functions
def hazard_functions(model_name, i, args):
	if model_name == "GM":
		return GM(args)
	elif model_name == "NB2":
		return NB2(i, args)
	elif model_name == "DW2":
		return DW2(i, args)
	elif model_name == "DW3":
		return DW3(i, args)
	elif model_name == "S":
		return S(i, args)
	elif model_name == "TL": 
		return TL(i, args)
	elif model_name == "IFRSB": 
		return IFRSB(i, args)
	elif model_name == "IFRGSB": 
		return IFRGSB(i, args)
	else:
		return 0

def GM(args):
	# args -> (b)
	f = args[0]
	return f

def NB2(i, args):
	# args -> (b)
	f = (i * args[0]**2)/(1 + args[0] * (i - 1))
	return f

def DW2(i, args):
	# args -> (b)
	f = 1 - args[0]**(i**2 - (i - 1)**2)
	return f

def DW3(i, args):
	# args -> (c, b)
	f = 1 - math.exp(-args[0] * i**args[1])
	return f

def S(i, args):
	# args -> (p, pi)
	f = args[0] * (1 - args[1]**i)
	return f

def TL(i, args):
	# args -> (c, d)
	f = (1 - math.exp(-1/args[1]))/(1 + math.exp(- (i - args[0])/args[1]))
	return f

def IFRSB(i, args):
	# args -> (c)
	f = 1 - args[0] / i
	return f

def IFRGSB(i, args):
	# args -> (c, alpha)
	f = 1 - args[0] / ((i - 1) * args[1] + 1)
	return f

def poisson_variate(lam): # algorithm to find pseudorandom variate of the Poisson distribution
	x = 0
	p = exp(-lam)
	s = p
	u = random.uniform(0, 1)
	
	while u > s:
		x += 1
		p *= lam / x
		s += p
	print(x)
	return x

def g(x, n, betas, interval): # Equation 15
	g = 0
	for i in range(0, n):
		g += betas[i] * x[i][interval]
	g = exp(g)
	return g

def p(interval, x, n, beta, h): # Equation 19
	pixi = 1 - pow(1 - h, g(x, n, beta, interval))
	for k in range(0, interval):
		pixi *= pow(1 - h, g(x, n, beta, k))
	return pixi
	
def Generate_IMPL(model_name, x, covNum, num_intervals, beta, omega, hazard_params):
	n = len(beta)
	cumulative = 0
	failures = []
	for j in range(0, num_intervals):	
		h = hazard_functions(model_name, j+1, hazard_params)
		prob = p(j, x, n, beta, h)
		failures.append(poisson_variate(omega * prob))
	return failures

##########################
##      Main code       ##
##########################

def Generate_Covariates(output_filename, model_name, cov_file, num_cov, betas, omega, hazard_params):
	cov_CSV = pd.read_csv(cov_file)
	cov_dataset = cov_CSV.values
	num_intervals = cov_dataset.shape[0]
		
	cov_array = []
	for i in range(3):
		newlist = []
		for j in range(num_intervals):
			newlist.append(float(cov_dataset[j, i + 2]))
		cov_array.append(newlist) 
	for i in range (num_cov - 3): 
		new_cov = []
		for j in range (num_intervals):
			u = random.uniform(0, 1)
			new_cov.append(u * cov_array[i % 3][j])
		cov_array.append(new_cov)
	
	print(num_intervals)
	FC = Generate_IMPL(model_name, cov_array, num_cov, num_intervals, betas, omega, hazard_params)
	with open(output_filename, 'w') as myfile:
		wr = writer(myfile)
		wr.writerow(['T', 'FC'] + [f'x{i+1}' for i in range(len(cov_array))])
		wr.writerows(zip(
			range(1, num_intervals + 1),
			FC,
			*cov_array))
	myfile.close()
	

#commandline args - output file, model name, covariate datafile, num covariates, betas, omega, hazard params

beta_vec = [0.127077673444274, 0.0306315435752751, 0.102135735499068]
#beta_vec = []
#Generate_Covariates("DS1/GM/GM_3cov_sim.csv", "GM", "DS1/DS1.csv", 3,  beta_vec, 55, [0.026198])
