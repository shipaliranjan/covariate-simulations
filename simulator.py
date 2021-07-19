import sys
from math import exp
import random
import math
import pandas as pd
from csv import writer

# 8 Different Hazard Functions
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
		b, c = params
		return 1 - exp(-c * i**b)

	elif model == "S":
		p, pi = params
		return p * (1 - pi**i)

	elif model == "TL":
		c, d = params
		return (1 - exp(-1/d))/(1 + exp(- (i - c)/d))

	elif model == "IFRSB":
		c = params[0]
		return 1 - c / i

	elif model == "IFRGSB":
		c, alpha = params
		return 1 - c / ((i - 1) * alpha + 1)

	raise Exception("model not implemented")

def poisson_variate(lam): # algorithm to find pseudorandom variate of the Poisson distribution
	x = 0
	p = exp(-lam)
	s = p
	u = random.uniform(0, 1)
	
	while u > s:
		x += 1
		p *= lam / x
		s += p
	return x

def g(x, n, betas, interval): # Equation 15
	g = 0
	for i in range(0, n):
		g += betas[i] * x[i][interval]
	g = exp(g)
	return g

def p(model, params, interval, x, n, beta): # Equation 19
	pixi = 1 - pow(1 - h0(model, params, interval), g(x, n, beta, interval))
	for k in range(0, interval):
		pixi *= pow(1 - h0(model, params, k), g(x, n, beta, k))
	return pixi
	
def generate_FC(model_name, x, covNum, num_intervals, beta, omega, hazard_params):
	n = len(beta)
	failures = []
	cumulative = 0
	for j in range(0, num_intervals):	
		prob = p(model_name, hazard_params, j, x, n, beta)
		failures.append(poisson_variate(omega * prob))
		cumulative += prob
	print(failures)
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
	FC = generate_FC(model_name, cov_array, num_cov, num_intervals, betas, omega, hazard_params)
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
