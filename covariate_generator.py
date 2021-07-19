import scipy.stats
import numpy as np
from csv import writer
import random
import simulator


def generate_hazardparams(hazard_name):
	param_ranges = {
		"GM":  [random.uniform(0.0147, 0.059)],
		"NB2": [random.uniform(0.079, 0.123)],
		"DW2": [random.uniform(0.995, 0.997)],
		"DW3": [random.uniform(0.005, 0.015), random.uniform(0.005, 0.015)],
		"S":   [random.uniform(0.018, 0.103), random.uniform(0.762, 0.911)],
		"TL":  [random.uniform(6.143, 19.418), random.uniform(6.062, 36.985)],
		"IFRSB": [random.uniform(0.993, 0.995)],
		"IFRGSB": [random.uniform(0.972, 0.999), random.uniform(0, 0.003)]
	}
	return param_ranges[hazard_name]


def generate_betas(num_covariates):
	betas = []
	for covariate in range(num_covariates):
		betas.append(random.uniform(0.01, 0.05))
	return betas;

def generate_omega(hazard_name):
	omega_ranges = {
		"GM":  random.uniform(54.217, 93.231),
		"NB2": random.uniform(71.845, 87.941),
		"DW2": random.uniform(93.753, 103.909),
		"DW3": random.uniform(67.015, 93.754),
		"S":   random.uniform(64.709, 89.747),
		"TL":  random.uniform(55.246, 96.743),
		"IFRSB": random.uniform(105.828, 137.003),
		"IFRGSB": random.uniform(64.483, 100.000),
	}
	return omega_ranges[hazard_name]

num_intervals = 20;
def generate_all_files(base_directory):
	dataset_names = ["DS1", "DS2"]
	hazard_names = ["GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB", "IFRGSB",]

	for hazard in hazard_names:
		print("using hazard ", hazard)
		for num_covariates in range(11):
			
			output_filename = base_directory + "/" + hazard + "/" \
				+ hazard + "_" + str(num_covariates) + "cov_dataset.csv"
			
			cov_dataset = []
			for covariate in range(num_covariates):
				cov_array = []
				for interval in range(num_intervals):
					cov_array.append(scipy.stats.expon.rvs(random.randint(2, 7)))
				cov_dataset.append(cov_array)
									
									
			FC = simulator.generate_FC(hazard, cov_dataset, num_covariates, num_intervals, \
				generate_betas(num_covariates), generate_omega(hazard), generate_hazardparams(hazard))
												
			with open(output_filename, 'w') as myfile:
				wr = writer(myfile)
				wr.writerow(['T', 'FC'] + [f'x{i+1}' for i in range(num_covariates)])
				wr.writerows(zip(
					range(1, num_intervals + 1),
					FC,
					*cov_dataset))	
					
	
			
			
			
for i in range(5): 
	base_dir = "cov_sims/sim" + str(i + 1)
	generate_all_files(base_dir)
