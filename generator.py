import pandas as pd  
import newthing

def read_from_column(column_name, covNum, cov_file):
    column = pd.read_csv(cov_file)[column_name].values
    currentValue = 0
    if covNum == 3:
        currentValue = column[len(column)-1] 
    elif covNum == 2:
        currentValue = column[4] 
    elif covNum == 1:
        currentValue = column[1] 
    elif covNum == 0:
        currentValue = column[0] 
    return currentValue

def grabOmega(covNum, cov_file):
	omega = read_from_column("\omega", covNum, cov_file)
	#print(omega)
	return omega

def grabHazardparams(covNum, cov_file, hazardName):
	param_names = {
		"GM": ["b"],
		"NB2": ["b"],
		"DW2": ["b"],
		"DW3": ["b","c"],
		"S":  ["p","pi"],
		"TL": ["c","d"],
		"IFRSB": ["c"],
		"IFRGSB": ["c", "alpha"]
	}
   
	correcthazard = param_names[hazardName]
	paramlist = []
	for parameter in correcthazard: 
		paramlist.append(read_from_column(parameter, covNum, cov_file))
	#print(paramlist)
	return paramlist

def grabBetas(covNum, cov_file): 
	betalist = []
	for i in range(covNum): 
		columname = "\\beta_" + str(i + 1) 
		betalist.append(read_from_column(columname, covNum, cov_file))
	#print(betalist)
	return betalist
    


def generate_all_files(base_directory):
	dataset_names = ["DS1", "DS2"]
	hazard_names = ["IFRGSB", "GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB"]
	
	for dataset in dataset_names:
		for hazard in hazard_names:
			for i in range(11):
				hazard_dir = base_directory + dataset + "/" + hazard + "/"
				output_file = hazard_dir + hazard + "_" + str(i) + "cov_sim.csv"

				mle_file = hazard_dir + "MLEs.csv"
				valid_dataset = True			
				# read omega from MLE file
				omega = grabOmega(min(3, i), mle_file)
				if omega > 200 or pd.isna(omega):
					valid_dataset = False				
				# read hazard parameters from MLE file
				params = grabHazardparams(min(3, i), mle_file,hazard)
				for j in range(len(params)):
					if pd.isna(params[j]):
						valid_dataset = False
					if params[j] < 0:
						params[j] = 0.01
				# read betas from MLE file
				betas = grabBetas(min(3, i), mle_file)
				for j in range(len(betas)):
					if pd.isna(betas[j]):
						valid_dataset = False
					if betas[j] < 0:
						betas[j] = 0.01
				if (not valid_dataset):
					continue
				# generate extra betas
				if j > 3:
					for k in range (j - 3):
						betas.append(0.01)
				cov_file = base_directory + dataset + "/" + dataset + ".csv"
				print("printing to", output_file)
				print(i)
				newthing.Generate_Covariates(output_file, hazard, cov_file, i, betas, omega, params)
				
				
generate_all_files("")
