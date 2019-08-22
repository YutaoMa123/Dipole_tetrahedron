import numpy as np
from scipy.special import gamma
import pickle
import json
from Update_functions import *
from compute_area import compute_area

def load_json(file):
	f = open(file,'r')
	parsed_json = json.load(f)
	print (json.dumps(parsed_json,indent=3, separators=(',',':')))
	f.close()
	return parsed_json

Ndim = 2
with open('iteration_data.pickle','rb') as f:
	iteration_data = pickle.load(f)
with open('optimzation.pickle','rb') as f:
	opt = pickle.load(f)
A = iteration_data[-1]['A']
kappa = iteration_data[-1]['kappa']
for i in range(len(A)):
	load_dir = "Run%d"%(i)
	json_file = "%s/run%d.json"%(load_dir,i)
	params = load_json(json_file)
	dipole_params = params['dipole']
	if ((not np.isclose(A[i],dipole_params['eps'])) or (not np.isclose(kappa,dipole_params['kappa']))):
		print ("Error loading file!")
Ngen = len(A)
mu = int(N_gen/2)
min_sigma = -5
c_C = 4/(Ndim+4)
a_cov = np.sqrt(1/mu)
c_cov = a_cov*(2/(Ndim+np.sqrt(2))**2) + (1-a_cov)*min(1,(2*mu-1)/(mu+(n+2)**2))
c_sigma = 4/(Ndim+4)
d_sigma = 1/(c_sigma) + 1
chi_n = np.sqrt(2)*gamma((Ndim+1)/2)/gamma(Ndim/2)

sigma = opt[-1]['sigma']
BD = opt[-1]['BD']
B = opt[-1]['B']
C = opt[-1]['C']
pc = opt[-1]['pc']
ps = opt[-1]['ps']
Z = opt[-1]['Z']
z_offspring = opt[-1]['z_offspring']
num_evals = 10*len(opt)

x_offspring = np.zeros((Ndim,Ngen))
x_offspring[0,:] = A
x_offspring[1,:] = kappa
sort_idx = np.argsort(fitness)
x_mean = np.mean(x_offspring[:,sort_idx[:mu]],axis=1)
z_mean = np.mean(z_offspring[:,sort_idx[:mu]],axis=1)

pc = UpdatePathCov(pc,BD,z_mean,mu,c_C)
Z = UpdateZ(BD,mu,z_offspring[:,sort_idx[:mu]])
ps = UpdatePathSigma(c_sigma,ps,mu,B,z_mean)
C = UpdatePathCov(pc,BD,z_mean,mu,c_C)
sigma = UpdateSigma(sigma,d_sigma,ps,chi_n)

if (not np.allclose(C,C.T)):
	print ("Covariance matrix is not symmetric up to numeric precision, enforcing it.....")
	C = np.triu(C) + np.triu(C,1).T
evals,B = np.linalg.eigh(C)
D = np.diag(np.sqrt(evals))
BD = np.dot(B,D)

z_offspring = np.random.multivariate_normal(np.zeros(Ndim),np.eye(Ndim),Ngen).T
x_offspring = x_mean + sigma*BD@z_offspring

temp_iter_dict = dict()
temp_iter_dict['A'] = x_offspring[0,:]
temp_iter_dict['kappa'] = x_offspring[1,:]
iteration_data.append(temp_iter_dict)
with open('iteration_data.pickle','wb') as f:
	pickle.dump(iteration_data,f)

temp_opt_dict = dict()
temp_opt_dict['sigma'] = sigma
temp_opt_dict['BD'] = BD
temp_opt_dict['B'] = B
temp_opt_dict['C'] = C
temp_opt_dict['D'] = D
temp_opt_dict['pc'] = pc
temp_opt_dict['ps'] = ps
temp_opt_dict['Z'] = Z
temp_opt_dict['z_offspring'] = z_offspring
temp_opt_dict['xmean'] = x_mean
temp_opt_dict['zmean'] = z_mean
opt.append(temp_opt_dict)
with open('optimzation.pickle','wb') as f:
	pickle.dump(opt,f)












