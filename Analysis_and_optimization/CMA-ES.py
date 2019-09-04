import numpy as np
from scipy.special import gamma
import pickle
import json
from Update_functions import *

def load_json(file):
	f = open(file,'r')
	parsed_json = json.load(f)
	#print (json.dumps(parsed_json,indent=3, separators=(',',':')))
	f.close()
	return parsed_json

Ndim = 2
with open('../../iteration_data.pickle','rb') as f:
	iteration_data = pickle.load(f)
with open('../../optimization.pickle','rb') as f:
	opt = pickle.load(f)
A = iteration_data[-1]['A']
kappa = iteration_data[-1]['kappa']
fitness = np.zeros(len(A))
for i in range(len(A)):
	sum = 0
	num_partners = 0
	for j in range(1,4):
		load_dir = "Run_%d_%d"%(i+1,j)
		json_file = "../%s/%s.json"%(load_dir,load_dir)	
		params = load_json(json_file)
		dipole_params = params['dipole']
		assert ((np.isclose(A[i],dipole_params['eps'])) and (np.isclose(kappa[i],dipole_params['kappa'])))
		dihedral_data = np.loadtxt('../%s/dihedral_angle.txt'%(load_dir))
		if (np.all(np.isnan(dihedral_data))):
			fitness[i] = np.inf
			break;
		else:
			sum += 60 - np.mean(dihedral_data)#[int(len(dihedral_data)/2),:])
			num_partners += 1
	assert num_partners is 3
	fitness[i] = sum/num_partners

iteration_data[-1]['fitness'] = fitness

Ngen = len(A)
mu = int(Ngen/2)
min_sigma = -5
c_C = 4/(Ndim+4)
a_cov = np.sqrt(1/mu)
c_cov = a_cov*(2/(Ndim+np.sqrt(2))**2) + (1-a_cov)*min(1,(2*mu-1)/(mu+(Ndim+2)**2))
c_sigma = 4/(Ndim+4)
d_sigma = 1/(c_sigma) + 1
chi_n = np.sqrt(2)*gamma((Ndim+1)/2)/gamma(Ndim/2)

sigma = opt[-1]['sigma']
BD = opt[-1]['BD']
B = opt[-1]['B']
C = np.asarray(opt[-1]['C'])
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
C = UpdateCov(c_cov,C,a_cov,pc,Z)
sigma = UpdateSigma(sigma,d_sigma,ps,chi_n)

if (not np.allclose(C,C.T)):
	print ("Covariance matrix is not symmetric up to numeric precision, enforcing it.....")
	C = np.triu(C) + np.triu(C,1).T
evals,B = np.linalg.eigh(C)
sort_idx = np.argsort(-evals)
D = np.diag(np.sqrt(evals[sort_idx]))
B = B[:,sort_idx]
BD = np.dot(B,D)

z_offspring = np.random.multivariate_normal(np.zeros(Ndim),np.eye(Ndim),Ngen).T
x_offspring = x_mean[:,np.newaxis] + sigma*BD@z_offspring
while (np.any(x_offspring[1,:] < 0.042)):
    print ("re-sampling....")
    z_offspring = np.random.multivariate_normal(np.zeros(Ndim),np.eye(Ndim),Ngen).T
    x_offspring = x_mean[:,np.newaxis] + sigma*BD@z_offspring

temp_iter_dict = dict()
temp_iter_dict['A'] = x_offspring[0,:]
temp_iter_dict['kappa'] = x_offspring[1,:]
iteration_data.append(temp_iter_dict)
with open('../iteration_data.pickle','wb') as f:
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
with open('../optimization.pickle','wb') as f:
	pickle.dump(opt,f)












