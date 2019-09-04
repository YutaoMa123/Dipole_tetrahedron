import numpy as np

def UpdatePathCov(pc,BD,z_mean,mu,c_C):
    print(BD.shape, z_mean.shape)
    print(pc.shape)
    return (1-c_C)*pc + np.sqrt(c_C*(2-c_C))*np.sqrt(mu)*np.dot(BD,z_mean)

def UpdatePathSigma(c_sigma,p_sigma,mu,B,z_mean):
	return (1-c_sigma)*p_sigma + np.sqrt(c_sigma*(2-c_sigma)*mu)*np.dot(B,z_mean)

def UpdateZ(BD,mu,z_mu):
	return 1/mu*(BD@(z_mu@z_mu.T)@BD.T)

def UpdateCov(c_cov,C,a_cov,pc,Z):
	return (1-c_cov)*C + c_cov*(a_cov*np.outer(pc,pc) + (1-a_cov)*Z)

def UpdateSigma(sigma,d_sigma,ps,chi_n):
	return sigma*np.exp((np.linalg.norm(ps)-chi_n)/(d_sigma*chi_n))



