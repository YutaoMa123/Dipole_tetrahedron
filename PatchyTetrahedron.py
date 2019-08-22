import numpy as np
from pyquaternion import Quaternion
from scipy.linalg import expm, norm

def compute_quaternion(v1,v2):
	v1 = v1/np.linalg.norm(v1)
	v2 = v2/np.linalg.norm(v2)
	if (np.allclose(v1,v2)):
		return np.array([1,0,0,0])
	if (np.allclose(v1,-v2)):
		if (np.isclose(np.abs(v1[2]),1)):
			q = Quaternion(matrix = np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
			return q.elements
		else:
			fx,fy,fz = v1[0],v1[1],v1[2]
			R = 1./((1-fz)**2)*np.array([[-fx**2+fy**2,-2*fx*fy,0],[-2*fx*fy,fx**2-fy**2,0],[0,0,-1+fz**2]])
			q = Quaternion(matrix = R)
			return q.elements
	u = np.cross(v1,v2)
	w = np.linalg.norm(v1)*np.linalg.norm(v2) + np.dot(v1,v2)
	q = [w,u[0],u[1],u[2]]
	return q/np.linalg.norm(q)

class PatchyTetrahedron():
	def __init__(self):
		print('Creating Patchy Ring Tetrahedron')

	def rotation_matrix(self,axis,phi):
		return expm(np.cross(np.eye(3), axis/norm(axis)*phi))

	def azimuth_reference(self,axis):
		v = np.array([1,0,0])
		ref = np.cross((np.cross(axis,v)),axis)
		return ref/norm(ref)

	def gen_ring(self,R,rp,cur_vertex,a,theta):
		# Generate patches at given polar angle
		axis = cur_vertex/norm(cur_vertex)
		if (np.isclose(theta,0) or np.isclose(theta,2*np.pi)):
			return (cur_vertex + axis * R).reshape((1,3))
		if (np.isclose(theta,np.pi)):
			return (cur_vertex - axis * R).reshape((1,3))
		r = R*np.sin(theta)
		d = 2*rp
		d_phi = np.arccos(1-self.alpha**2*d**2/(2*r**2))
		n = int(np.ceil(2*np.pi/d_phi))
		vec = np.zeros((n,3))
		for i in range(n):
			M = self.rotation_matrix(axis,i*d_phi)
			q = np.dot(M,a)
			vec[i,:] = cur_vertex + q*r + axis*R*np.cos(theta)
		return vec



	def setup(self,types,sigma,mass,alpha,phi_max):
		# First type in types represents center of tetrahedron
		# Second type represents vertices of tetrahedron
		# Third type represents patches
		self.sigma = sigma
		self.types = types
		self.mass = mass
		self.alpha = alpha
		self.type_nums = [0 for i in range(len(sigma))]
		self.type_nums[0] = 1

		# Put vertices of tetrahedron in place
		self.locs = sigma[1]/np.sqrt(8./3) * np.array([[np.sqrt(8./9),0,-1./3],
			                                           [-np.sqrt(2./9),np.sqrt(2./3),-1./3],
			                                           [-np.sqrt(2./9),-np.sqrt(2./3),-1./3],
			                                           [0.,0.,1.]])
		self.orientations = np.zeros((4,4))
		for i in range(4):
			v1 = np.array([1,0,0])
			v2 = self.locs[i,:]
			q = compute_quaternion(v1,v2)
			self.orientations[i,:] = q
		self.labels = [types[1] for i in range(4)]
		self.type_nums[1] = 4

		# Now generate patches on vertices
		R = self.sigma[1]/2.
		rp = self.sigma[2]/2.
		d_theta = np.arccos(1-self.alpha**2*self.sigma[2]**2/(2*R**2))
		self.n_layers = int(np.ceil(phi_max/d_theta))
		for i in range(4):
			cur_vertex = self.locs[i,:]
			reference = self.azimuth_reference(cur_vertex/norm(cur_vertex))
			for j in range(self.n_layers):
				theta = j*d_theta
				vec = self.gen_ring(R,rp,cur_vertex,reference,theta)
				self.locs = np.vstack((self.locs,vec))
				self.orientations = np.vstack((self.orientations,np.ones((len(vec),4))*np.array([1,0,0,0])))
				self.type_nums[2] += vec.shape[0]
				self.labels = np.hstack((self.labels,[types[2] for k in range(len(vec))]))

	def get_pos(self):
		return tuple(map(list,self.locs))

	def get_orientation(self):
		return tuple(map(list,self.orientations))

	def get_type_nums(self):
		return self.type_nums
	
	def get_labels(self):
		return list(self.labels)










