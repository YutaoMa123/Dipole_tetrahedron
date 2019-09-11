import numpy as np
import scipy.spatial.distance
from hoomd import *
from hoomd import md
from hoomd import deprecated
import gsd.hoomd
from math import pi
import json,sys,os,glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from periodic_kdtree import PeriodicCKDTree

def load_json(file):
	f = open(file,'r')
	parsed_json = json.load(f)
	print (json.dumps(parsed_json,indent=3, separators=(',',':')))
	f.close()
	return parsed_json

def angle(v1,v2):
	theta = np.arctan2(np.linalg.det([v1,v2]),np.dot(v1,v2))
	return np.rad2deg(theta)

def min_image(v,box):
	L = box.Lx
	return v - L*np.rint(v/L)

def distance(pos1,pos2,box):
	disp = min_image(pos1-pos2,box)
	return np.linalg.norm(disp, axis=-1)

def bonded(snap,i,j,box):
	type_list = snap.particles.types
	index_B = type_list.index('B')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	pos_i = positions[i,:]
	pos_j = positions[j,:]
	loc_B_i = np.where((bodies == i) & (type_id == index_B))[0]
	pos_B_i = positions[loc_B_i,:]
	loc_B_j = np.where((bodies == j) & (type_id == index_B))[0]
	pos_B_j = positions[loc_B_j,:]

	bounds = np.array([box.Lx,box.Ly,box.Lz])
	T = PeriodicCKDTree(bounds, pos_B_j)

	for m in range(len(pos_B_i)):
		cur_pos = pos_B_i[m,:]
		nn_dist, idx = T.query(cur_pos, k=1)
		if (nn_dist < 2.5):
			return True
	return False

def select_A_indices(snap,i,j,box):
	type_list = snap.particles.types
	index_A = type_list.index('A')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	loc_A_i = np.where((bodies == i) & (type_id == index_A))[0]
	loc_A_j	= np.where((bodies == j) & (type_id == index_A))[0]
	pos_A_i = positions[loc_A_i,:]
	pos_A_j = positions[loc_A_j,:]

	min_dist = np.inf
	max_dist = 0
	for m in range(len(loc_A_i)):
		cur_distance = distance(pos_A_i[m,:],pos_A_j,box)
		min_cur_distance = np.amin(cur_distance)
		max_cur_distance = np.amax(cur_distance)
		if (min_cur_distance <= min_dist):
			min_dist = min_cur_distance
			center_idx_i = loc_A_i[m]
			center_idx_j = loc_A_j[np.argmin(cur_distance)]
		if (max_cur_distance >= max_dist):
			max_dist = max_cur_distance
			ref_idx_i = loc_A_i[m]
			ref_idx_j = loc_A_j[np.argmax(cur_distance)]
	return center_idx_i,center_idx_j

def bond_angle(snap,i,j,box):
	type_list = snap.particles.types
	index_B = type_list.index('B')
	index_A = type_list.index('A')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	pos_i = positions[i,:]
	pos_j = positions[j,:]

	disp_ij = min_image(pos_j-pos_i,box)
	loc_A_i = np.where( (bodies == i) & (type_id == index_A))[0]
	pos_A_i = positions[loc_A_i,:]
	disp = min_image(pos_A_i - pos_i,box)
	vectors = np.append(disp,disp_ij[np.newaxis,:],axis=0)
	similarity = scipy.spatial.distance.pdist(vectors, 'cosine')
	theta_1 = np.rad2deg(np.arccos(1 - similarity))
	min_theta_i = np.amin(theta_1)

	disp_ji = min_image(pos_i-pos_j,box)
	loc_A_j = np.where((bodies == j) & (type_id == index_A))[0]
	pos_A_j = positions[loc_A_j,:]
	disp = min_image(pos_A_j - pos_j,box)
	vectors = np.append(disp,disp_ji[np.newaxis,:],axis=0)
	similarity = scipy.spatial.distance.pdist(vectors, 'cosine')
	theta_2 = np.rad2deg(np.arccos(1-similarity))
	min_theta_j = np.amin(theta_2)

	return max(min_theta_i,min_theta_j)

def compute_dihedral(b1,b2,b3):
	n1 = np.cross(b1,b2)
	n2 = np.cross(b2,b3)
	n1 = n1/np.linalg.norm(n1)
	n2 = n2/np.linalg.norm(n2)
	m1 = np.cross(n1,b2/np.linalg.norm(b2))
	x = np.dot(n1,n2)
	y = np.dot(m1,n2)
	return abs(np.arctan2(y,x))

def dihedral_angle(snap,i,j,center_idx_i,center_idx_j,box):
	positions = snap.particles.position
	type_id = snap.particles.typeid
	type_list = snap.particles.types
	type_A = type_list.index('A')
	bodies = snap.particles.body
	loc_A_i = np.where( (bodies == i) & (type_id == type_A))[0]
	loc_A_j = np.where( (bodies == j) & (type_id == type_A))[0]
	b2 = min_image(positions[j,:]-positions[i,:],box)
	dihedral_list = []
	for m in range(len(loc_A_i)):
		if (loc_A_i[m] == center_idx_i):
			print ("rua")
			continue
		min_dihedral = np.inf
		b1 = min_image(positions[i,:]-positions[loc_A_i[m],:],box)
		for n in range(len(loc_A_j)):
			if (loc_A_j[n] == center_idx_j):
				print ("rua")
				continue
			b3 = min_image(positions[loc_A_j[n],:]-positions[j,:],box)
			cur_dihedral = np.rad2deg(compute_dihedral(b1,b2,b3))
			if (cur_dihedral <= min_dihedral):
				min_dihedral = cur_dihedral
		dihedral_list.append(min_dihedral)
	return np.mean(dihedral_list)


if __name__ == "__main__":
	folder = sys.argv[1]
	os.chdir(folder)
	json_file = glob.glob('*.json')[0]
	params = load_json(json_file)
	dipole_params = params['dipole']
	context.initialize('--mode=cpu')
	file = sys.argv[2]
	traj = gsd.hoomd.open(file,'rb')
	theta = np.zeros(len(traj))
	phi = np.zeros(len(traj))

	select_flag = False
	for f in range(len(traj)):
		print ("Processing frame %d" %(f))
		snap = data.gsd_snapshot(filename=file,frame = f)
		box = snap.box
		if (bonded(snap,0,1,box)):
			theta[f] = bond_angle(snap,0,1,box)
			if (not select_flag):
				center_idx_0,center_idx_1 = select_A_indices(snap,0,1,box)
				print ("Selected ",[center_idx_0,center_idx_1])
				phi[f] = dihedral_angle(snap,0,1,center_idx_0,center_idx_1,box)
				select_flag = True
			else:
				phi[f] = dihedral_angle(snap,0,1,center_idx_0,center_idx_1,box)
		else:
			print ("Not bonded in frame %d!" %(f))
			theta[f] = np.nan
			phi[f] = np.nan

	np.savetxt('bond_angle.txt',theta,fmt='%.5f')
	np.savetxt('dihedral_angle.txt',phi,fmt='%.5f')

	f = plt.figure()
	plt.plot(theta,'.-')
	plt.title('Bond Angle')
	plt.xlabel('Frame number')
	plt.ylabel('Degree')
	#plt.title('mu = %.2f, epsilon = %.2f' %(dipole_params['mu'],dipole_params['eps']))
	f.savefig("bond_angle.pdf", bbox_inches='tight')

	f = plt.figure()
	plt.plot(phi,'.-')
	plt.title('Dihedral Angle')
	plt.xlabel('Frame Number')
	plt.ylabel('Degree')
	#plt.title('mu = %.2f, epsilon = %.2f' %(dipole_params['mu'],dipole_params['eps']))
	f.savefig("dihedral_angle.pdf", bbox_inches='tight')

	f = plt.figure()
	plt.hist(phi,bins=100,range=(-180,180),density=True)
	plt.title('Dihedral Anlgle Frequency')
	plt.xlabel('Degree')
	plt.ylabel('PDF')
	#plt.title('mu = %.2f, epsilon = %.2f' %(dipole_params['mu'],dipole_params['eps']))
	f.savefig("dihedral_angle_freq.pdf", bbox_inches='tight')
	plt.show()



