from hoomd import *
from hoomd import md
from hoomd import deprecated
import json
import numpy as  np
from math import pi
from PatchyTetrahedron import PatchyTetrahedron
import os

def MIXS(s1,s2):
	return (s1+s2)/2
def MIXE(eps1,eps2):
	return np.sqrt(eps1*eps2)
def load_json(file):
	f = open(file,'r')
	parsed_json = json.load(f)
	print (json.dumps(parsed_json,indent=3, separators=(',',':')))
	f.close()
	return parsed_json

context.initialize()
json_file = sys.argv[1]
params = load_json(json_file)
out_dir = params['output_dir']
if not os.path.exists(out_dir):
	os.makedir(out_dir)
L = params['L']
types = params['types']
phi_max = np.deg2rad(params['phi_max'])
sigma = params['sigma']
mass = params['mass']
alpha = params['alpha']

uc = lattice.unitcell(N=2,a1 = [L,0.,0.], a2 = [0,L,0.], a3 = [0.,0.,L], position = [[L/6,0,0],[0.,0.,0.]], type_name = ['C','C'])
sys = init.create_lattice(unitcell=uc,n=1)
sys.particles.types.add('A')
sys.particles.types.add('B')
snap = sys.take_snapshot()
print (snap.particles.types)

tetra = PatchyTetrahedron()
tetra.setup(types,sigma,mass,alpha,phi_max)
type_nums = tetra.get_type_nums()
print (type_nums)

rigid = md.constrain.rigid()
rigid.set_param(types[0],types=tetra.get_labels(),positions=tetra.get_pos(),orientations=tetra.get_orientation())
rigid.create_bodies()
for p in sys.particles:
	p.diameter = sigma[p.typeid]
	p.mass = mass[p.typeid]

# Set up pair potentials
nl = md.nlist.cell()
dipole_params = params['dipole']
dipole = md.pair.dipole(r_cut = dipole_params['r_c'], nlist=nl)
for i in range(len(types)):
	dipole.pair_coeff.set('C',types[i],mu=0.0, A=0.0, kappa=0.0)
	dipole.pair_coeff.set('B',types[i],mu=0.0, A=0.0, kappa=0.0)
dipole.pair_coeff.set('A','A',mu=dipole_params['mu'], A=dipole_params['eps'] , kappa=dipole_params['kappa'])


lj_params = params['lj']
lj_eps = lj_params['epsilon']
lj = md.pair.lj(r_cut = lj_params['r_c'],nlist =nl)
for i in range(len(types)):
	lj.pair_coeff.set('C',types[i],epsilon=0,sigma=sigma[i])
	lj.pair_coeff.set('A',types[i],epsilon=0,sigma=sigma[i])
lj.pair_coeff.set('B','B',epsilon = lj_eps[2]/(type_nums[2]/4),sigma=sigma[2])
lj.set_params(mode='shift')
wca = md.pair.slj(r_cut=2**(1./6),nlist=nl,d_max=max(sigma))
for i in range(len(types)):
	wca.pair_coeff.set('C',types[i],epsilon=0,sigma=1)
wca.pair_coeff.set('A','A',epsilon=lj_eps[1],sigma=1)
wca.pair_coeff.set('A','B',epsilon=MIXE(lj_eps[1],lj_eps[2]/(type_nums[2]/4)),sigma=1)
wca.pair_coeff.set('B','B',epsilon=0,sigma=1)
wca.set_params(mode='shift')

# Run
deprecated.dump.xml(group=group.all(),filename='%s/init.xml'%(out_dir),all=True)
tetra_centers = group.rigid_center()
for p in tetra_centers:
	p.moment_inertia = [1250.,1250.,1250.]
	print (p.type)
	print (p.tag)

# Randomization
dipole.disable()
md.integrate.mode_standard(dt = 0.005)
langevin1 = md.integrate.langevin(group = tetra_centers,seed = 12345 ,kT = 0.8, dscale=1.0)
dumper = dump.gsd(filename = '%s/randomization.gsd'%(out_dir), period = 1e3, group=group.all(), overwrite=True)
logger = analyze.log(filename = '%s/randomization.txt'%(out_dir),quantities=['num_particles','potential_energy','kinetic_energy','temperature'] ,period = 1e3, overwrite=True)
run(5e6)
dumper.disable()
logger.disable()
langevin1.disable()

# Test production run
dipole.enable()
md.integrate.mode_standard(dt = 0.005)
langevin2 = md.integrate.langevin(group = tetra_centers,seed = 234 ,kT = 0.8, dscale=1.0)
dumper = dump.gsd(filename = '%s/test_production.gsd'%(out_dir), period = 1e4, group=group.all(), overwrite=True)
logger = analyze.log(filename = '%s/test_production.txt'%(out_dir),quantities=['num_particles','potential_energy','pair_lj_energy','kinetic_energy','temperature'] ,period = 1e4, overwrite=True)
run(2e7)
dumper.disable()
logger.disable()

T_ramp = variant.linear_interp(points = [(0, 0.8), (1e8, 0.3)])
langevin2.set_params(kT = T_ramp)
dumper = dump.gsd(filename = '%s/test_cooling.gsd'%(out_dir), period = 2e4, group=group.all(), overwrite=True)
logger = analyze.log(filename = '%s/test_cooling.txt'%(out_dir),quantities=['num_particles','potential_energy','pair_lj_energy','kinetic_energy','temperature'] ,period = 2e4, overwrite=True)
run(1e8)
dumper.disable()
logger.disable()
