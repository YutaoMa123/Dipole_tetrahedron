import json
import pickle
import os
import numpy as np

def gen_json(A,kappa,file_prefix):
  structure = {
    "HOOMD_params": "Values for setting up HOOMD",
    "output_dir": file_prefix,
    "System params": "Values for setting up system",
    "L": 100.0,
    "N": 2,
    "dt": 0.005,
    "seed": np.random.randint(100,99999),
    "Colloid params": "Colloidal particles params",
    "types": ["C", "A", "B"],
    "sigma": [ 0.0, 8.0, 1.0],
    "mass": [100.0, 1.0, 1.0],
    "phi_max": 30.0,
    "alpha": 0.3,
    "dipole": {
      "r_c": 24.0,
      "mu": 1.0,
      "eps": A,
      "kappa": kappa
    },
    "lj": {
      "r_c": 4.0,
      "epsilon" : [0.0,2.0,40.0]
    }
  }
  with open('%s.json'%(file_prefix), 'w') as json_file:
    json.dump(structure, json_file, indent=4)

def gen_sbatch(sbatch_prefix,file_prefix,filename = 'test_production.gsd'):
    with open ('%s.sbatch'%(sbatch_prefix),'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --mail-user=jaulicino@uchicago.edu\n')
        f.write('#SBATCH --job-name=%s\n'%(file_prefix))
        f.write('#SBATCH --output=%s/%s.out\n'%(file_prefix,file_prefix))
        f.write('#SBATCH --partition=fela\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --cpus-per-task=1\n')
        f.write('#SBATCH --constraint=ib\n')
        f.write('module load hoomd\n')
        f.write('source /home/jaulicino/midway2_python_env/bin/activate\n')
        f.write('python test_tetra.py %s.json\n'%(file_prefix))
        f.write('mv %s.json %s\n'%(file_prefix,file_prefix))
        f.write('mv %s.sbatch %s/\n'%(sbatch_prefix,file_prefix))
        f.write('python analyze_bond_angle.py %s %s\n'%(file_prefix, filename))

if __name__ == "__main__":
  with open('../iteration_data.pickle','rb') as f:
    iteration_data = pickle.load(f)
  data = iteration_data[-1]
  A = data['A']
  kappa = data['kappa']
  n_candidates = len(A) 
  for i in range(n_candidates):
    cur_A = A[i]
    cur_kappa = kappa[i]
    for j in range(1,4):
      file_prefix = 'Run_%d_%d'%(i+1,j)
      os.mkdir(file_prefix)
      gen_json(cur_A,cur_kappa,file_prefix)
      sbatch_prefix = 'sub_%d_%d'%(i+1,j)
      gen_sbatch(sbatch_prefix,file_prefix)


