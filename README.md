# Natural Evolution Strategies and Quantum Approximate Optimization #

Tianchen Zhao, Giuseppe Carleo, James Stokes and Shravan Veerapaneni

This repository includes the codes for the paper "Natural Evolution Strategies and Quantum Approximate Optimization" (https://arxiv.org/pdf/2005.04447.pdf).

## How to Use ##

Download this repository.
```
git clone https://github.com/Ericolony/quantum_optimization.git
```

Get into working directory.
```
cd quantum_optimization
```


### 1. Environment Setup ###
Create a file named "setup.sh". Make it executable.
```
chmod u+x setup.sh
```

Copy the following code into the file setup.sh.
```
conda create -n cqo python=3.7
conda activate cqo

conda install -c conda-forge tensorflow=1.14 -y
conda install -c pytorch pytorch -y
conda install cmake mpich numpy scipy -y
conda install -c anaconda cupy -y
conda install -c anaconda mpi4py -y
conda install -c numba numba -y

pip install cvxpy
pip install cvxopt
pip install netket
pip install networkx==2.3
pip install ising
pip install cvxgraphalgs

cd src/offshelf/maxcut
python setup.py install --user
cd ../../..
chmod u+x ./run.sh
# conda deactivate cqo
```
Execute the setup file.
```
./setup.sh
```


### 2. Demo ###

Run the following script for evaluations of the maxcut algorithms on a graph instance with 20 nodes

```
./run.sh
```

which produces the following results:
```
./results.txt

[ground truth] - Score: 35.00, Time: 245.67 seconds

optimal state: {19: -1, 18: 1, 17: -1, 16: -1, 15: -1, 14: -1, 13: -1, 12: 1, 11: 1, 10: 1, 9: -1, 8: 1, 7: 1, 6: 1, 5: 1, 4: -1, 3: -1, 2: -1, 1: 1, 0: -1}

[random_cut] - Score: 17.00, Time: 0.01 seconds

[goemans_williamson] - Score: 33.00, Time: 0.29 seconds

[manopt] - Score: 35.00, Time: 0.58 seconds, Bound: 35.80

[netket] - Score: 34.81, Time: 20.29 seconds
```

Visualization of the graph instance:

![MaxCut20](https://github.com/Ericolony/quantum_optimization/blob/master/data/maxcut/graph20.png)


