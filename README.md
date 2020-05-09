# quantum_optimization #

This repository includes the codes for the quantum computing project.

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
pip install -e ./FlowKet
chmod u+x ./script.sh
pip install networkx==2.3
pip install ising
pip install cvxgraphalgs
# conda deactivate cqo
cd src/offshelf/maxcut
python setup.py install --user
cd ../../..
```
Execute the setup file.
```
./setup.sh
```


### 2. Train Model ###

```
./run.sh
```

![MaxCut15](https://github.com/Ericolony/quantum_optimization.git/data/maxcut/graph15.png)
