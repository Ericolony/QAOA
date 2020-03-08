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
conda install cmake mpich numpy scipy -y
pip install netket
chmod u+x ./script.sh
pip install networkx==2.3
pip install ising
# conda deactivate cqo
```
Execute the setup file.
```
./setup.sh
```


### 2. Train Model ###

```
./script.sh
```

