# quantum_optimization #

This repository includes the codes for the river detection project.

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
source activate cqo

conda install -c conda-forge tensorflow=1.14 -y
conda install -c anaconda seaborn -y
pip install flowket
conda install cmake mpich numpy scipy -y
pip install netket
chmod u+x ./script.sh
```
Execute the setup file.
```
./setup.sh
```


### 2. Train Model ###

```
./script.sh
```

