# quantum_optimization #

This repository includes the codes for the river detection project.

## How to Use ##

Download this repository.
```
git clone https://github.com/Ericolony/RiverDetection.git
```

Get into working directory.
```
cd RiverDetection
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
```
Execute the setup file.
```
./setup.sh
```

Then, download the zip file in the box named "dataset_FinalVersion.zip" to the current working directory. Unzip the zipfile, clean up the files.
```
unzip dataset_FinalVersion.zip
mv ./data/data ./data1
rm -r -f data
mv ./data1 ./data
rm dataset_FinalVersion.zip
```


### 2. Train new classifier&detector ###
Eric: This will be modified later.

```
cd src
# train a classifier
python train_discriminator.py --num_gpu=8 --lr=1e-2 --batch_size=128 --crop_size=128 --sample_intensity=75 --discriminator_name=c1 --pos_weight=1.0
# train a detector
python main.py --num_gpu=8 --lr=2e-4 --model_name=nesteduet --crop_size=128 --sample_intensity=75
```

If you are working in a environment without GPUs, change the flag "--num_gpu=8" to "--num_gpu=0".


### Download the data for the whole state ###
Download gsutil (https://cloud.google.com/storage/docs/gsutil_install#linux), restart terminal if necessary.
```
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

Get back to the directory RiverDetection (in parallel with src) and download dataset.
```
cd ..
gsutil cp -r gs://rrap-or-pipeline ./
```


### Inference ###
```
cd src
python predict_map.py --statemap_name=Sep_Prediction --map_name=sep --model_name=nesteduet --model_load_path=DETECTOR_PATH --discriminator_name=c1 --discriminator_load_path=CLASSIFIER_PATH --crop_size=128 --clip_limit=1.0
python predict_map.py --statemap_name=June_Prediction --map_name=june --model_name=nesteduet --model_load_path=DETECTOR_PATH --discriminator_name=c1 --discriminator_load_path=CLASSIFIER_PATH --crop_size=128 --clip_limit=1.0
```


