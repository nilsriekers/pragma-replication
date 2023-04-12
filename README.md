# Project Guide

This guide provides instructions for replicating our project with on the basis of the paper "Reasoning about Pragmatics with Neural Listeners and Speakers" by Jacob Andreas and Dan Klein.

## Installation of Apollocaffe

To use the Apollocaffe source code in this project, you'll need to install the framework first. Here's how to do it:

### System Environment
We recommend using a virtual machine with Ubuntu 16.04 to ensure that the code runs correctly. We have encountered difficulties running the code on newer Ubuntu distributions.

### Download the Source Code

Clone the Apollocaffe repository by running the following command in your terminal:
```bash
git clone git@github.com:jacobandreas/apollocaffe.git
```

### Dependencies
To install the necessary dependencies for ApolloCaffe, you can follow the steps below:

1. Install the required packages using the APT package installer:
```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libopenblas-dev python-dev python-numpy libgoogle-glog-dev libgflags-dev liblmdb-dev
```

2. Install the following Python related packages using APT:
```bash
sudo apt-get install python-dev python-pip python-scipy python-h5py python-numpy
```

3. Install the required Python packages using pip by running the following command:
```bash
for req in $(cat /python/requirements.txt); do pip install $req; done
```

4. If problems occur, install the following packages in an older version using pip:
```bash
sudo apt-get install build-essential autoconf libtool
pip install protobuf==3.15.8
pip install pyyaml==3.13
```
5. If you encounter more problems with pip, you can manually install the required Python packages from source by following these steps:

Download the package archive from the internet. Extract the archive using the following command:
```bash
tar zxvf Package-Name.tar.gz
```
Change into the extracted directory and install the package using the following command:
```bash
cd Package-Name
python setup.py install
```

### Compilation of ApolloCaffe
To compile ApolloCaffe, follow these steps:

1. Modify the "Makefile.config.example" file,  and save it as a new file called "Makefile.config".
2. Set the flag "CPU_ONLY := 1" in the "Makefile.config" file.
3. Add the path "/usr/include/hdf5/serial/" to the "INCLUDE_DIRS" variable in the "Makefile.config" file.
4. Add the path "/usr/lib/x86_64-linux-gnu/hdf5/serial/" to the "LIBRARY_DIRS" variable in the "Makefile.config" file.
5. Run the command "make all".
6. Run the command "make test" (optional).
7. Run the command "make runtest" (optional).
8. Set the following environment variables in the "~/.bashrc" file:
```bash
export LD_LIBRARY_PATH=/path/to/apollocaffe/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/apollocaffe/python:$PYTHONPATH
```
9. Reload the Bash Settings:
```bash
source ~/.bashrc
```
Note: If you are experiencing problems with the available working memory, use our script to enable swapping on your local machine (requires 8GB of disk space):
```bash
bash swap.sh
```

## Running the Model
To run the model, follow these steps:
1. Clone our repo and navigate to the directory:
```bash
git clone git@github.com:nilsriekers/pragma-replication.git
cd pragma-replication
```
2. Download the "Abstract Scenes Dataset" and unpack it in a folder called "data".
3. Create a folder called "models".
4. Comment out the line "apollocaffe.set_device(0)" in the "main.py" file.
5. Adjust the "BASE_DIR" variable with path to the correct location of the data in the corpus.py file.
6. Use the following command to train the model on the abstract dataset:
```bash
python main.py train.base abstract
```
7. Use the following command to run the experiments with the trained model:
```bash
python main.py sample.base abstract
```

## Annotated Python3 Code
The "annotated_python3_code" folder contains the extensively commented source files of Andreas and Klein's implementation. These files differ from the original only in that they are Python 3 code instead of Python 2 code as in the original. The conversion was done automatically using the standard Python tool "2to3" to replace the old ApolloCaffe with PyTorch. At this early stage, we did not know that we would be able to run the original code with ApolloCaffe, despite all the challenges. However, as we had already commented much of the code early on, we decided to leave it in Python 3 and not manually convert it back (which would have been necessary) because of the work invested. In terms of content and concept, the Python version does not affect the functionality.

## Hyperparameter Optimization
1. Use this command to perform the hyperparameter optimization:
```bash
python hyperparameter_optimization.py
```
The Python script "hyperparameter_optimization.py" produces a number of output files. These include:
- Text files with the model name, containing the loss and accuracy values of the test/validation data set of the trained "listener0_model" and "speaker0_model" for each epoch. These files are stored in the "performance" folder.
- The generated apollocaffe models themselves are saved with the corresponding model name. These files are stored in the "models" folder.
2. Use this command to create the figures from our research paper use this command:
```bash
python create_plots.py
```
The generated figures will be stored in the "figures" folder.

## PyTorch (TBD)
The idea was to migrate the existing ApolloCaffe codebase to PyTorch. However, it is not yet executable. 

TBD.

## Written Paper
A written paper on the project is also available in the repository: [Click here to download the PDF paper](Reasoning_about_Pragmatics_with_Neural_Listeners_and_Speakers.pdf)
