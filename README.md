# Machine Learning Sample

## Getting Started

### Note to Windows Users

The code will not run on Windows due to package incompatibilities. If developing from Windows, please be sure to install
WSL and configure your IDEs accordingly.

### Hardware

The code is designed to run on an NVIDIA GPU.

### Operating System

The code is designed to run on Linux. The code is written with PyCharm on Windows, but is run using WSL 2.0 within
Windows using a clean Linux Ubuntu 24.04 LTS distribution.

> Remember to ensure that the NVIDIA CUDA Toolkit and drivers are installed on Windows for use by WSL. The NVIDIA 
> CUDA Toolkit will also need to be installed on the Linux distribution. As always, ensure that all operating systems
> are up-to-date before you begin.

### Software

To simplify the process, the following steps are recommended in exact order:
1. Install the NVIDIA CUDA Toolkit, version 12.4. You may run into an error, see the guidance within
   [this article](https://askubuntu.com/questions/1491254/installing-cuda-on-ubuntu-23-10-libt5info-not-installable) 
   to resolve the issue.
2. Install cuDNN for CUDA 12.
3. Ensure that Python 3.12, pip, and the `venv` module are installed.
4. Install `venv` and create the `machine_learning_sample` virtual environment.
5. Install `pytorch` for CUDA 12.4, `tensorflow`, `pytest`, and `pytest-mock`.

In Ubuntu 24.04, Python 3.12 is installed by default. The following commands are used to install the remaining 
Python components:

```shell
sudo apt install python3-pip
sudo apt install python3-venv
````

To create and activate the virtual environment, the following commands are used:

```shell
python3 -m venv ~/.local/share/virtualenvs/machine_learning_sample
source ~/.local/share/virtualenvs/machine_learning_sample/bin/activate
````

> You'll need to configure your IDE to use the virtual environment.

To deactivate the virtual environment, the following command is used:

```shell
deactivate
````

The commands used for PyTorch and TensorFlow as well as the remaining packages are as follows:

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install tensorflow
pip3 install pytest pytest-mock
```

#### References

- [NVIDIA CUDA Toolkit 12.4 Installation](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- [cuDNN for CUDA 12 Installation](https://developer.nvidia.com/cudnn-downloads)
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [TensorFlow Installation](https://www.tensorflow.org/install)
- [Hugging Face Transformers Installation](https://huggingface.co/docs/transformers/installation)

## Tests

Tests are provided to make changing the code easier, which facilitates learning activities.
