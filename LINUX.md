## **GNU/Linux installation instructions**
**!!! FFmpeg and NVIDIA Driver shall be already installed !!!**

First of all, i strongly recommend to install Anaconda, that it was convenient to work with DeepFaceLab.

Official instruction: https://docs.anaconda.com/anaconda/install/linux

After, you can create environment with packages, needed by DeepFaceLab:
```
conda create -y -n deepfacelab python==3.6.6 pathlib==1.0.1 scandir h5py==2.7.1 Keras==2.1.6 tensorflow-gpu==1.8.0 scikit-image tqdm
```
Then activate environment:
```
source activate deepfacelab
```
And install the remained packages:
```
conda install -y -c conda-forge opencv==3.4.1
pip install dlib==19.10.0 git+https://www.github.com/keras-team/keras-contrib.git
```
Now clone the repository and run...  Good luck ;-)
```
git clone https://github.com/iperov/DeepFaceLab
cd DeepFaceLab && chmod +x main.sh && ./main.sh
```
**NOTE !!! Before launching DeepFaceLab, you should convince in that you already executed "source activate deepfacelab" !!!**

P.S. English is not my native language, so please be kind to my mistakes.
