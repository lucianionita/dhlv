set -e 
#install script

#Step 0. Create an AWS GPU instance with an Ubuntu 12.04 AMI

#Step 1.


echo -e "\e[41m\e[97mGetting system up to date and installing requred packages"
echo -e "\e[0m"
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y install git make python-dev python-setuptools gfortran g++ python-pip screen libblas-dev liblapack-dev
#(python-numpy python-scipy)

# misc stuff to add
echo -e "\e[41m\e[97mInstalling misc stuff"
echo -e "\e[0m"
sudo pip install ipython nose
sudo apt-get -y install htop
sudo apt-get -y install libjpeg-dev libfreetype6-dev zlib1g-dev libpng12-dev 
sudo pip install tiffany jpeg freetype-py Pillow





############## install cuda:
echo -e "\e[41m\e[97mInstalling cuda"
echo -e "\e[0m"

mkdir Downloads
cd Downloads
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_5.5-0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1204_5.5-0_amd64.deb
sudo apt-get -y update
sudo apt-get -y install cuda

echo "export PATH=$PATH:/usr/local/cuda-6.0/bin/" >> ~/.bashrc 
echo export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib/:/usr/local/cuda-6.0/targets/x86_64-linux/lib/ >> ~/.bashrc 
source ~/.bashrc


#############misc


################### blas
echo -e "\e[41m\e[97mInstalling blas"
echo -e "\e[0m"
cd ~/Downloads
git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/usr/local/ install

cd /usr/local/lib
sudo ln -s libopenblas.so /usr/lib/libblas.so
sudo ln -s libopenblas.so.0 /usr/lib/libblas.so.3gf
cd /usr/lib/lapack
sudo ln -s liblapack.so.3gf /usr/lib/liblapack.so.3gf



## numpy scipy and numpy
echo -e "\e[41m\e[97mInstalling numpy"
echo -e "\e[0m"
cd
sudo pip install cython
cd ~/Downloads
git clone https://github.com/numpy/numpy
cd numpy
python setup.py build
sudo python setup.py install

echo -e "\e[41m\e[97mInstalling scipy"
echo -e "\e[0m"
cd
sudo pip install scimath
cd ~/Downloads
git clone https://github.com/scipy/scipy
cd scipy
python setup.py build
sudo python setup.py install



## THEANO
echo -e "\e[41m\e[97mInstalling theano"
echo -e "\e[0m"
cd
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

echo export OPENBLAS_NUM_THREADS=16 >> ~/.bashrc
echo export OMP_NUM_THREADS=16 >> ~/.bashrc
echo -e "[cuda]\nroot = /usr/local/cuda/bin\n\n[global]\ndevice = gpu\nfloatX = float32\nallow_gc = False\n\n[blas]\nldflags = -lopenblas" > ~/.theanorc
source ~/.bashrc


## TEST IF YOU DARE
echo -e "\e[41m\e[97mSimple test"
echo -e "\e[0m"
python -c “import numpy; import scipy; import theano”


### get the code
echo -e "\e[41m\e[97mDownloading deeplearning.net code"
echo -e "\e[0m"
cd
mkdir data
wget -c -r -np http://deeplearning.net/tutorial/code/
mv deeplearning.net/tutorial/code .
rm -fr deeplearning.net
rm code/*htm*
rm code/*/*htm*



### Final test
echo -e "\e[41m\e[97mTesting on CPU and GPU"
echo -e "\e[0m"
source ~/.bashrc
cd code
THEANO_FLAGS="device=cpu" python logistic_sgd.py  
python logistic_sgd.py  

### Finished
echo -e "\e[41m\e[97mInstalation finished!"
echo -e "\e[0m"


