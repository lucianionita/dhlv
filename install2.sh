################### blas
echo -e "\e[41m\e[97mInstalling blas"
echo -e "\e[0m"
cd ~/Downloads
git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS
make -j 8 FC=gfortran
sudo make PREFIX=/usr/local/ install
sudo make PREFIX=/usr/ install

cd /usr/local/lib
sudo unlink /usr/lib/libblas.so
sudo unlink /usr/lib/libblas.so.3gf
sudo ln -s libopenblas.so /usr/lib/libblas.so
sudo ln -s libopenblas.so.0 /usr/lib/libblas.so.3gf
#sudo ln -s libopenblas.so /usr/lib/libopenblas.so.0
#sudo ln -s libopenblas.so libopenblas.so.0
cd /usr/lib/lapack
# !!! REMOVE THIS COMMENT
#sudo unlink /usr/lib/liblapack.so.3gf
sudo ln -s liblapack.so.3gf /usr/lib/liblapack.so.3gf



## numpy scipy and numpy
echo -e "\e[41m\e[97mInstalling numpy"
echo -e "\e[0m"
cd ~/Downloads
sudo pip install cython
git clone https://github.com/numpy/numpy
cd ~/Downloads
cd numpy
python setup.py build
sudo python setup.py install

echo -e "\e[41m\e[97mInstalling scipy"
echo -e "\e[0m"
cd
# for some reason this doesn't work anymore:
# sudo pip install scimath 
# so we do this instead
cd ~/Downloads
git clone https://github.com/enthought/scimath
cd scimath
python setup.py build
sudo python setup.py install

# and we move on to installing scipy
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
python -c "import numpy; import scipy; import theano"


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


