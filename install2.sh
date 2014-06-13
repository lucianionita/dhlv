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


