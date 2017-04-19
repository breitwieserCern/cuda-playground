Install cuda 8 on Amazon AWS (ubuntu 16.04 LTS)

```
sudo apt-get update
sudo apt-get install -y g++ git cmake valgrind
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# REBOOT instance

echo "export PATH=/usr/local/cuda-8.0/bin:\$PATH" >> ~/.bashrc
. ~/.bashrc
cuda-install-samples-8.0.sh .
cat /proc/driver/nvidia/version
nvcc -V
cd NVIDIA_CUDA-8.0_Samples/
# https://askubuntu.com/questions/891003/failure-in-running-cuda-sample-after-cuda-8-0-installation
find . -type f -execdir sed -i 's/UBUNTU_PKG_NAME = "nvidia-367"/UBUNTU_PKG_NAME = "nvidia-375"/g' '{}' \;
make
bin/x86_64/linux/release/deviceQuery
bin/x86_64/linux/release/bandwidthTest
```
