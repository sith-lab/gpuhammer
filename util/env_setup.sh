cd $HAMMER_ROOT/src
if [ ! -d "rmm" ]; then
  echo "rmm does exist."
  git clone -b branch-25.04 https://github.com/rapidsai/rmm.git
fi
cd $HAMMER_ROOT/src/rmm

if ! (conda info --envs | grep -q rmm_dev); then
  conda env create --name rmm_dev --file conda/environments/all_cuda-128_arch-x86_64.yaml
fi

conda init
source activate base
conda activate rmm_dev

if [ ! -d "rmm_lib" ]; then
    rm -rf build
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../rmm_lib
    make -j
    make install
fi

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install matplotlib
python3 -m pip install scipy
