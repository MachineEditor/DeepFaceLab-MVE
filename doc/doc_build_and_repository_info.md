## Build and Repository Info

DeepFaceLab officially supports Windows-only. If you want to support Mac/Linux/Docker - create a fork, it will be referenced here.

[Linux fork](https://github.com/lbfs/DeepFaceLab_Linux) by @lbfs

#### **Installing dlib on Windows**

The version of `dlib` in pip is compiled without CUDA support. Therefore you have to compile it manually in order to use the `dlib` face extractor.

Command line example for Windows: `python setup.py install -G "Visual Studio 14 2015" --yes DLIB_USE_CUDA`

#### **CPU mode**

It is possible to run from script for all stages using the `--cpu-only` flag. To run from script, install the separate dependencies for CPU mode using `pip -r requirements-cpu.txt`.

Please note that extraction and training will take much long without a GPU and performance will greatly suffer without one. In particular, do not use DLIB extractor in CPU mode, it's too slow to run without a GPU. Train only on 64px resolution models like H64 or SAE (with low settings) and the lightweight encoder.