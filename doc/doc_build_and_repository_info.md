### **CPU only mode**

CPU mode enabled by arg --cpu-only for all stages. Follow requirements-cpu.txt to install req packages.
Do not use DLIB extractor in CPU mode, it's too slow.
Only H64 or SAE (with low settings) models reasonable to train on home CPU.

### **Build info**

dlib==19.10.0 from pip compiled without CUDA. Therefore you have to compile DLIB manually, orelse use MT extractor only.

Command line example for windows: `python setup.py install -G "Visual Studio 14 2015" --yes DLIB_USE_CUDA`

### Mac/linux/docker script support.

If you want to support mac/linux/docker - create fork, it will be referenced here.
