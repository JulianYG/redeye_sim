#redeye_sim

**RedEye is a vision sensor designed to execute early stages of a deep convolutional neural network (ConvNet) in the analog domain. This repo is a modification of Caffe for three main goals**:
- **Simulate and visualize the noise vs. energy tradeoffs of analog ConvNet processing**
- **Train and test RedEye-specific ConvNets**
- **Build RedEye-ConvNet programs to interface with RedEye hardware (PRIVATE).**

## redeye_sim Modifications to Caffe
- Gaussian Noise Layer: Parameterized with SNR, this layer simulates the signal infidelity of analog processing. 
- Uniform Noise Layer: Parameterized with SNR, this layer uses uniform noise to simulate the signal infidelity of the quantization of the analog-to-digital converter.
- Quantization Layer: Parameterized with bits, this layer truncates data to simulate the quantization of the analog-to-digital converter.
- Weight Digitization: This represents the bit-resolution of the of the fixed-point model weights used in the mixed-signal multiply-accumulate unit.
- (Un)Gamma: Converts nonlinear input pixel values into linear luminance values. 

## Getting Started
- You can either choose to use the caffe source included in redeye_sim, or download and replace the `redeye_sim/caffe` directory with the latest caffe source. For the latter choice, execute `redify.sh` under the `scripts` directory to add redeye_sim modifications. 
- Copy the `makefile` and `config` files from `redeye_sim/configs/caffe` to the `redeye_sim/caffe` directory.
- Call ```make redeye``` command under the root redeye_sim directory.
- Download the ImageNet dataset. Execute `create_lmdb.sh` to generate lmdb files for train and validation.
- Download the pre-trained GoogLeNet caffemodel. Note: keep paths in `data.config` consistent when making data and run workflow.

## Tuning and testing
`redeye_sim/simulation` contains workflows to train the models (`tune.py`) and generates statistics (`validate.py`) for their performance. The workflows automatically inject noise layers to existing models, simulating analog noise behavior. Each workflow sweeps through a range of given noise SNRs, configured as follows:
- Configure datapath directories, batch sizes, and data layer noise in `configs/data.config`. Recommended: use soft links for large data files.
- Configure parameter intervals and prototxt templates in `configs/run.config`. 
  *Specify the sweeping interval of Gaussian noise “gaussian_intvl” and uniform noise “uniform_intvl”, which represents circuit noise and ADC noise, respectively. 
  *Specify the digitization bit-resolution “digitization_bits” of fixed-point weights.
  *Specify the depth (number of convolution layers)  “depth” of analog execution.

`redeye_sim/simulation` also contains a `workflow.py` to procedurally train and test models in the SNR ranges. An example run of the complete workflow: under `/simulation`, execute ```python workflow.py --test_iter 500 --tune_iter 12000 --dest ../stats/test.csv --g 15,30 --q 15,25```. This will sweep the SNR space and run fine tune for each noise-contaminated model for 12000 iterations, write results to `test.csv`, and save the `test.png` plot under `plots` folder. 
After too many runs, you can run `/tools/cleaner.sh` to clean up the messy .prototxt’s and .caffemodel’s if you do not need them anymore. (Keep them if you want to save time running later)

## Limitations & Assumptions
- The training data for simulation does not account for real-world image signal effects, e.g., over-exposure, white-balance.
- The simulation does not include circuit non-idealities, e.g., offset, gain, signal decay.
- Energy estimates for timing circuitry and signal propagation are not included.

## Common Compilation Errors
If you are not using `Makefile` and `Makefile.config` under `configs`, here are some common errors that may occur:
- Cannot find `arrayobject.h` or `Python.h`: look in `site-packages` instead of `dist-packages`, or `/usr/local/lib` instead of `/usr/lib`
- Cannot find `cblas.h`:
Modify the following lines in `Makefile.config`:
(The `BLAS_INCLUDE ?=` in else statement, modify the path to your actual path that contains `cblas.h`. This path can be found by unix command ```find / -name cblas.h```)

``` BASH
		ifneq (,$(findstring version: 6,$(XCODE_CLT_VER)))
			BLAS_INCLUDE ?= /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
			LDFLAGS += -framework Accelerate
		else
			BLAS_INCLUDE ?= /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/
			#LDFLAGS += -framework vecLib
			LDFLAGS += -framework Accelerate
		endif
```
- Cannot load `image _caffe.so`:
Ensure that `DYLD_LIBRARY_PATH` contains `~/caffe/build/lib`, `PYTHONPATH` contains `~/caffe/python`, and `Makefile` contains `libc++11`. Ensure that `libc++11` is correctly set in `CXXFLAGS` and `NVCCFLAGS`. These changes should be made in `Makefile` and `Makefile.config` under `configs/caffe`.

If importing python results in a segfault, you are not using the right python for which caffe is installed. This is usually `/usr/bin/python`.

## License
This repo is distributed for free usage.

