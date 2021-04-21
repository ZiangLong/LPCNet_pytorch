# LPCNet_pytorch
A Pytorch version of LPCNet, including dump weight

- J.-M. Valin, J. Skoglund, [A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet](https://jmvalin.ca/papers/lpcnet_codec.pdf), *Submitted for INTERSPEECH 2019*.
- J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, arXiv:1810.11846, 2019.

# Download Data
Use together with this [repo](https://github.com/mozilla/LPCNet).
Suitable training material can be obtained from the [McGill University Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/).  Download the ISO and extract the 16k-LP7 directory, the src/concat.sh script can be used to generate a headerless file of training samples.
```
cd 16k-LP7
sh /path/to/concat.sh
```
# Dump Training Data
Use together with this [repo](https://github.com/mozilla/LPCNet).
```
./dump_data -train input.s16 features.f32 data.u8
```

# Training
```
python main.py --feat features.f32 --data data.u8
```

# Testing
```
./dump_data -test test_input.s16 test_features.f32
python test.py --feat test_features.f32
```

# Dump LPCNet weights
```
python dump_lpcnet.py --load <path/to/checkpoint>
```
