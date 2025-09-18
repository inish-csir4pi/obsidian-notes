## Pytorch 2.10-Cuda 12.8
##### Create New Environment 
```bash
conda create -n torch python=3.11
conda activate torch
```
#### Install Pytorch Nightly with RTX 5070 Support
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
#### Verify Installation
[torch_test.py](torch_test.py)
![[torch_test.py]]
#### Run GPU Test
![[gpu_test.py]]
#### (Optional) Check for cuda-out-of-memory
![[5070-memory-test.py]]
