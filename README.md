# ResidualDenseNetwork-Pytorch 

Pytorch implement: [Residual Dense Network for Image Super-Resolution](https://arxiv.org/pdf/1802.08797.pdf)
# Requirements
- python3.5 / 3.6
- pytorch >= 0.2
- opencv 


# Usage

    
    python3 main.py --model_name 'RDN' --load demo_x3_RDN --dataDir ./DIV2K/ --need_patch True  --patchSize 144 --nDenselayer 3 --nFeat 64 --growthRate 32  --scale 3 --epoch 10000 --lrDecay 2000 --lr 1e-4 --batchSize 16 --nThreads 4 --lossType 'L1' 
        

# References
[densenet-pytorch](https://github.com/andreasveit/densenet-pytorch) 

[EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)
