# ResidualDenseNetwork-Pytorch  

Pytorch implement: [Residual Dense Network for Image Super-Resolution](https://arxiv.org/pdf/1802.08797.pdf)  

I think two advantage ideas of the paper:
- join denese connect layer to ResNet
- concatenation of hierarchical features


Different with the paper, I just use there RDBs(Residual dense block), every RDB has three dense layers. So ,this is a sample implement the RDN(Residual Dense Network) proposed by the author.



# Requirements
- python3.5 / 3.6
- pytorch >= 0.2
- opencv 


# Usage
you need prepare DIV2K dataset (./data/)
train model :
    
    python3 main.py --model_name 'RDN' --load demo_x3_RDN --dataDir ./DIV2K/ --need_patch True  --patchSize 144 --nDenselayer 3 --nFeat 64 --growthRate 32  --scale 3 --epoch 10000 --lrDecay 2000 --lr 1e-4 --batchSize 16 --nThreads 4 --lossType 'L1' 
        
        

# References
[densenet-pytorch](https://github.com/andreasveit/densenet-pytorch) 

[EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)
