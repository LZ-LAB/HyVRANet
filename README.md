This paper has been submitted to KDD 2026.

All local experiments are obtained on 4*NVIDIA GeForce RTX 3090 GPUs and PyTorch 1.12.0.



## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch 1.12.0
    -- numpy 1.26.0



## How to Run HyVRANet

```
## WP20K dataset
python main-WP20K.py --dataset WP20K --batch_size 400 --lr 0.00092 --dr 0.99 --input_drop 0.8 --hidden_drop 0.4 --feature_drop 0 --VarRAC_Size 5 --PosRAC_Size 18 --gen_lr 0.0001 --dis_lr 0.0001

## WP40K dataset
python main-WP40K.py --dataset WP40K --batch_size 400 --lr 0.00030 --dr 0.995 --input_drop 0.7 --hidden_drop 0.2 --feature_drop 0.3 --VarRAC_Size 7 --PosRAC_Size 15 --gen_lr 0.0004 --dis_lr 0.0003

## WP20K-4 dataset
python main-WP20K-4.py --dataset WP20K-4 --batch_size 400 --lr 0.00035 --dr 0.99 --input_drop 0.6 --hidden_drop 0.3 --feature_drop 0.6 --VarRAC_Size 12 --PosRAC_Size 4 --gen_lr 0.0004 --dis_lr 0.0003

## WP40K-5 dataset
python main-WP40K-5.py --dataset WP40K-5 --batch_size 800 --lr 0.00022 --dr 0.999 --input_drop 0.6 --hidden_drop 0.3 --feature_drop 0.6 --VarRAC_Size 3 --PosRAC_Size 5 --gen_lr 0.0001 --dis_lr 0.0001

## FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 1000 --lr 0.00022 --dr 0.995 --input_drop 0.7 --hidden_drop 0.5 --feature_drop 0.2 --VarRAC_Size 2 --PosRAC_Size 16 --gen_lr 0.0005 --dis_lr 0.0001

## WikiPeople dataset
python main-WikiPeople.py --dataset WikiPeople --batch_size 400 --lr 0.00080 --dr 0.995 --input_drop 0.7 --hidden_drop 0.2 --feature_drop 0.2 --VarRAC_Size 4 --PosRAC_Size 16 --gen_lr 0.0001 --dis_lr 0.0001

## WikiPeople-3 dataset
python main-WP3.py --dataset WikiPeople-3 --batch_size 900 --lr 0.00012 --dr 0.99 --input_drop 0.3 --hidden_drop 0.1 --feature_drop 0.1 --VarRAC_Size 18 --PosRAC_Size 18 --gen_lr 0.0005 --dis_lr 0.0002

## JF17K-4 dataset
python main-JF4.py --dataset JF17K-4 --batch_size 800 --lr 0.00012 --dr 0.99 --input_drop 0.5 --hidden_drop 0 --feature_drop 0.1 --VarRAC_Size 13 --PosRAC_Size 8 --gen_lr 0.0003 --dis_lr 0.0003

```


