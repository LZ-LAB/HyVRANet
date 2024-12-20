All local experiments are obtained on NVIDIA GeForce RTX 3090 GPUs and PyTorch 1.12.0.



## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch 1.12.0
    -- numpy 1.26.0
    -- tqdm 4.65.0



## How to Run HyMRformer

#### 1. Mixed Arity Knowledge Hypergraph
```
## JF17K dataset
python main-JF.py --dataset JF17K --batch_size 400 --lr 0.00020 --dr 0.999 --input_drop 0.5 --hidden_drop 0.4 --feature_drop 0.2 --dembed 100 --RAD_Size 2 --num_heads 4

## WikiPeople dataset
python main-WP.py --dataset WikiPeople --batch_size 700 --lr 0.00025 --dr 0.995 --input_drop 0.7 --hidden_drop 0.9 --feature_drop 0.5 --dembed 500 --RAD_Size 3 --num_heads 1

## FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 800 --lr 0.00010 --dr 0.995 --input_drop 0.6 --hidden_drop 0.5 --feature_drop 0.5 --dembed 400 --RAD_Size 6 --num_heads 4
```


#### 2. Fixed Arity Knowledge Hypergraph
```
## WikiPeople-3 dataset
python main-3ary.py --dataset WikiPeople-3 --batch_size 600 --lr 0.00003 --dr 0.995 --input_drop 0.3 --hidden_drop 0.2 --feature_drop 0.2 --dembed 500 --RAD_Size 10 --num_heads 8

## JF17K-4 dataset
python main-4ary.py --dataset JF17K-4 --batch_size 600 --lr 0.00021 --dr 0.995 --input_drop 0.5 --hidden_drop 0.2 --feature_drop 0.5 --dembed 500 --RAD_Size 3 --num_heads 4

## JF17K-5 dataset
python main-5ary.py --dataset JF17K-5 --batch_size 600 --lr 0.00031 --dr 0.995 --input_drop 0.6 --hidden_drop 0.1 --feature_drop 0.4 --dembed 300 --RAD_Size 7 --num_heads 12

## WikiPeople-5 dataset
python main-5ary.py --dataset WikiPeople-5 --batch_size 700 --lr 0.00056 --dr 0.999 --input_drop 0.6 --hidden_drop 0.5 --feature_drop 0.2 --dembed 200 --RAD_Size 3 --num_heads 4
```


#### 2. Binary Knowledge Graph
```
## FB15K-237 dataset
python main-2ary.py --dataset FB15K-237 --batch_size 100 --lr 0.00058 --dr 0.99 --input_drop 0.5 --hidden_drop 0.4 --feature_drop 0.6 --dembed 100 --RAD_Size 6 --num_heads 1

## JF17K-2 dataset
python main-2ary.py --dataset JF17K-2 --batch_size 900 --lr 0.00012 --dr 0.99 --input_drop 0.1 --hidden_drop 0.6 --feature_drop 0.1 --dembed 500 --RAD_Size 6 --num_heads 8

## WikiPeople-2 dataset
python main-2ary.py --dataset WikiPeople-2 --batch_size 800 --lr 0.00037 --dr 0.995 --input_drop 0.7 --hidden_drop 0.6 --feature_drop 0.5 --dembed 400 --RAD_Size 10 --num_heads 1

## FB-AUTO-2 dataset
python main-2ary.py --dataset FB-AUTO-2 --batch_size 300 --lr 0.00027 --dr 0.999 --input_drop 0.9 --hidden_drop 0.2 --feature_drop 0.7 --dembed 500 --RAD_Size 1 --num_heads 1
```

## Baseline Models
We are very grateful for all open-source baseline models:

#### HypE/HSimplE/m-TransH/m-CP/m-DitMult - https://github.com/ElementAI/HypE
```
1. HypE
python main.py -model HypE -dataset *** -batch_size 128 -lr 0.1  -filt_w 1 -out_channels 6 -stride 2 -emb_dim 200 -nr 10 -num_iterations 300

2. HSimplE
python main.py -model HSimplE -dataset *** -batch_size 128 -lr 0.01 -emb_dim 200 -nr 10 -num_iterations 300

3. m-TransH
python main.py -model MTransH -dataset *** -batch_size 128 -lr 0.06 -emb_dim 200 -nr 10 -num_iterations 300

4. m-CP
python main.py -model MCP -dataset *** -batch_size 128 -lr 0.02 -emb_dim 34 -nr 10 -num_iterations 500

5. m-DitMult
python main.py -model MDistMult -dataset *** -batch_size 128 -lr 0.02 -emb_dim 200 -nr 10 -num_iterations 500
```

#### GETD - https://github.com/liuyuaa/GETD
```
1. JF17K-3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset JF17K-3 --num_iterations 200 --batch_size 128 --edim 50 --rdim 50 --k 4 --n_i 50 --TR_ranks 50 --dr 0.99 --lr 0.0008658318809880197 --input_dropout 0.12747824547053027 --hidden_dropout 0.501929359180091

2. JF17K-4
CUDA_VISIBLE_DEVICES=0 python main.py --dataset JF17K-4 --num_iterations 200 --batch_size 128 --edim 25 --rdim 25 --k 5 --n_i 25 --TR_ranks 40 --dr 0.995 --lr 0.0006071265071591076 --input_dropout 0.010309222253012645 --hidden_dropout 0.43198147413900445

3. WikiPeople-3
CUDA_VISIBLE_DEVICES=0 python main.py --dataset JF17K-3 --num_iterations 200 --batch_size 128 --edim 50 --rdim 50 --k 4 --n_i 50 --TR_ranks 50 --dr 0.995 --lr 0.0009267003174594345 --input_dropout 0.3740776415163665 --hidden_dropout 0.45137914784181227

4. WikiPeople-4
CUDA_VISIBLE_DEVICES=0 python main.py --dataset JF17K-3 --num_iterations 200 --batch_size 128 --edim 25 --rdim 25 --k 5 --n_i 25 --TR_ranks 40 --dr 0.995 --lr 0.006701566797680926 --input_dropout 0.46694419227220374 --hidden_dropout 0.18148844341064124
```

#### RAM - https://github.com/liuyuaa/RAM
```
1. JF17K dataset
python main-JF.py --dataset JF17K --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4

2. WikiPeople dataset
python main-WP.py --dataset WikiPeople --batch_size 128 --lr 0.003 --dr 0.995 --K 10 --rdim 25 --m 2 --drop_role 0.0 --drop_ent 0.2

3. FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4
```

#### PosKHG - https://github.com/zirui-chen/PosKHG
```
1. JF17K dataset
python main-JF.py --dataset JF17K --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4

2. WikiPeople dataset
python main-WP.py --dataset WikiPeople --batch_size 64 --lr 0.003 --dr 0.995 --K 10 --rdim 25 --m 2 --drop_role 0.0 --drop_ent 0.2

3. FB-AUTO dataset
python main-FB.py --dataset FB-AUTO --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4
```

#### ReAlE - https://github.com/baharefatemi/ReAlE
```
1. ReAlE—Small
python main.py -dataset *** -lr 0.08 -nr 10 -window_size 2 -batch_size 128 -num_iterations 500

2. ReAlE
python main.py -dataset *** -lr 0.08 -nr 100 -window_size 2 -batch_size 512 -num_iterations 500
```

### Supplementary Information
Since the source codes of baseline models HypE, HSimplE, m-TransH, m-CP, m-DitMult, RD-MPNN, and ReAlE support a maximum Arity of 6, none of them can handle the semantically rich WikiPeople dataset.
We have rationalized their source codes to make them applicable to the WikiPeople dataset.
A modified version of `Arity>6` with the `HypE/HSimplE` model as the core framework is uploaded in our open source code. The code frameworks of other models (e.g. m-TransH, m-CP, m-DitMult, RD-MPNN, and ReAlE) are based on the HypE model, so they are all the same.



