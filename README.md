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








## Variable n-ary dataset details
### 1. WP20K dataset
| Dataset | \|E\| | \|R\| | Arity | All Facts | #Train | #Valid | #Test | #2-ary | #3-ary | #4-ary | #5-ary | #6-ary | #7-ary | #8-ary | #9-ary | #>=5-ary |
|---------|---------|-------|-------|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| **WP20K**   | 19,398  | 41    | 3-8   | 43,769  | 35,116  | 4,326  | 4,327  | -      | 25,296 | 15,169 | 2,514  | 718    | 45     | 27     | -      | 3,304    |
| **WP20K-3** | 11,920  | 41    | 3     | 25,296  | 20,098  | 2,618  | 2,580  |        | 25,296 |        |        |        |        |        |        | -        |
| **WP20K-4** | 9,510   | 40    | 4     | 15,169  | 12,303  | 1,421  | 1,445  |        |        | 15,169 |        |        |        |        |        | -        |
| **WP20K-5** | 3,432   | 20    | 5     | 2,514   | 2,039   | 230    | 245    |        |        |        | 2,514  |        |        |        |        | 2,514    |
| **WP20K-6** | 1,839   | 11    | 6     | 718     | 607     | 56     | 55     |        |        |        |        | 718    |        |        |        | 718      |
| **WP20K-7** | 245     | 6     | 7     | 45      | 43      | 1      | 1      |        |        |        |        |        | 45     |        |        | 45       |
| **WP20K-8** | 106     | 5     | 8     | 27      | 26      | -      | 1      |        |        |        |        |        |        | 27     |        | 27       |

### 2. WP40K Dataset
| Dataset | \|E\| | \|R\| | Arity | All Facts | #Train | #Valid | #Test | #2-ary | #3-ary | #4-ary | #5-ary | #6-ary | #7-ary | #8-ary | #9-ary | #>=5-ary |
|---------|---------|-------|-------|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| **WP40K** | 38,859 | 71 | 2-9 | 355,722 | 283,460 | 36,103 | 36,159 | 311,442 | 25,791 | 15,180 | 2,513 | 718 | 45 | 28 | 5 | 3,309 |
| **WP40K-2** | 30,491 | 71 | 2 | 311,442 | 247,937 | 31,732 | 31,773 | 311,442 | | | | | | | | - |
| **WP40K-3** | 12,249 | 63 | 3 | 25,791 | 20,492 | 2,661 | 2,638 | | 25,791 | | | | | | | - |
| **WP40K-4** | 9,517 | 47 | 4 | 15,180 | 12,314 | 1,421 | 1,445 | | | 15,180 | | | | | | - |
| **WP40K-5** | 3,430 | 19 | 5 | 2,513 | 2,038 | 230 | 245 | | | | 2,513 | | | | | 2,513 |
| **WP40K-6** | 1,839 | 11 | 6 | 718 | 607 | 56 | 55 | | | | | 718 | | | | 718 |
| **WP40K-7** | 245 | 6 | 7 | 45 | 43 | 1 | 1 | | | | | | 45 | | | 45 |
| **WP40K-8** | 106 | 5 | 8 | 28 | 26 | 1 | 1 | | | | | | | 28 | | 28 |
| **WP40K-9** | 27 | 3 | 9 | 5 | 3 | 1 | 1 | | | | | | | | 5 | 5 |
