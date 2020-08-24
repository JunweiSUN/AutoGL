# AutoGL

## What is AutoGL？

AutoGL is graph learning framework with automatic machine learning techniques. AutoGL now mainly focus on node classification problems, but it's easy to apply this program to other graph learning problems.

AutoGL is the 6th solution for AutoGraph Challenge@KDD'20, the competition rules can be found [here](https://www.automl.ai/competitions/3). We achieve 1st, 4th, 1st, 6th and 27th on 5 final phase datasets. 
| #   | Dataset1     | Dataset2    | Dataset3   | Dataset4   | Dataset5 | Avg |
| --- | -------- | ------- | -------- | ------ | ----------- | ---------------------- | 
| rank   | 1  | 4   | 1      | 6  | 27 | 7.8 |

## Usage
Clone this repository to your machine:
```
git clone https://github.com/JunweiSUN/AutoGL.git
```
Download datasets from [here](https://www.automl.ai/competitions/6?secret_key=c10be8ef-9a94-417d-bb7a-5711aa6c895b#learn_the_details). You can also create your own datasets with required format.<br>
When the download process finished, unzip the datasets and move them to the `data` folder. Or you can just simple use the demo dataset in `data`.<br>

AutoGL could be easily started with [docker](https://www.docker.com/):
```
cd path/to/AutoGL/
docker run --gpus=0 -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2
python run_local_test.py --dataset_dir=./data/demo --code_dir=./code_submission
```
You can change the argument dataset_dir to other datasets. On the other hand, you can also modify the directory containing your other sample code.<br>

You can also use your own python environment to run this program. In this way, you must install all the necessary packages. So we recommend users to run this program with docker.

## Acknowledgements
We refer to these packages and codes when developing this program:<br>

[nni](https://github.com/microsoft/nni): An open source AutoML toolkit from microsoft<br>
[AutoDL (tabular part)](https://github.com/DeepWisdom/AutoDL/tree/master/AutoDL_sample_code_submission/Auto_Tabular): Automated Deep Learning without ANY human intervention<br>
[pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): Geometric Deep Learning Extension Library for PyTorch<br>
[sparsesvd](https://github.com/RaRe-Technologies/sparsesvd): a fast library for sparse Singular Value Decomposition<br>
[DropEdge](https://github.com/DropEdge/DropEdge): a Pytorch implementation of paper: DropEdge: Towards Deep Graph Convolutional Networks on Node Classification

## Contact us
If you have any question or advice, please feel free to contact our team members:<br>
Junwei Sun: junweisun@bupt.edu.cn<br>
Ruifeng Kuang: kuangruifeng@bupt.edu.cn<br>
Wei Huang: 18262998091@163.com
Changrui Mu: u3553427@connect.hku.hk
Jiayan Wang: jiayanwangno1@gmail.com

## License 
[Apache License 2.0](https://github.com/JunweiSUN/AutoGL/blob/master/LICENSE)
