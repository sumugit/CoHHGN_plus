This is our implementation for the paper:

_Pseudo Session-based Recommendation with Hierarchical Embedding and Session Attributes [[arXiv link](https://arxiv.org/abs/2306.10029)]_ 

Yuta Sumiya, Ryusei Numata, Satoshi Takahashi

_at DS, 2023_

## Environments
- Python 3.9
- Pytorch 1.13.1
- Numpy 1.24.1

## Datasets
The data used in this study was provided by the sponsors of the Data Analysis Competition, Joint Association Study Group of Management Science (JASMAC) and Rakuten Group, Inc., and it consists of Rakuten marketplace purchase history data from 2019 to 2020.

By using any observed browsing and purchase history data that includes criteria such as the user's gender and location, you can conduct similar experiments to evaluate pseudo-session behaviors.

## Usage

Data preprocessing:

The code for data preprocessing can refer to following:
- [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN)
- [GCE-GNN](https://github.com/CCIIPLab/GCE-GNN)
- [CoHHN](https://github.com/Zhang-xiaokun/CoHHN)

~~~~
~$ cd preprocess
preprocess$ python3 preprocess.py
preprocess$ python3 cohhgn_plus_build_graph.py
~~~~


Train and evaluate the model:
~~~~
~$ cd CoHHGN_plus
CoHHGN_plus$ python3 cohhgn_plus_main.py --dataset datasetname
~~~~

## Citation
Please cite our paper if you use our codes. Thanks!
```
@InProceedings{CoHHGN_plus,
author="Sumiya, Yuta
and Numata, Ryusei
and Takahashi, Satoshi",
editor="Bifet, Albert
and Lorena, Ana Carolina
and Ribeiro, Rita P.
and Gama, Jo{\~a}o
and Abreu, Pedro H.",
title="Pseudo Session-Based Recommendation with Hierarchical Embedding and Session Attributes",
booktitle="Discovery Science",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="582--596",
isbn="978-3-031-45275-8"
}
```

In case that you have any difficulty about the implementation or you are interested in our work,  please feel free to communicate with us by:

Author: Yuta Sumiya (sumiya@uec.ac.jp / diddy2983@gmail.com)

Also, welcome to visit my academic homepage: https://sumugit.github.io
