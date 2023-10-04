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
@inproceedings{CoHHGN_plus,
  author    = {Yuta Sumiya and
               Ryusei Numata and
               Satoshi Takahashi and
               },           
  title     = {Pseudo Session-based Recommendation with Hierarchical Embedding and Session Attributes},         
  booktitle = {{DS} '23: 26th International Conference on Discovery Science, Porto, Portugal, October 9-11, 2023},          
  pages     = {},
  year      = {2023},
  crossref  = {DBLP:conf/DS/2023},
  url       = {},
  doi       = {},
  biburl    = {}
}
```

In case that you have any difficulty about the implementation or you are interested in our work,  please feel free to communicate with us by:

Author: Yuta Sumiya (sumiya@uec.ac.jp / diddy2983@gmail.com)

Also, welcome to visit my academic homepage: https://yusumi.github.io

