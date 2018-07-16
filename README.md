# EC-Net: an Edge-aware Point set Consolidation Network [Paper](https://yulequan.github.io/papers/ECCV18_EC-Net.pdf)
by [Lequan Yu](https://yulequan.github.io/), Xianzhi Li, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for our ECCV 2018 paper '[EC-Net: an Edge-aware Point set Consolidation Network](https://yulequan.github.io/ec-net/index.html)'. This project is based on our previous project [PU-Net](https://github.com/yulequan/PU-Net). 

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.3 (higher version should also work) and Python 2.7 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. You also need to remove `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly if necessary.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

We adopt the Dijkstra algorithm implemtned in python-graph library, you can follow the instruction in [here](https://github.com/wting/python-graph) to install it. 

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/yulequan/EC-Net.git
   cd EC-Net
   ```
2. Compile the TF operators
   Follow the above information to compile the TF operators. 
   
3. Train the model:
 
   ```shell
   cd code
   python main.py --phase train --gpu 0
   ```

4. Evaluate the model:
    We provide the pretrained model in folder 'model/pretrain'.
    To evaluate the model, you need to put the test point cloud file (in .xyz format) in folder 'eval_input'.
    Then run:
   ```shell
   cd code
   python main.py --phase test --log_dir ../model/pretrain
   ```
   You will see the input point cloud, output point cloud, and the identified edge points in the folder `eval_result`.
   

## Citation

If EC-Net is useful for your research, please consider citing:

    @inproceedings{yu2018ec,
         title={EC-Net: an Edge-aware Point set Consolidation Network},
         author={Yu, Lequan and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
         booktitle = {ECCV},
         year = {2018}
   }
## Related project
1. [PU-Net](https://github.com/yulequan/PU-Net)
2. [PointNet++](https://github.com/charlesq34/pointnet2)

### Questions

Please contact 'lqyu@cse.cuhk.edu.hk'
