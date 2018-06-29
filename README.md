# Rotational Unit of Memory (RUM) - Tensorflow

Here we present the implementation of a novel RNN model--RUM--which outperforms well-established RNN models in a variety of sequential tasks. The Rotational Unit combines the memorization advantage of unitary/orthogonal matrices with the dynamic structure of associative memory.
    
If you find this work useful, please cite [arXiv:1710.09537] (https://arxiv.org/pdf/1710.09537.pdf). The model and tasks are described in the same paper. 

## Model: `RUM.py`
- Here we implement the operation `Rotation` and the `RUM` model. If you want to use the efficient implementation O(N_b * N_h) of `Rotation` then import `rotate`. If you want to produce a rotation matrix between two vectors, import `rotation_operator`. A simple script to test the Rotation functionality is in `rotation_test.py`. 
- If you want to use the `lambda = 0 RUM` then import `RUMCell`. Likewise, import `ARUMCell` for the `lambda = 1 RUM` model.
- On top of the model we use regularization techniques via `auxiliary.py` (modified from [1]). 
## Copying Memory Task: `copying_task.py` 
Please inspect the arguments in the code in order to test your own hyper-parameters. For example, if you want to run `lambda = 1 RUM` with time normalization `eta = 2.0` on this task, enter `python copying_task.py ARUM -norm 2.0`. 
## Associative Recall Task: `recall_task.py`
Please inspect the arguments in the code in order to test your own hyper-parameters. For example, if you want to run `lambda = 1 RUM` with time normalization `eta = 0.1` on this task, enter `python recall_task.py ARUM -norm 0.1`. 
## bAbI Question Answering Task: `babi_task.py` 
Please inspect the arguments in the code in order to test your own hyper-parameters. For example, if you want to run `lambda = 0 RUM` without time normalization on subtask number 5, enter `python babi_task.py RUM 5`. 
## Penn Treebank Character-level Language modeling Task: `ptb_task.py`
Please inspect the arguments in the code and the models in `ptb_configs.py` in order to conduct your own grid search. For example, if you want to run the model `FS-RUM-2`, which achieves 1.189 BPC, enter `python ptb_task.py --model ptb_fs_rum`. The code is adapted from [1], from where we also use a layer-normalized LSTM `LNLSTM.py` and the FSRNN higher-level model `FSRNN.py`.
## License
This project is licensed under the terms of the MIT license.
## References 
[1] https://github.com/amujika/Fast-Slow-LSTM
