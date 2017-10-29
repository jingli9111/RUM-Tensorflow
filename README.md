# Rotational Unit of Memory (RUM)-Tensorflow

Here we present the implementation of a novel RNN model--RUM--which outperforms significantly the current frontier of models in a variety of sequential tasks. The Rotational Unit combines the memorization advantage of unitary/orthogonal matrices with the dynamic structure of associative memory.
    
If you find this work useful, please cite [arXiv:1710.09537] (https://arxiv.org/pdf/1710.09537.pdf). The model and tasks are described in the same paper. 

## RUM.py
- Here we implement the operation `Rotation` and the `RUM` model. If you want to use the efficient implementation O(N_b * N_h) of `Rotation` then import `rotate`. If you want to produce a rotation matrix between two vectors, import `rotation_operator`.
- If you want to use the `lambda = 0 RUM` then import `RUMCell`. Likewise, import `ARUMCell` for the `lambda = 1 RUM` model.
- On top of the model we use regularization techniques via `auxiliary.py` (modified from [1]). 
## copying_task.py 
Please inspect the arguments in the code in order to conduct your own grid search. For example, if you want to run `lambda = 1 RUM` with time normalization `eta = 2.0` on this task, enter `python copying_task ARUM -norm 2.0`. 


[1] https://github.com/amujika/Fast-Slow-LSTM
