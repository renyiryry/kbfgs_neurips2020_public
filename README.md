# kbfgs_neurips2020_v2

This is the code for the paper [Practical Quasi-Newton Methods for Training Deep Neural Networks][kbfgs-paper].

[kbfgs-paper]: https://arxiv.org/abs/2006.08877

## Specification of dependencies

Python 3.7.6;
GCC 7.3.0;
Cuda compilation tools, release 10.1, V10.1.243

torch 1.4.0,
numpy 1.18.1,
scipy 1.4.1,
pytz 2019.3,
mat4py 0.4.2,
psutil 5.7.0


## How to get results

See Demo.ipynb for results and command to produce the results.

For the function train_model(), set the argument "home_path" as the directory containing Demo.ipynb. 

### To tune the hyper-paramters:

For K-BFGS, K-BFGS(L), KFAC, change the arguments "lr" and "lambda_damping" in train_model();
For Adam, RMSprop, change the arguments "lr" and "RMSprop_epsilon";
For SGD-momentum, change the arguments "lr".

