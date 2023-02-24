# TV-SBL
This contains the codes to implement the TV-SBL algorithm. The repository contains the codes to both implmentations using the (1) cvx toolbox and (2) EM algorithm.


## 1) TV-SBL - Convex Optimization
The TV-SBL penalty introduced here is formulated as a convex optimization problem. The cvx toolkit is used to solve this optimization. 

This is based on the work in the following:

Sant, A., Leinonen, M. and Rao, B.D., 2021, June. General total variation regularized sparse Bayesian learning for robust block-sparse signal recovery. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5604-5608). IEEE.


## 2) TV-SBL - General EM Algorithm
The TV-SBL penalty introduced here is immplemented using the general EM algorithm. The specific strategy involves an alternating optimization updating the even and odd indices.

This is based on the work in the following:

Sant, A., Leinonen, M. and Rao, B.D., 2022. Block-Sparse Signal Recovery via General Total Variation Regularized Sparse Bayesian Learning. IEEE Transactions on Signal Processing, 70, pp.1056-1071.
