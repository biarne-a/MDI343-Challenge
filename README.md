1 Introduction

The goal of this project is to build a model that combines the scores of different algorithms aiming to classify pairs of images that represent or not the same person. We have been given 14 different algorithm scores that must be combined with respect to a time constraint. Each algorithm takes a certain amount of time to be executed. Our time budget goes up to 600 ms. Given the different execution times, one can easily see that the combination can only take up to 4 algorithms. The combination of the algorithms is to be done using a fusion matrix
enabling to calculate the fusion scores in a vector calculation. The fusion scores are then used to calculate the FRR (False Recognition Rate) at a
given FAR (False Acceptance Rate). The overall objective is to build a fusion matrix that will give us the lowest FRR with respect to a FAR of 0.01%.

2 Data preparation
A quick view of the common statistics on the data indicates half the algorithm scores contain some infinite values which could cause issues for the rest of the process. The simplest strategy to get rid of the problem was to remove the samples containing at least one infinite value. Using such a strategy the training dataset went from 2048853 samples to 2048840.

3 Dimension reduction
Given the fact that we can only use a subset of the different scores, one can see this problem as a dimension reduction problem where the features
are the algorithm scores. What we want to do is to select the best features to combine in regards to the goal of minimizing the FRR. One of the first thing to do is to study the correlation between the different features. For that we computed the following correlation matrix.
It is quite obvious that the algorithms that have similar computation time which go in continuous pairs ((0,1), (2, 3), ..., (10, 11), (12, 13)) are very correlated. Our goal is to combine the performance of the different algorithms in order to maximize the overall performance. Therefore it seems quite intuitive to try to combine the algorithms that are the least correlated.

An approach to feature selection is to use a linear model penalized with the l1 norm. Given the fact that we are trying to predict a binary output, the penalized logistic regression is well suited for the task. What was done is trying increasing values of the regularization hyper-parameter and seeing which coefficient gets away from 0 first. This gives us the regularization path.

Using this approach we determine that the best features to use was the combination 6, 8, 10 and 12. But it is known that the lasso method does not perform optimally when features are correlated. Indeed it has difficulties choosing between correlated features. Given the fact the features are correlated in pairs, it seamed quite a good idea to include all the combinations of the following pairs : ((6, 7), (8, 9), (10, 11), (12, 13)).

4 Model
Once a subset of algorithm combinations was chosen for further analysis it was time to find a method to find the coefficients to use in the fusion. For that task I used a penalized logistic regression with the l2 norm. The penalization hyper-parameter was chosen with cross-validation. Once the FRRs were computed for each combination, the polynomial features have been included to try to improve the goodness of the fit. The FRR at 0.001% FAR has been significantly reduced using the polynomial features. One could also see the coefficients for these new features were an order of magnitude smaller (10 −7) than the other coefficients (10 −3). Some other important facts were to include the intercept at coordinate (0,0) in the fusion matrix and to print the coefficients up to 20 digits after the floating point.

5 Results
Finally the combination that gave the minimal FRR at 0.001% FAR was the combination 6, 8, 11, 12 with a FRR of 0.0808529803399 on the train set and 0.0652591170825 on the test set.