- Try increasing the weights of the failed points, retraining, and ensambling based on confidence( the value closest to 0 or 1 is the one that counts)
- Try increasing the weights of different splits, and making an ensamble based on the original split
- Try to use bagging classifier. At least to give importance to columns.
    - Bagging source sklearn: https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/ensemble/bagging.py#L44
    - bagging classifier stratified k fold. https://www.kaggle.com/code/arungupta84/end-to-end-data-science-project-with-adult-income
- Identify hardest points by parsimony and compare them with hardest points to predict by model
- Add EJ to stratified split
- Usar sample weights for training xgboost and lgb
- Try one of the libraries i'd found, or convolutional neural network from tables as seen in the exampel of a previous competition.
- try using additional features: https://www.kaggle.com/code/gzguevara/brute-force-features-cv-25-20
- try improving using this that has similar mechanics: https://www.kaggle.com/code/hd00000/tabfpn-xgb-oversampling-cv-0-18-lb-0-16

