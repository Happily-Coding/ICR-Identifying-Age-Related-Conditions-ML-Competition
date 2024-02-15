## First approach: Deep learning from pipeline parameters
In this approach I created deep learning pipelines.

To make experimentation faster, and to ensure the results could be recreated, and tracked, each pipeline was created from parameters with a single method.
This method called all relevant auxiliary methods, passing the parameters they required, building both the data processing pipeline, and the machine learning model.

This approach allowed extensibility without breaking previous pipelines, or duplicating any code: Any time an additional way of processing data or creating a model was required, the necessary auxiliary methods were simply added or expanded upon, and it was a simple matter of calling the same method with the right parameters to use the new features.

While this approach was interesting and a great learning experience, during the competition it became apparent that perceptron neural networks didn't perform as well as decision trees to solve this problem which was a tabular data analysis problem, so I moved on to a different


## Second approach: Best Scoring approach: Hyperparameter tuned gradient boosted decision trees
In this approach I used XgBoost to create gradient boosted decision trees, Optuna for hyperparameter tuning and stratified k-fold-cross-validation

The datasets used for the competition were small, since they were of hard to collect health-related data, but at the same time they werent properly split, so the test dataset provided during the competition wasnt representative of the test dataset used to determine the standings at the end.

This approach was my best scoring approach on the cross-validated training set, and the test set that decided the final competition standings. But it did not score very well on the test set available during the course of the competition, which led me to keep looking for different approaches.

