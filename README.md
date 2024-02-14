# ICR-Identifying-Age-Related-Conditions-ML-Competition
A data science project involving data exploration, data engineering, and machine learning engineering to detect diseases in patients using machine learning. 


Here you can find more information about the competition:
https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview


During the competition I tackled the disease detection using different approaches.
I'll be adding the most relevant ones in this repository.

## First approach : Deep learning from pipeline parameters
In this approach I created deep learning pipelines.

To make experimentation faster, and to ensure the results could be recreated, and tracked, each pipeline was created from parameters with a single method.
This method called all relevant auxiliary methods, passing the parameters their required, building both the data processing pipeline, and the machine learning model.

This approach allowed extensibility without breaking previous pipelines, or duplicating any code: Any time an additional way of processing data or creating a model was required, the necessary auxiliary methods were simply added or expanded upon, and it was a simple matter of calling the same method with the right parameters to use the new features.

While this approach was interesting and a great learning experience, during the competition it became apparent that perceptron neural networks didn't perform aswell as decision trees to solve this problem which was a tabular data analysis problem, so I moved on to a different