# Disclaimer
Please note that this is a collection of notes mostly for my own reference.


They were taken during various stages of the competition, which was my first real world ML engineering experience, they were compiled from various files of various stages of learning, and haven't been properly organized, so they are messy, and prone to containing errors or outdated thoughts on different topics. Take them with a grain of salt.


Additionally parts are in spanish and parts are in english since they were written in different contexts. The titles are in english and were written during the merging of the notes in a single file, to make the file a bit more usable.

# Steps to solve a data science problem.
0. Create a repository
1. Create an exploratory data analysis notebook.
2. Understand the problem you are trying to solve. Document it in the EDA notebook:
    - Write the problem that you think needs solving.
    - Problems are solved to achieve a goal. Identify and write that goal.
    - Write alternative ways to achieve that goal.
    - Problems do not exist in a void. Identify elements upstream and downstream of the problem.
        - Problems can often be solved upstream or downstream(preventing the problem from needing to be solved)
    - Considering the goal the other entities, and the other possibilities, make sure that the best way to achieve your goal is to solve your problem.
    - Break the problem into parts and write the general steps needed to solve the problem.
3. Find data suitable to solve your problem.
4. Create a data preparation python file.
5. Create the methods to load the data in the data preparation file.
6. Load the data with those methods, explore the data in the EDA notebook.
    - Keep an insights cell at the begining of the notebook summaryzing your findings.
    - Look at each variable and document what needs to be done. As a general rule, you might want to:
        - Identify numerical variables.
        - Identify ordinal categorical variables.
        - Identify not-ordinal categorical variables.
        - Identify date variables.
        - Identify target variables.
        - Identify image and free text variables or other complex data sources.
        - Identify all categorical variables requiring additional processing, and what that processing is.
            - Identify categorical variables that contain multiple info in a single value, and how they should be split into columns.
        - Identify how to merge data sources
        - Plot a histogram of value - frequency value for all numerical variables
        - Analyze correlation with other variables? (value vs other variable valuehistogram?)
7. Create a TODO and ideas notebook. Document all ideas for data exploration, preparation and ml.
    - The file should first contain the todo list, in order of what should be done first, with the person that needs to do it.
        - If multiple people are working on the project, whenever someone begins to work on a todo item they should note on the todo item that they are working on it.
        - The todo list might to list the reason for each todo.
    - The file should afterwards contain ideas
        - First there should be a section per ml pipeline part.
        - Then there should be numerated idea subsections (so they can be easily referenced)
        - Each idea as a header cell + a body cell, so they can easily be collapsed but detail can be added.
8. Create the methods to create a ml pipeline from parameters in the python file.
    - create cross validation train and test sets. For each cross_validation set process the data
    - You'll need to allow the data pipeline to be fit with training data, and then be returned to transform the training, validation and prediction data.  You might wanna save the ideal epochs for each model, as well as their performance with their test set and then average their statistics and the epoch to use for training.
    - For data preparation you migth want to :            
    - Process numerical variables
        - Identify and handle outliers.
        - Normalize the variables (bring them to a -1 or 0 to 1 range)
        - Identify its distribution, and create a value frequency histogram
        - Transform its distribution so its normal if reasonable
        - Impute missing values KNN (+ masking?)
    - Process any categorical variables containing multiple informations together.
        - Split them into two variables.
    - Process ordinal categorical variables
        - Vectorize them
        - Impute missing values
        - Add a column for present - missing values (except you use masking)
    - Process non-ordinal categorical variables
        - One hot encode them. Make sure to have an encoding for unknown values, you could have one for missing values, but its probably pointless since you can have 0 for all as missing.
    - Process text, images or other complex data sources.
        - Ignore them or create a machine learning pipeline for their processing, and use the results as input for the main model.
    - all variables that cannot be used in a standard manner should probably be passed as pipeline elements to be added to the pipeline.
9. Create and test suitable pipelines. Probably one per notebook, so the results of each run are recorded.
10. Hyperparameter tune the pipeline
    - for each hyperparameter combination:
        - for each cross validation tuple, create the model , prepare the data, save the score as described above.
    - The model parameters be saved along with the results during tuning.
11. Create boosts or at least bags for each non boosted model, and ensembles, specially of models that learn differently. Might want to consider the level of correlation of the results.
12. Prepare the final model:
    - Consider the score of the model the average score of the validation of each fold.
    - Prepare all the data and train the model with all the data, to have the best imputation normalization and model for predictions.
    - Save the model file, and the data pipeline if its even possible.
13. Use the final model:
    - Load the model and pipelien from file if possible. Prepare the data to predict and predict using the models.
    - If its impossible to load the pipeline it really shouldnt be the problem as long as you save the pipeline weights and can load them.
    - Might want to make sure the model can work with dirty data by creating a dataset with dirty data and testing it.


## This is the criteria that should be used to prepare each variable
- ¿Tiene orden?
  - Si:
      - Hay que volverla numerica si todavia no lo es.
      - Hay que imputear missing values (KNN imputing | iterative imputing) (idealmente manteniendo el tipo numerico(creo)) (ej imputear enteros en variables enteras)
      - Hay que normalizarla (incluso variables discretas (creo))
      - La normalización y si imputear o no probablemente deberia ser un parametro de la pipeline ya que hay modelos que los pueden manejar (no nn)
  - No:
      - Hay que onehot encodearla.
      - Los missing & unknown values se tienen que volver una categoria. o ser maskeados. 
      - Si maskear deberia ser un parametro de la pipeline, ya que algunos tal vez lo pueden manejar. Creo que en todos los modelos onehot encodear no viene mal.
- Las variables binarias se condiseran sin orden.

## This is why you cant drop a column when one hot encoding if you want to be able to handle unknowns and missing values:
You cannot place 4 states in solo 2 values
#EsA? EsB -> No se puede dropear una columna si quiero poder handlear unknowns y missings.
#SI   NO -> Es A
#NO   SI -> es B
#NO   NO -> es desconocido.
#SI   SI -> es missing.


## About one hot encoding with scikitlearn
- Example ('one_hot_encode_categories', OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype='int'), ['EJ'])
- handle_unknown = ignore is incompatible with drop first, because it will assign unknown values to all zeroes, which is incorrect
- sparse output is not supported with pandas
- drop_first reduces one the number of elements in the array, since the last element would be duplicated info.
- dtype int makes the encodings be arrays of integers
- Por ahi puedo usar un mejor one hot encoder si le paso labels, o primero hago que la columna tenga vector encoding con unique + n+2 (para misisng y unknown values)
- Me parece que unknown values se pueden manejar de manera razonalbe con el por defecto sin usar drop first, y que misisng se pueden manejar con masking. y eso seria la mejor esrategia

#MAYBE COULD BE USED IF A column WITH MISSING & 'OTHER' VALUES is added to the fitting data?
#One hot encodes using the minimal number of columns (ie binary variables will use a single column). Does not handle missing values properly since they'll be assigned to an existing category (all 0)
#Will error on unknown categories since they'd need to be assigned to a leftover space which doesn't exist either.
#Should probably only be used if you hadled misisng values prior to this (filteirng or creating a mask)
#('one_hot_encode_categories_without_missing_or_unknowns_in_submissions', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='error', dtype='int'), ['EJ']),

#One hot encodes using one column per category. This while first masking missing values could be perfect. Though imputation may have use for knowing the missing values. (could probably use them as is?)
