# Disclaimer
Please note that this is a collection of notes mostly for my own reference.


They were taken during various stages of the competition, which was my first real world ML engineering experience, they were compiled from various files of various stages of learning, and haven't been properly organized, so they are messy, and prone to containing errors or outdated thoughts on different topics. Take them with a grain of salt.


Additionally parts are in spanish and parts are in english since they were written in different contexts. The titles are in english and were written during the merging of the notes in a single file, to make the file a bit more usable.


# Steps to create a robust ml model.
0. Hacer un EDA de la data, y determinar que hay que hacerle a cada variable.
1. Leer la data
2. Crear cross validation train y test sets.
3. Decidir los siguientes hiperparametros para probar (que podrian incluir data preprocessing hyperparameters)

3. Por cada cross validation set tuple:
    - Preparar la data de cada cross validation set:
        - Crear transformador que:
            - Manejar Variables categoricas (one hot encoding).
            - Aplicar transformaciones necesarias para que la data tenga distribución normal.
            - Eliminar outliers?
            - Normalizar la data
            - Imputear los missing values numericos. KNN (+ masking?)
        - fit-transformar train set
        - transformar test set
    - Crear y entrenar un modelo, con el train transformado y de tener epocas guardar el numero de epocas ideales.
    - Calcular su performance con el test transformado.
4. Promediar las estadisticas de validación y determinar la epoca ideal para el modelo entrenado con todo de ser necesario.
5. Guardar esos dos datos

5. Preparar el dataset entero:
   - , posiblemente incluso con las submissions para poder imputear y normalizar y manejar missing values mejor.
5. Entrenar el modelo 
6. Con el mejor modelo:
   - Verificar que funcione con dirty data
   - Aplicar
   
- TODO Meter hyper parameter tuning en esta lista.


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
