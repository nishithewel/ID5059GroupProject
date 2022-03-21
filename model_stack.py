# At the moment this is taken from
# https://medium.com/geekculture/how-to-use-model-stacking-to-improve-machine-learning-predictions-d113278612d4#:~:text=Model%20Stacking%20is%20a%20way,model%20called%20a%20meta%2Dlearner.

# This example provides code that stacks some user-defined models (catboost,
# xgboost, some NNs). Need to alter so that it can take arbitrary models to
# stack. 

# First import necessary libraries
import pandas as pd
from sklearn.ensemble import StackingRegressor

# Decision trees
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Add, Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import concatenate
from tensorflow.keras import regularizers
from keras.regularizers import l1
from keras.regularizers import l2

# Wrapper to make neural network compitable with StackingRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Linear model as meta-learn
from sklearn.linear_model import LinearRegression

# Create generic dataset for regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create dummy regression dataset
X, y = make_regression(n_targets=1, random_state=42)

# Convert to pandas
X = pd.DataFrame(X)
y = pd.DataFrame(y)

#Rename column
y = y.rename(columns={0: 'target'})

# Split into validation set
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)

#Peak at our dummy data
X_train.head()

y_train.head()

'''
 A neural network to show you how to incorporate models that may not be compatable with the stacking model
 (You don't need to understand this code for the purposes of the tutorial, just need to know that certain models can crash the stacking regressor without the appropriate wrapper around it)
'''
def create_neural_network(input_shape, depth=5, batch_mod=2, num_neurons=20, drop_rate=0.1, learn_rate=.01,
                      r1_weight=0.02,
                      r2_weight=0.02):
    '''A neural network architecture built using keras functional API'''
    act_reg = l1(r2_weight)
    kern_reg = l1(r1_weight)
    
    inputs = Input(shape=(input_shape,))
    batch1 = BatchNormalization()(inputs)
    hidden1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(batch1)
    dropout1 = Dropout(drop_rate)(hidden1)
    hidden2 = Dense(int(num_neurons/2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(dropout1)
    
    skip_list = [batch1]
    last_layer_in_loop = hidden2
    
    for i in range(depth):
        added_layer = concatenate(skip_list + [last_layer_in_loop])
        skip_list.append(added_layer)
        b1 = None
        #Apply batch only on every i % N layers
        if i % batch_mod == 2:
            b1 = BatchNormalization()(added_layer)
        else:
            b1 = added_layer
        
        h1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(b1)
        d1 = Dropout(drop_rate)(h1)
        h2 = Dense(int(num_neurons/2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(d1)
        d2 = Dropout(drop_rate)(h2)
        h3 =  Dense(int(num_neurons/2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(d2)
        d3 = Dropout(drop_rate)(h3)
        h4 =  Dense(int(num_neurons/2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(d3)
        last_layer_in_loop = h4
        c1 = concatenate(skip_list + [last_layer_in_loop])
        output = Dense(1, activation='sigmoid')(c1)
    
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam()
    optimizer.learning_rate = learn_rate
    
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    return model

def get_stacking(input_shape=None):
    '''A stacking model that consists of CatBoostRegressor,
    XGBRegressor, a linear model, and some neural networks'''
    # First we create a list called "level0", which consists of our base models"
    # Basically, you'll want to pick a assortment of your favorite machine learning models
    # These models will get passed down to the meta-learner later
    level0 = list()
    level0.append(('cat', CatBoostRegressor(verbose=False)))
    level0.append(('cat2', CatBoostRegressor(verbose=False, learning_rate=.0001)))
    level0.append(('xgb', XGBRegressor()))
    level0.append(('xgb2', XGBRegressor(max_depth=5, learning_rate=.0001)))
    level0.append(('linear', LinearRegression()))
    #Create 5 neural networks using our function above
    for i in range(5):
        # Wrap our neural network in a Keras Regressor to make it
        #compatible with StackingRegressor
        keras_reg = KerasRegressor(
                create_neural_network, # Pass in function
                input_shape=input_shape, # Pass in the dimensions to above function
                epochs=6,
                batch_size=32,
                verbose=False)
        keras_reg._estimator_type = "regressor"
        # Append to our list
        level0.append(('nn_{num}'.format(num=i), keras_reg))
    # The "meta-learner" designated as the level1 model
    # In my experience Linear Regression performs best
    # but feel free to experiment with other models
    level1 = LinearRegression()
    # Create the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=2, verbose=1)
    return model

#Get our input dimensions for neural network
input_dimensions = len(X_train.columns)
# Create stacking model
model = get_stacking(input_dimensions)
model.fit(X_train, y_train.values.ravel())
# Creating a temporary dataframe so we can see how each of our models performed
temp = pd.DataFrame(y_val)
# The stacked models predictions, which should perform the best
temp['stacking_prediction'] = model.predict(X_val)
# Get each model in the stacked model to see how they individually perform
for m in model.named_estimators_:
        temp[m] = model.named_estimators_[m].predict(X_val)

# See how each of our models correlate with our target
# In most instances of running the program the stacked predictions should outperform any singular model
print("Correlations with target column")
print(temp.corr()['target'])

# See what our meta-learner is thinking (the linear regression)
print("Coeffecients of each specific model")
for coef in zip(model.named_estimators_, model.final_estimator_.coef_):
    print(coef)