import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

"""resample = used for downsampling the classes required for classification as different classes might have different datapoints

train_test_split = Splits a dataset into training and testing subsets.

This is crucial for evaluating the performance of a machine learning model on unseen data.

Prevents overfitting by ensuring the model is tested on data it hasn’t seen during training.

GridSearchCV = Performs hyperparameter tuning by exhaustively searching over a specified parameter grid.

Uses cross-validation to evaluate the model's performance for each combination of hyperparameters.

cross_val_score = Evaluates a model using cross-validation.

Splits the data into k folds, trains the model on k-1 folds, and validates it on the remaining fold. This process is repeated k times.
"""

data = pd.read_csv("E:\Python\Cloudburst Prediction System\Rainfall (1).csv")

print(type(data))

data.shape

data.head()

data.tail()

data.info()

#we have two missing values in last 2 columns and also a whitespace
data.columns = data.columns.str.strip()
data.columns

data.info()

#day is a temporal column so removing it

data = data.drop(columns = ['day'])
data.head()

data.isnull().sum()

data['winddirection'].fillna(data['winddirection'].mean(), inplace = True)

data['windspeed'].fillna(data['windspeed'].mean(), inplace = True)

data.isnull().sum()

data.duplicated()

data['rainfall'].unique()

data['rainfall'] = data['rainfall'].map({"yes":1, "no":0})

data

"""**Exploratory Data Analysis(EDA)**"""

#setting plot style for all plots
sns.set(style="whitegrid")

data.columns

plt.figure(figsize=(20,10))

for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine','windspeed'],1):
  plt.subplot(3,3,i)
  sns.histplot(data[column], kde = True)
  plt.title(f"Distribution of {column}")


plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()

"""As 1 has 250 datapoints and 0 has around 120 datapoints, so we need to downsample the column(dataPoints) to properly train the model."""

#observe correlation matrix by heatmaps
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

"""highly correlated columns lead to high colinearity which means those columns contribute the same thing to the target variable which should be avoided"""

#identifying outliers using boxplot
plt.figure(figsize=(15,10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine','windspeed'],1):
  plt.subplot(3,3,i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()

"""hollow circles outside the thin horizontal lines represents outliers, as they are few in numbers so no action

**DATA PREPROCESSING**
"""

#dropping highly correlated columns
data = data.drop(columns = ['maxtemp', 'temparature', 'mintemp'])

data.head()

#downsampling
#separate majority and minority classes
print(data['rainfall'].value_counts())

#so majority is 1 and minority is 0

df_majority = data[data['rainfall'] == 1]
df_minority = data[data['rainfall'] == 0]

print(df_majority.shape)
print(df_minority.shape)

#downsample majority class to minority class
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority),random_state=42)

"""replace=false makes sure to remove duplicated rows while resampling"""

df_majority_downsampled.shape

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled.shape

df_downsampled.head()

#shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

df_downsampled.head()

df_downsampled["rainfall"].value_counts()

#splitting data into training data and test data
#split features and target as x and y
x = df_downsampled.drop(columns=['rainfall'])
y = df_downsampled['rainfall']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""test_size=0.2 means 20% is testing data and remaining 80% is training data"""

rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50,100,200],
    'max_features': ["sqrt","log2"],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(x_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("best parameters for Random Forest:", grid_search_rf.best_params_)

"""**Model Evaluation**"""

cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=5)
print("Cross validation sscores:", cv_scores)
print("Mean cross validation score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(x_test)
print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix", confusion_matrix(y_test, y_pred))
print("Test set Classification Report", classification_report(y_test, y_pred))

"""y_test = testing part to check tuning of model
Model accuracy = 74% (not that good, changes in parameters values will increase it)

**Prediction on unknown data**
"""

input = (1010.4, 20.1, 88, 79, 0.1, 78.9, 48.2)

input_df = pd.DataFrame([input], columns = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])

print(input_df)

prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")

input = (1166.2, 99.2, 53, 11, 24.1, 191, 11.8)
input_df = pd.DataFrame([input], columns = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])

prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")

#using pickle to save model
model_data = {"model":best_rf_model, "features_names":x.columns.tolist()}

with open("rainfall_prediction_model.pkl", "wb") as file:
  pickle.dump(model_data, file)

"""**Load pickle model and use for prediction**"""

import pickle
import pandas as pd

with open("rainfall_prediction_model.pkl","rb") as file:
  model_data = pickle.load(file)

best_model = model_data["model"]
features_names = model_data["features_names"]

input =(1134.2,88,56,88.45,23,212.4,46)

input_df = pd.DataFrame([input], columns=features_names)

prediction = best_model.predict(input_df)
print("Prediction results: ", "Rainfall" if prediction[0] == 1 else "No Rainfall")


