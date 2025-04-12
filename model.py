import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


data = pd.read_csv("E:\Python\Cloudburst Prediction System\Rainfall (1).csv")

print(type(data))

data.shape

data.head()

data.tail()

data.info()

data.columns = data.columns.str.strip()
data.columns

data.info()



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

#Exploratory Data Analysis(EDA)

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

#HEATMAP FOR DETECTING CORRELATIONS

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

#BOXPLOT 

plt.figure(figsize=(15,10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine','windspeed'],1):
  plt.subplot(3,3,i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()


#DATA PREPROCESSING

data = data.drop(columns = ['maxtemp', 'temparature', 'mintemp'])

data.head()

print(data['rainfall'].value_counts())


df_majority = data[data['rainfall'] == 1]
df_minority = data[data['rainfall'] == 0]

print(df_majority.shape)
print(df_minority.shape)

#Downsampling

df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority),random_state=42)

df_majority_downsampled.shape

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled.shape

df_downsampled.head()


df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

df_downsampled.head()

df_downsampled["rainfall"].value_counts()

x = df_downsampled.drop(columns=['rainfall'])
y = df_downsampled['rainfall']


#MODEL TRAINING
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



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

#Model Evaluation

cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=5)
print("Cross validation scores:", cv_scores)
print("Mean cross validation score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(x_test)
print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix", confusion_matrix(y_test, y_pred))
print("Test set Classification Report", classification_report(y_test, y_pred))


# Calculate ROC-AUC score
y_pred_proba = best_rf_model.predict_proba(x_test)[:, 1]  
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("Test set ROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest Classifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


#Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
print("Average Precision Score:", avg_precision)

#Plotting Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Random Forest Classifier")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


#Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    best_rf_model, x_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Compute mean and std for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curves for Random Forest Classifier")
plt.legend(loc="best")
plt.grid(True)
plt.show()


#Prediction on unknown data
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


