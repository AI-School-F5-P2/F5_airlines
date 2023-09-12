# Librerías básicas
import pandas as pd
import numpy as np
#from tabulate import tabulate
import math

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid",font_scale=1, palette="pastel")

#Libreria para separacion de datos train y test
import pickle
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, GridSearchCV
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer, PowerTransformer, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, r2_score,  make_scorer, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer, PowerTransformer
from sklearn.compose import make_column_transformer, ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_hist_gradient_boosting

# Flag que indica, cuando es True, que es la última vez que se entrena al modelo definitivo y se puede guardar
save_pickle = False

def model_pred(model, X, y, flag):  
    try:
        # Define a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', MinMaxScaler())]), num_cols),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])

        # Create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        # Define the metrics for evaluation
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted')
        }

        # Perform 5-fold cross-validation
        cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, return_train_score=True)

        # Calculate overfitting as the difference between training and validation accuracy
        overfitting = (np.mean(cv_results['train_accuracy']) - np.mean(cv_results['test_accuracy'])) * 100

        y_pred = cross_val_predict(pipeline, X, y)

        # Store the evaluation results
        results = {
            'fit_time': np.mean(cv_results['fit_time']),
            'accuracy': np.mean(cv_results['test_accuracy']),
            'f1': np.mean(cv_results['test_f1']),
            'recall': np.mean(cv_results['test_recall']),
            'precision': np.mean(cv_results['test_precision']),
            'overfitting': overfitting,
            'cm': confusion_matrix(y, y_pred),
        }

        # Train the model on the entire dataset
        pipeline.fit(X, y)
        
        print("Pipeline fitted. Checking for feature importances.")
        
        # Code to display feature importances
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            print("Model has feature importances. Displaying them.")
            importances = pipeline.named_steps['model'].feature_importances_
            num_features = num_cols
            cat_features = (pipeline.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(input_features=cat_cols))
            all_features = np.concatenate([num_features, cat_features])
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:10]
            
            print("Ranking of most important features:")
            for f in range(top_indices.shape[0]):
                print(f"{f + 1}. Feature {all_features[top_indices[f]]} (Importance: {importances[top_indices[f]]})")
        
        if flag:
            # Save the pipeline using Pickle
            with open('data_pipeline.pkl', 'wb') as file:
                pickle.dump(pipeline, file)
                
        return results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"X columns: {X.columns}")
        print(f"y head: {y.head()}")
        print(f"num_cols: {num_cols}")
        print(f"cat_cols: {cat_cols}")


#Lectura del dataset
df = pd.read_csv("airline_passenger_satisfaction.csv")

# Para llenar los valores nulos con la media: (MIRAR SI SE IMPUTAN CON MEDIA O MEDIANA)
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

# Para convertir una columna de tipo float a int 
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].astype(int)

columna_a_borrar = ['Unnamed: 0', "id"]
df = df.drop(columna_a_borrar, axis=1)

num_cols = ["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
                 "Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service",
                 "Leg room service","Baggage handling","Checkin service","Inflight service","Cleanliness","Departure Delay in Minutes",
                 "Arrival Delay in Minutes"]

# Cambiar solo 'disloyal Customer' a 'Disloyal Customer'. Acordarme de ponerlo en el pipeline
df.loc[df['Customer Type'] == 'disloyal Customer', 'Customer Type'] = 'Disloyal Customer'

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

cat_cols

mean = df["Flight Distance"].mean()
std = df["Flight Distance"].std()

threshold = 3 * std

outliers = df[abs(df["Flight Distance"] - mean) > threshold]

df = df.drop(index=outliers.index)
modelos = [
    
    AdaBoostClassifier()
]

# Separar Variable Objetivo, target o variable dependiente de las variables independientes
df_f = pd.DataFrame(df)
y = df_f["satisfaction"]
X = df_f.drop(columns="satisfaction")
# Se crea un diccionario para almacenar los resultados de cada modelo
resultados_por_modelo = {}

# Itera sobre los modelos
for modelo in modelos:
    # Entrena y evalúa el modelo actual
    resultados = model_pred(modelo, X, y, save_pickle)
        
    # Almacena los resultados en el diccionario
    nombre_modelo = type(modelo).__name__
    resultados_por_modelo[nombre_modelo] = resultados
    

# Imprime los resultados para cada modelo
for nombre_modelo, resultados in resultados_por_modelo.items():
    print(f"Resultados para el modelo {nombre_modelo}:")
    for metrica, valor in resultados.items():
        if metrica in ['fit_time', 'accuracy', 'f1', 'recall', 'precision', 'overfitting']:
            print(f"{metrica.capitalize()}: {valor:.2f}")
            
 # Supongamos que escogemos AdaBoostClassifier. Volvemos a llamar a la función model_pred()

# Flag que indica, cuando es True, que es la última vez que se entrena al modelo definitivo y se puede guardar
#save_pickle = True 
#model_pred(AdaBoostClassifier(n_estimators=200,  random_state=1), X, y, save_pickle)

#y_prob = model_pred.predict_proba(X)[:, 1]

save_pickle = False