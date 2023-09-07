import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, make_scorer

# Load the dataset
file_path = 'airline_passenger_satisfaction.csv'
df = pd.read_csv(file_path)

# Preprocessing steps
# Drop unnecessary columns
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# Handle missing values and type conversions
df['Arrival Delay in Minutes'].fillna(0, inplace=True)
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].astype(int)
df['Customer Type'] = df['Customer Type'].replace('disloyal Customer', 'Disloyal Customer')

# Separate features and target variables
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Identify numerical and categorical columns
num_cols = [col for col in X.columns if X[col].dtype == 'int64']
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

# Define a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                ('scaler', MinMaxScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])

# Define the machine learning models to be used
models = {
    'Logistic_Regression': LogisticRegression(),
    'k-NN': KNeighborsClassifier(),
    'Random_Forest': RandomForestClassifier(),
    'Gradient_Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Define the metrics for evaluation
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted')
}

# Initialize a dictionary to store the results
results = {}

# Train, evaluate, and save each model
for name, model in models.items():
    # Create a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    # Perform 5-fold cross-validation
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, return_train_score=True)
    
    # Calculate overfitting as the difference between training and validation accuracy
    overfitting = np.mean(cv_results['train_accuracy']) - np.mean(cv_results['test_accuracy'])
    
    # Store the evaluation results
    results[name] = {
        'fit_time': np.mean(cv_results['fit_time']),
        'accuracy': np.mean(cv_results['test_accuracy']),
        'f1': np.mean(cv_results['test_f1']),
        'recall': np.mean(cv_results['test_recall']),
        'precision': np.mean(cv_results['test_precision']),
        'overfitting': overfitting
    }
    
    # Train the model on the entire dataset
    pipeline.fit(X, y)
    
    # Save the trained model as a pickle file
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(pipeline, f)

    # Nicely print the evaluation results for the current model
    print(f"{name.upper()}:\n")
    print(f"Accuracy:  {results[name]['accuracy']}")
    print(f"Recall:    {results[name]['recall']}")
    print(f"Precision: {results[name]['precision']}")
    print(f"F1:        {results[name]['f1']}")
    print(f"Overfitting: {results[name]['overfitting']*100:.2f}%")
    print("="*50)

    #TODO: Probar rendimiento con y sin eliminación de outliers
    #TODO: ajustar hiperparámetros, al menos los de Random Forest que obtuvimos con GridSearch