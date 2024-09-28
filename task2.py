import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load the Titanic dataset from seaborn
data = sns.load_dataset('titanic')

# Fill missing values without using inplace=True to avoid chained assignment warnings
data['age'] = data['age'].fillna(data['age'].median())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
data['fare'] = data['fare'].fillna(data['fare'].median())

# Convert categorical columns to numeric using LabelEncoder
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['embarked'] = label_encoder.fit_transform(data['embarked'])
data['who'] = label_encoder.fit_transform(data['who'])
data['class'] = label_encoder.fit_transform(data['class'])
data['deck'] = label_encoder.fit_transform(data['deck'].astype(str))
data['embark_town'] = label_encoder.fit_transform(data['embark_town'])
data['alive'] = label_encoder.fit_transform(data['alive'])
data['alone'] = label_encoder.fit_transform(data['alone'])

# Define features (X) and target (y)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who', 'alone', 'deck', 'class']
X = data[features]
y = data['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model with hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Use the best model found by GridSearchCV
best_rf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
