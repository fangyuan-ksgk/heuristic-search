# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load the processed data
with open('../data/processed_data.json', 'r') as f:
    processed_data = json.load(f)

# Train model and get predictions
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(processed_data['prompt'])
y = np.array(processed_data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom scoring function that combines our two metrics
def custom_score(y_true, y_pred):
    non_no_mask = y_true != 'No'
    no_mask = y_true == 'No'
    
    # Wrong No ratio (want to minimize)
    wrong_no_ratio = np.sum((y_pred[non_no_mask] == 'No')) / (np.sum(non_no_mask) + 1e-10)
    # Correct No ratio (want to maximize)
    correct_no_ratio = np.sum((y_pred[no_mask] == 'No')) / (np.sum(no_mask) + 1e-10)
    
    # Combined score: maximize correct_no_ratio while minimizing wrong_no_ratio
    return correct_no_ratio - (2 * wrong_no_ratio)  # Penalize wrong predictions more heavily

# Create custom scorer
custom_scorer = make_scorer(custom_score, greater_is_better=True)

# Set up model with grid search
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', {
        'No': 1.0,
        'Yes': 2.0  # Increase penalty for misclassifying 'Yes' as 'No'
    }]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    dt, 
    param_grid, 
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1
)

# Train model
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate and display metrics
non_no_mask = y_test != 'No'
no_mask = y_test == 'No'

wrong_no_ratio = np.sum((y_pred[non_no_mask] == 'No')) / np.sum(non_no_mask)
correct_no_ratio = np.sum((y_pred[no_mask] == 'No')) / np.sum(no_mask)

print("\nBest Parameters:", grid_search.best_params_)
print("\nKey Metrics:")
print(f"1. Non-'No' cases wrongly predicted as 'No': {wrong_no_ratio:.2%}")
print(f"2. 'No' cases correctly predicted as 'No': {correct_no_ratio:.2%}")

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(['Over Rejections', 'Correct Rejections'], 
        [wrong_no_ratio, correct_no_ratio])
plt.ylim(0, 1)
plt.title('Key Prediction Metrics')
plt.ylabel('Ratio')
for i, v in enumerate([wrong_no_ratio, correct_no_ratio]):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
plt.show()