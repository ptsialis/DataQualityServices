import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import plotly.express as px
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def train_and_visualize_model_randomforest(df, target, problem_type, train_size=0.7, test_size=0.15, val_size=0.15):
    # Ensure the splits add up to 1
    assert train_size + test_size + val_size == 1, "Splits must sum up to 1"

    # Split the data into train, test, and validation sets
    train_data, temp_data = train_test_split(df, train_size=train_size, stratify=df[target] if problem_type=='classification' else None, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=val_size/(test_size+val_size), stratify=temp_data[target] if problem_type=='classification' else None, random_state=42)
    
    # Determine the model based on problem type
    if problem_type == 'classification':
        model = RandomForestClassifier(random_state=42)
        scorer = accuracy_score
    else:
        model = RandomForestRegressor(random_state=42)
        scorer = mean_squared_error

    # Define parameter space for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [1],
        'max_depth': [5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [5]
    }

    
    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(train_data.drop(columns=[target]), train_data[target])

    # Best model
    best_model = random_search.best_estimator_

    # Evaluate the best model on the test data
    predictions = best_model.predict(test_data.drop(columns=[target]))
    performance = scorer(test_data[target], predictions)
    print("Performance on Test Data:", performance)

    # Feature importance
    feature_importance = pd.Series(best_model.feature_importances_, index=train_data.drop(columns=[target]).columns)

    # Visualize feature importance using Plotly
    fig = px.bar(feature_importance, x=feature_importance.values, y=feature_importance.index, orientation='h', labels={'index': 'Features', 'value': 'Importance'})
    #fig.update_layout(title_text='Feature Importance')
    

    return best_model,fig

# Example usage:
# df = pd.read_csv('your_data.csv')  # Load your dataset
# model = train_and_visualize_model(df, 'target_variable_name', problem_type='classification' or 'regression')


def train_and_visualize_model_decisiontree(df, target, problem_type, train_size=0.7, test_size=0.15, val_size=0.15):
    # Ensure the splits add up to 1
    assert train_size + test_size + val_size == 1, "Splits must sum up to 1"

    # Split the data into train, test, and validation sets
    train_data, temp_data = train_test_split(df, train_size=train_size, stratify=df[target] if problem_type == 'classification' else None, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=val_size / (test_size + val_size), stratify=temp_data[target] if problem_type == 'classification' else None, random_state=42)
    
    # Determine the model based on problem type
    if problem_type == 'classification':
        model = DecisionTreeClassifier(random_state=42)
        scorer = accuracy_score
    else:
        model = DecisionTreeRegressor(random_state=42)
        scorer = mean_squared_error

    # Define parameter space for RandomizedSearchCV
    param_distributions = {
        'max_depth': [ 4 ],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(train_data.drop(columns=[target]), train_data[target])

    # Best model
    best_model = random_search.best_estimator_

    # Evaluate the best model on the test data
    predictions = best_model.predict(test_data.drop(columns=[target]))
    performance = scorer(test_data[target], predictions)
    print("Performance on Test Data:", performance)

    # Visualize the decision tree
    fig = plt.figure(figsize=(20,10))
    plot_tree(best_model, filled=True, feature_names=df.drop(columns=[target]).columns, class_names=str(best_model.classes_) if problem_type == 'classification' else None, rounded=True, fontsize=10)
    plt.show()

    return best_model, fig