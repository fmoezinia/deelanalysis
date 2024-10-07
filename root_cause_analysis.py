
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy
# chargeback_data, acceptance_data = read_csv_files()
# full_data = join_chargeback_acceptance(chargeback_data, acceptance_data)

# Load your data (assuming you already have the data in a pandas DataFrame)
data = pd.read_csv('full_acceptance_data.csv')
data = data.drop(columns=['external_ref', 'source', 'ref', 'rates', 'country'])  # Drop columns not used in modeling

# Step 1: Prepare the data
# Assuming 'state' is the target and we need to convert it into binary (0 = Declined, 1 = Accepted)
data['state'] = data['state'].apply(lambda x: 1 if x == 'ACCEPTED' else 0)

# Convert categorical features like 'country', 'currency' to dummy variables (one-hot encoding)
data = pd.get_dummies(data, columns=['currency'], drop_first=False)

# datetime
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce')
data['hour'] = data['date_time'].dt.hour  # Extracting the hour
data['day_of_week'] = data['date_time'].dt.dayofweek  # Extracting the day of the week (0=Monday, 6=Sunday)
data['month'] = data['date_time'].dt.month  # Extracting the month

# pairwise
data['amount_x_currency_USD'] = data['amount'] * data['currency_USD'] 
data['month_x_currency_USD'] = data['month'] * data['currency_USD'] 
data['hour_x_currency_USD'] = data['hour'] * data['currency_USD'] 
data['hour_x_amount'] = data['hour'] * data['amount'] 


# Select relevant features
X = data[['amount', 'cvv_provided', 'hour', 'day_of_week', 'month', 'currency_USD', \
          'amount_x_currency_USD', 'month_x_currency_USD', 'hour_x_currency_USD', 'hour_x_amount', \
          'currency_EUR', 'currency_GBP', 'currency_CAD', 'currency_MXN']]  # relevant features
print(X.head(10))  # View the first few rows of the feature matrix

y = data['state']  # The target column

# Step 2: Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def run_decision_tree_classifier():
    # Step 3: Initialize and train the decision tree model
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=3)
    tree_model.fit(X_train, y_train)

    # Step 4: Make predictions
    y_pred = tree_model.predict(X_test)

    # Step 5: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Visualize the tree
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15,10))
    plot_tree(tree_model, feature_names=X.columns, class_names=['Declined', 'Accepted'], filled=True)
    plt.show()

# run_decision_tree_classifier()

def run_logistic_regression():

    # Initialize and train the logistic regression model
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
    logreg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = logreg.predict(X_test)

    # Get the coefficients from the trained model
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': logreg.coef_[0]
    })

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, zero_division=1))

    # Sort by absolute value of the coefficients to see the most influential features
    coefficients['abs_Coefficient'] = numpy.abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values(by='abs_Coefficient', ascending=False)

    # Display the top features
    print(coefficients)

run_logistic_regression()
