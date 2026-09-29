import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to read and split the CSV files
def read_and_split_csv(training_file_path, testing_file_path):
    # Read the training and testing CSV files into pandas DataFrames
    train_df = pd.read_csv(training_file_path, encoding="latin-1")
    test_df = pd.read_csv(testing_file_path, encoding="latin-1")
    
    # Split the DataFrames into alphabetical and numerical parts
    alphabetical_columns = ['cc_num', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
    numerical_columns = ['amt', 'zip', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
    
    train_alphabetical = train_df[alphabetical_columns]
    train_numerical = train_df[numerical_columns]
    
    test_alphabetical = test_df[alphabetical_columns]
    test_numerical = test_df[numerical_columns]
    
    return train_alphabetical, train_numerical, test_alphabetical, test_numerical

# Replace 'training_file.csv' and 'testing_file.csv' with the actual file paths of your CSV files
train_alphabetical, train_numerical, test_alphabetical, test_numerical = read_and_split_csv('fraudTrain.csv', 'fraudTest.csv')

# Step 1: Handle missing values in textual columns
text_columns = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob']
for col in text_columns:
    train_alphabetical[col].fillna("", inplace=True)
    test_alphabetical[col].fillna("", inplace=True)

# Step 2: Perform Tfidf Vectorization on the text columns
vectorizers = {col: TfidfVectorizer(stop_words=None, lowercase=True) for col in text_columns}

train_text_vectorized = {col: vectorizers[col].fit_transform(train_alphabetical[col]) for col in text_columns}
test_text_vectorized = {col: vectorizers[col].transform(test_alphabetical[col]) for col in text_columns}

# Step 3: Perform Standard Scaling on the numerical data
numerical_column_transformer = ColumnTransformer([('scaler', StandardScaler(), numerical_columns)], remainder='passthrough')

# Transform the numerical data
train_numerical_scaled = numerical_column_transformer.fit_transform(train_numerical)
test_numerical_scaled = numerical_column_transformer.transform(test_numerical)

# Concatenate the transformed data back together
train_X_text = pd.DataFrame({col: train_text_vectorized[col].toarray().tolist() for col in text_columns})
train_X = pd.concat([train_X_text, pd.DataFrame(train_numerical_scaled, columns=train_numerical.columns)], axis=1)

test_X_text = pd.DataFrame({col: test_text_vectorized[col].toarray().tolist() for col in text_columns})
test_X = pd.concat([test_X_text, pd.DataFrame(test_numerical_scaled, columns=test_numerical.columns)], axis=1)

# Step 5: Prepare the target variable 'is_fraud' for training
train_y = train_numerical['is_fraud']
test_y = test_numerical['is_fraud']

# Step 6: Train the SVM model
svm_model = SVC(kernel='rbf', random_state=10, max_iter=10000)
svm_model.fit(train_X, train_y)

# Step 7: Make predictions on the test set
predictions = svm_model.predict(test_X)

# Step 8: Evaluate the model
accuracy = accuracy_score(test_y, predictions)
print("Accuracy of SVM model:", accuracy)