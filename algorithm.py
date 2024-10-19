#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data preprocessing


# In[2]:


#remove unwanted fields


# In[5]:


import pandas as pd

# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('student_performance_data1.csv')

# Step 2: Remove unwanted columns
# List the columns you want to remove
unwanted_columns = ['Gender', 'Major','PartTimeJob']  # Replace with the actual column names

# Drop the unwanted columns using drop() method
df = df.drop(columns=unwanted_columns)

# Step 3: (Optional) Save the updated DataFrame to a new CSV file
df.to_csv('updated_data_without_unwanted_columns.csv', index=False)

print("Unwanted columns removed and saved to 'updated_data_without_unwanted_columns.csv")
import pandas as pd


# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('student_performance_data1.csv')

# Step 2: Remove unwanted columns
# List the columns you want to remove
unwanted_columns = ['Gender', 'Major','PartTimeJob']  # Replace with the actual column names

# Drop the unwanted columns using drop() method
df = df.drop(columns=unwanted_columns)

# Step 3: (Optional) Save the updated DataFrame to a new CSV file
df.to_csv('updated_data_without_unwanted_columns.csv', index=False)

print("Unwanted columns removed and saved to 'updated_data_without_unwanted_columns.csv'")


# In[6]:


#converting the string binary to int binary


# In[8]:


import pandas as pd

# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('updated_data_without_unwanted_columns.csv')

# Step 2: Convert 'Yes' and 'No' in specific columns to 1 and 0
# Specify the columns that contain 'Yes' and 'No' values
yes_no_columns = ['ExtraCurricularActivities']  # Replace with the actual column names

# Apply the conversion for each column where 'Yes' = 1 and 'No' = 0
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Step 3: (Optional) Save the updated DataFrame to a new CSV file
df.to_csv('updated_yes_no_to_binary.csv', index=False)

print("Yes/No columns converted to binary and saved to 'updated_yes_no_to_binary.csv'")


# In[9]:


#correlation coefficent


# In[13]:


import pandas as pd
import numpy as np

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('updated_yes_no_to_binary.csv')

# Step 2: Compute the Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')

# Step 3: Insert correlation coefficients back into the DataFrame
# Calculate correlation between 'StudyHoursPerWeek' and 'AttendanceRate' and store it
df['Correlation_StudyHoursPerWeek_AttendanceRate'] = df[['StudyHoursPerWeek', 'AttendanceRate']].corr().iloc[0, 1]



# Step 4: Save the updated DataFrame back to CSV
df.to_csv('updated_data_with_correlations.csv', index=False)

print("Correlation coefficients added and saved to 'updated_data_with_correlations.csv'")


# In[14]:


import pandas as pd
import numpy as np

# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('Details.csv')

# Step 2: Fill two specified columns with random numbers between 1 and 10
# Specify the column names you want to fill wi
th random numbers
columns_to_fill = ['Local_level_hackathon', 'National_level_hackathon']  # Replace with your actual column names

# Generate random integers between 1 and 10 (inclusive) for each specified column
for col in columns_to_fill:
    df[col] = np.random.randint(1, 11, size=len(df))  # 11 is exclusive

# Step 3: Save the updated DataFrame back to CSV
df.to_csv('updated_data.csv', index=False)

print("Random numbers filled in specified columns and saved to 'updated_data.csv'")


# In[15]:


#multiple regression


# In[18]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the CSV file into a pandas DataFrame
# Replace 'student_performance_data.csv' with the actual path to your CSV file
df = pd.read_csv('updated_data.csv')

# Step 2: Define the features to be used and their corresponding weights
features = {
    'GPA': 0.4,  # 40% weight for GPA
    'Local_level_hackathon': 0.2,  # 20% weight for attendance
    'ExtraCurricularActivities': 0.1,  # 10% weight for hackathon participation
    'Correlation_StudyHoursPerWeek_AttendanceRate': 0.2,  # 20% weight for research papers
    'National_level_hackathon': 0.1  # 10% weight for assisting course teachers
}

# Step 3: Normalize the features
scaler = MinMaxScaler()
df_normalized = df.copy()

# Normalize only the selected features
for feature in features:
    if feature in df.columns:
        df_normalized[feature] = scaler.fit_transform(df[[feature]])
    else:
        print(f"Warning: Feature '{feature}' not found in the dataset. It will be skipped.")

# Step 4: Calculate the final score based on the weighted sum of normalized features
df_normalized['FinalScore'] = sum(df_normalized[feature] * weight for feature, weight in features.items() if feature in df_normalized.columns)

# Step 5: Sort students based on their final score in descending order
df_normalized_sorted = df_normalized.sort_values(by='FinalScore', ascending=False)

# Step 6: Save the updated DataFrame with final scores back to CSV
df_normalized_sorted.to_csv('student_scores.csv', index=False)

print("Final scores calculated and saved to 'student_scores.csv'.")


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset
# Replace 'student_data.csv' with the actual path to your CSV file
df = pd.read_csv('student_scores.csv')

# Step 2: Define the independent (X) and dependent (y) variables
# Adjust these column names based on your dataset
X = df[['GPA', 'Local_level_hackathon', 'ExtraCurricularActivities','Correlation_StudyHoursPerWeek_AttendanceRate','National_level_hackathon']]  # Features
y = df['FinalScore']  #Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit the multivariate regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Combine predictions with the original DataFrame
results = pd.DataFrame(X_test.copy())
results['PredictedFinalScore'] = y_pred

# Step 7: Sort the results to find the top 3 students
top_students = results.nlargest(3, 'PredictedFinalScore')

# Print the top 3 students
print("Top 3 Students Based on Predicted Final Scores:")
print(top_students)


# In[ ]:




