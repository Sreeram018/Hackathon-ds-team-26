#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[15]:


from flask import Flask, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sqlite3
import threading

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load CSV data into the SQLite database (if it doesn't exist)
def load_csv_to_db():
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()

    # Drop the students table if it already exists (use cautiously)
    cursor.execute('DROP TABLE IF EXISTS students')

    # Create the students table with the correct columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        StudentID INTEGER,
        name TEXT,
        GPA REAL,
        Local_level_hackathon REAL,
        ExtraCurricularActivities REAL,
        Correlation_StudyHoursPerWeek_AttendanceRate REAL,
        National_level_hackathon REAL,
        FinalScore REAL
    )
    ''')

    # Check if the table is already populated
    cursor.execute('SELECT COUNT(*) FROM students')
    if cursor.fetchone()[0] == 0:  # If empty, load CSV
        df = pd.read_csv('student_scores.csv', header=0)  # Load the CSV data
        
        # Print the column names for debugging
        print("Columns in CSV:", df.columns.tolist())
        
        # Normalize column names
        df.columns = df.columns.str.strip()  # Strip whitespace from headers
        
        # Use iterrows to insert each row into the database
        for index, row in df.iterrows():
            cursor.execute('''
            INSERT INTO students (StudentID, GPA, Local_level_hackathon, ExtraCurricularActivities, 
            Correlation_StudyHoursPerWeek_AttendanceRate, National_level_hackathon, FinalScore) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['StudentID'], row['GPA'], row['Local_level_hackathon'], 
                  row['ExtraCurricularActivities'], row['Correlation_StudyHoursPerWeek_AttendanceRate'], 
                  row['National_level_hackathon'], row['FinalScore']))
    
    conn.commit()
    conn.close()


# Step 2: Function to fetch data from the database
def get_student_data():
    conn = sqlite3.connect('students.db')
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df

# Step 3: Calculate the top 3 students using the regression model
def calculate_top_students():
    df = get_student_data()  # Fetch data from the database
    
    if df.empty:
        return pd.DataFrame({'name': [], 'PredictedFinalScore': []})

    X = df[['GPA', 'Local_level_hackathon', 'ExtraCurricularActivities', 'Correlation_StudyHoursPerWeek_AttendanceRate', 'National_level_hackathon']]
    y = df['FinalScore']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict final scores
    y_pred = model.predict(X_test)

    # Combine predictions with the original DataFrame
    results = pd.DataFrame(X_test.copy())
    results['PredictedFinalScore'] = y_pred
    results['StudentID'] = df.loc[X_test.index, 'StudentID']  # Use StudentID instead of name
  # Add student names back using indices

    # Get the top 3 students based on predicted final scores
    top_students = results.nlargest(3, 'PredictedFinalScore')

    return top_students[['StudentID', 'PredictedFinalScore']]


# Step 4: Route to display the top 3 students
@app.route('/')
def top_students():
    try:
        # Load CSV to database (if needed)
        load_csv_to_db()

        # Get top 3 students
        top_students = calculate_top_students()

        # Simple HTML template (you can expand this)
        html_template = """
        <h1>Top 3 Students</h1>
        <table border="1">
          <tr>
            <th>Student ID</th>
            <td>{{ student.StudentID }}</td>
          </tr>
          {% for student in students %}
          <tr>
            <td>{{ student.name }}</td>
            <td>{{ student.PredictedFinalScore }}</td>
          </tr>
          {% endfor %}
        </table>
        """

        # Render the HTML page with top students
        return render_template_string(html_template, students=top_students.to_dict(orient='records'))
    except Exception as e:
        return f"An error occurred: {e}"

# Run the Flask app in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False)

# Start Flask in a thread to avoid blocking Jupyter
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


# In[ ]:




