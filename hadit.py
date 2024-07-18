import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset (replace 'news.csv' with your actual dataset path)
df = pd.read_csv('news.csv')

# Split the dataset into features (X) and labels (y)
X = df['title']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer and transform the training and testing data
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier and fit it to the training data
classifier = MultinomialNB()
classifier.fit(X_train_count, y_train)

def main():
    st.title("Fake News Detection App")

    # Use st.text_area to get user input
    user_input = st.text_area("Paste the article here:")

    # Check if the "Check" button is clicked
    if st.button("Check"):
        if user_input:
            # Transform the user input using the same vectorizer
            user_input_count = vectorizer.transform([user_input])

            # Make a prediction using the trained classifier
            prediction = classifier.predict(user_input_count)

            # Display the result
            st.success(f"The article is classified as: {prediction[0]}")
        else:
            st.warning("Please enter an article for analysis.")

if __name__ == "__main__":
    main()
