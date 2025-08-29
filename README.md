# SMS Spam Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham** using **TF-IDF** and **Multinomial Naive Bayes**. The project is deployed via a **Streamlit** web app.

## Dataset
- Source: SMS Spam Collection  
- 5,169 messages after cleaning  
- 87% Ham, 13% Spam  

## Preprocessing
- Convert text to lowercase and tokenize  
- Remove special characters and stopwords  
- Apply stemming (Porter Stemmer)  
- TF-IDF vectorization  

## Model
- **Multinomial Naive Bayes** selected for perfect spam precision  

