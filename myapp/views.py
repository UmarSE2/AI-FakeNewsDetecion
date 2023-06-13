from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score


def news_detector(request):
    if request.method == 'POST':
        text = request.POST['text']
        df = pd.DataFrame({'text': [text]})
        df = df.replace(np.nan, '')

        # Load the pre-trained model and perform prediction
        df = pd.read_csv(r'E:\6thsemseter\fakepc\new folder\fakenews\myapp\news.csv', dtype=str)
        df = df.replace(np.nan, '')

        labels = df.label
        x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train.astype('U'))
        tfidf_test = tfidf_vectorizer.transform(x_test.astype('U'))

        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        y_pred = pac.predict(tfidf_test)

        # Calculate evaluation metrics
        score = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        # Prepare the results to pass to the template
        results = {
            'accuracy': round(score * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'recall': round(recall * 100, 2),
            'precision': round(precision * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']),
            'predictions': zip(x_test, y_pred)
        }

        return render(request, 'results.html', results)

    return render(request, 'index.html')

# def home(request):
#     return render(request, 'home.html')  # Replace 'home.html' with your desired template
