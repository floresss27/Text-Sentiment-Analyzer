from textblob import TextBlob
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(text)

    words_without_stopwords = [word for word in words if word.lower() not in stopwords.words('english')]
    
    processed_text = ' '.join(words_without_stopwords)
    
    return processed_text

def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    print("Text after preprocessing:", processed_text)
    
    analysis = TextBlob(processed_text)
    
    return analysis.sentiment.polarity

textos = [
    "I had an amazing experience at the new restaurant downtown. The food was delicious, and the service was outstanding!",
    "This movie is the worst I've ever seen. The plot is confusing, and the acting is terrible.",
    "Just booked my dream vacation to a tropical paradise! Can't wait to relax on the beach and soak up the sun.",
    "The customer support team was very helpful in resolving my issue. I appreciate their quick response and professionalism.",
    "Today was a challenging day at work. The workload was overwhelming, and I felt stressed throughout the day.",
    "The concert last night was phenomenal! The band played all my favorite songs, and the energy in the crowd was electric.",
    "I received a defective product, and the company's return process was a nightmare. Very disappointed with their customer service.",
    "Enjoying a quiet evening at home with a good book. Sometimes, it's the simple things that bring the most joy.",
    "The traffic on the way to work was unbearable this morning. It took me twice as long to get to the office.",
    "Just finished a great workout at the gym. Feeling energized and ready to tackle the day!"
]

for i in textos:
    sentiment = analyze_sentiment(i)
    print("Final Result:", sentiment)
