import os
from transformers import pipeline
import pandas as pd

#function that loads the classifier 
def define_classifier():
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=True)
    return classifier

#function loading the dataframe
def load_data(filepath): #takes path to the data as argument 
    filename = os.path.join(filepath)

    data = pd.read_csv(filename, index_col=0)
    return data

data = load_data("../data/fake_or_real_news.csv")

classifier = define_classifier() #calling classifier function and saving it into variable 'classifier'

#Perform emotion classification and saving as a pandas dataframe
def emotion_classification():  
    titles = data["title"]

    anger = []
    disgust = []
    fear = []
    joy = []
    neutral = []
    sadness = []
    surprise = []

    for title in titles:
        score = classifier(title)
        for scores in score:
            anger.append(scores[0]["score"])
            disgust.append(scores[1]["score"])
            fear.append(scores[2]["score"])
            joy.append(scores[3]["score"])
            neutral.append(scores[4]["score"])
            sadness.append(scores[5]["score"])
            surprise.append(scores[6]["score"])
    
    return titles, anger, disgust, fear, joy, neutral, sadness, surprise


def save_emotion_df():
    titles, anger, disgust, fear, joy, neutral, sadness, surprise = emotion_classification()
    df = pd.DataFrame(list(zip(titles, anger, disgust, fear, joy, neutral, sadness, surprise)), columns=['headline', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    
    data_filepath = "../out/data.csv"  # name your output file
    df.to_csv(data_filepath)

save_emotion_df()