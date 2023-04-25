import os
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

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

#creating plot with matplotlib
titles, anger, disgust, fear, joy, neutral, sadness, surprise = emotion_classification()

#function for finding average number of a list 
def average(lst):
    return sum(lst) / len(lst)

plt.style.use('_mpl-gallery')

#x axis
x = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
#finding the average of each emotion via average() function 
avr_anger = average(anger)
avr_disgust = average(disgust)
avr_fear = average(fear)
avr_joy = average(joy)
avr_neutral = average(neutral)
avr_sadness = average(sadness)
avr_surprise = average(surprise)

#y axis
y = avr_anger, avr_disgust, avr_fear, avr_joy, avr_neutral, avr_sadness, avr_surprise

#create plot
fig, ax = plt.subplots()
plt.barh(x, y, edgecolor="white", linewidth=0.7, align='center')
plt.title('Average score of each emotion (all headlines)')

#saving plot
plt.savefig('../out/average.png')
