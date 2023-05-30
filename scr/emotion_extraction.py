#%%
import os
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

#%%
#function that loads the classifier 
def define_classifier():
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=True)
    return classifier

#%%
#function loading the dataframe
def load_data(keyword): #takes path to the data as argument 
    print("loading data")
    filename = os.path.join("..", "data", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)

    print("Loading the data..")
    if keyword.upper() == "ALL":
        return data
    elif keyword.upper() == "FAKE":
        data_fake = data[data['label'] == 'FAKE']
        return data_fake
    elif keyword.upper() == "REAL":
        data_real = data[data['label'] == 'REAL']
        return data_real

classifier = define_classifier() #calling classifier function and saving it into variable 'classifier'

#%%
#Perform emotion classification and saving as a pandas dataframe
def emotion_classification(keyword):  
    print("emotion classification..")
    data = load_data(keyword)
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
    
    print("Extracting emotions..")
    return titles, anger, disgust, fear, joy, neutral, sadness, surprise

#
def save_emotion_df(keyword):
    titles, anger, disgust, fear, joy, neutral, sadness, surprise = emotion_classification(keyword)
    df = pd.DataFrame(list(zip(titles, anger, disgust, fear, joy, neutral, sadness, surprise)), columns=['headline', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    
    data_filepath = "../out/data.csv"  # name your output file
    df.to_csv(data_filepath)

#save_emotion_df()

#function for finding average number of a list 
def average(lst):
    return sum(lst) / len(lst)

plt.style.use('_mpl-gallery')

#x axis
def find_average(keyword):
    print("find avarage..")
    titles, anger, disgust, fear, joy, neutral, sadness, surprise = emotion_classification(keyword)
    x = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
    #finding the average of each emotion via average() function 
    avr_anger = average(anger)
    avr_disgust = average(disgust)
    avr_fear = average(fear)
    avr_joy = average(joy)
    avr_neutral = average(neutral)
    avr_sadness = average(sadness)
    avr_surprise = average(surprise)

    return avr_anger, avr_disgust, avr_fear, avr_joy, avr_neutral, avr_sadness, avr_surprise

#visualizing fake and real data 
def df_fake_barplot(keyword):
    print("Creating barplot..")

    avr_anger, avr_disgust, avr_fear, avr_joy, avr_neutral, avr_sadness, avr_surprise = find_average(keyword)
    #saving average values in y variable
    y = avr_anger, avr_disgust, avr_fear, avr_joy, avr_neutral, avr_sadness, avr_surprise
    x = ('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise')

    #create plot
    plt.barh(x, y)

    plt.xlabel('Emotion')
    plt.ylabel('Average')
    plt.title(f'Average emotion for {keyword} headlines')

    plt.subplots_adjust(left=0.40, bottom=0.40)
    #saving plot
    plt.savefig(os.path.join(f"../out/avarage_{keyword}.png"))

    plt.show()

keyword = input("Generate outout for all, fake or real news: ")
df_fake_barplot(keyword)