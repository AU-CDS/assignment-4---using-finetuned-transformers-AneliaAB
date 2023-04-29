[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10837145&assignment_repo_type=AssignmentRepo)

# Assignment 4 - Using finetuned transformers via HuggingFace

## Project Description by Ross
In previous assignments, you've done a lot of model training of various kinds of complexity, such as training document classifiers or RNN language models. This assignment is more like Assignment 1, in that it's about *feature extraction*.

For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines

- **MAKE SURE TO UPDATE YOUR README APPROPRIATELY!**

## Data
The dataset consists of four columns (number, title, text, and label) and 6336 rows. This project only uses the 'title' column, which is the headlines of the articles, and the label column. The data is already in the repository in the ```data``` folder, so there is no need to load it separately.

## How to Install and Run the Project
Installation:

1. First you need to clone the repository 
2. Navigate from the root of your repository to ```assignment-4---using-finetuned-transformers-AneliaAB```
3. Run the setup file, which will install all the requirements by writing ```bash setup.sh``` in the terminal
4. Navigate to the folder ```scr``` and run the script by writing ```python code.py``` in the terminal

## Results 
After installing and running the scripts, you should be able to see the results in the ```out``` folder. The results are in the form of a new dataset and a .png file. 

- Dataset
The new dataset consists of 6335 rows and 8 columns - headline, anger, disgust, fear, joy, neutral, sadness, and surprise. The dataset shows the score of every emotion for every headline of the original data. 

- Png 
The png file shows a bar plot of the average score of each emotion for all headlines. This is done by dividing the sum of the scores with the number of scores. The barplot shows the emotions on the y axis and the average score on the x axis. This helps visualize the overall emotion in all headlines.

## Challenges 
I struggle with importing tranformers when running the python script ```code.py```, so I created a notebook that runs the same code, which you can find in the ```nb```folder. When rinning the notebook I don't get any error or problem with transformers, and I am working on fixing the issue with the script. 