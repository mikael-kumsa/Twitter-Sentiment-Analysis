import numpy as np
import pandas as pd
import re
import nltk 
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# nltk.download('stopwords')

# print(stopwords.words('english'))

#Data Processing part - We will load the data from the csv file to pandas framework
dataPath = "training.1600000.processed.noemoticon.csv"
column_name = ['target','id','date','flag','user', 'text','stemmed_content']
twitter_data = pd.read_csv(dataPath, names=column_name , encoding='ISO-8859-1') # this will load the data to the variable called 'twitter_data'

#Now we will check the number of rows and columns

#print(twitter_data.shape)

# Now we will print the first 5 row of data just to check

#print(twitter_data.head())

#This will count the number of missing values if there are any
#print(twitter_data.isnull().sum())

#Now we have to check the number of positive and negative values 

#target_distribution = twitter_data['target'].value_counts()

#print(target_distribution)

#We will convert the target '4' to '1' for convenience

#twitter_data.replace({'target':{4:1}}, inplace=True)

#0-->'Negative'
#1-->'Positive'

#We will now perform Stemming (reduce a word to its root word)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('^a-zA-Z',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

print(twitter_data.head())

#Separate the data(tweet) and label(target/value)

tweets = twitter_data['stemmed_content'].values
sentiment = twitter_data['target'].values

#We will now split the data for training and test

tweets_train, tweets_test, sentiment_train, sentiment_test = train_test_split(tweets, sentiment, test_size=0.2, stratify=sentiment, random_state=2)

#Converting the textual data to numerical data

vectorizer = TfidfVectorizer()

tweets_train = vectorizer.fit_transform(tweets_train)
tweets_test = vectorizer.transform(tweets_test)

#Training the ML Model (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(tweets_train, sentiment_train)

#Model Evaluation using accuracy_score

tweets_train_prediction = model.predict(tweets_train)
training_data_accuracy = accuracy_score(sentiment_train, tweets_train_prediction)

tweets_test_prediction = model.predict(tweets_test)
test_data_accuracy = accuracy_score(sentiment_test, tweets_train_prediction)

#save the trained model to file using pickles

model_filename = 'trained_model.sav'
pickle.dump(model, open(model_filename, 'wb'))

#loading the saved model into code

loaded_model = pickle.load(open(model_filename), 'rb')

tweets_new = tweets_test[200]
print(sentiment_test[200])

prediction = loaded_model.predict(tweets_new)
print(prediction)
if (prediction[0] == 0):
    print('Negative Tweet')
else:
    print('Positive Tweets')



