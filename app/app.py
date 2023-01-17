import pandas as pd
import collections
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import heapq
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, brown
import string
import re
import nltk
from matplotlib import pyplot as plt
import os
from flask import Flask, render_template, request, redirect, url_for, Markup

app = Flask(__name__, static_url_path='/data')
"""
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "data/")
df = pd.read_csv(path + "Quran.csv")

df = df[["SurahNo", "AyahNo", "EnglishTranslation", "Tafseer"]]

arabic = pd.read_csv(path + "Arabic.csv")

df.rename(columns={"SurahNo": "Surah", "AyahNo": "Ayat"}, inplace=True)


df = arabic.merge(
    df,
    left_on=['Surah', 'Ayat'],
    right_on=['Surah', 'Ayat'],
    how='inner')

# Search Engine

nltk.download('averaged_perceptron_tagger')

df['useful_info'] = df.EnglishTranslation.astype(str) + df.Tafseer.astype(str)

df["useful_info"] = df.useful_info.replace(
    to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]', value=' ', regex=True)
# remove punctuation
df.useful_info = df.useful_info.replace(to_replace='-', value=' ', regex=True)
df.useful_info = df.useful_info.replace(
    to_replace='\s+', value=' ', regex=True)  # remove new line
df.useful_info = df.useful_info.replace(
    to_replace='  ', value='', regex=True)  # remove double white space
df.useful_info = df.useful_info.replace(to_replace="'", value='', regex=True)


df.useful_info = df.useful_info.apply(
    lambda x: x.strip().lower())  # Ltrim and Rtrim of whitespace

df["info_tokenize"] = [word_tokenize(entry)
                       for entry in tqdm(df["useful_info"])]


def wordLemmatizer(data):
    tag_map = collections.defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    clean_k = pd.DataFrame()
    word_Lemmatized = WordNetLemmatizer()
    for index, entry in tqdm(enumerate(data)):

        Final_words = []
        for word, tag in pos_tag(entry):
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)

        clean_k.loc[index, 'Keyword_final'] = str(Final_words)
    clean_k = clean_k.replace(to_replace="'", value='', regex=True)
    clean_k = clean_k.replace(to_replace=" ", value='', regex=True)
    clean_k = clean_k.replace(to_replace="\[", value='', regex=True)
    clean_k = clean_k.replace(to_replace='\]', value='', regex=True)
    return clean_k


df["Keyword_final"] = wordLemmatizer(df['info_tokenize'])


# Using Google Universal Sentence Encoder
USEmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# NEED TO DOWNLOAD THE MODEL
#USEmodel = hub.load("model/universal-sentence-encoder_4/")

train = USEmodel(df.Keyword_final)
train_m = tf.train.Checkpoint(v=tf.Variable(train))

train_m.f = tf.function(lambda x: exported_m.v * x,
                        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])

model = train_m.v.numpy()

model.shape


def SearchDocument(query, size=10):

    q = wordLemmatizer([query.split(" ")])
    # print(q.values[0])
    Q_Train = USEmodel(q.values[0])

    linear_similarities = linear_kernel(Q_Train, model).flatten()

    Top_index_doc = linear_similarities.argsort()[:-(size+1):-1]
    # print(Top_index_doc)
    linear_similarities.sort()
    find = pd.DataFrame()
    for i, index in enumerate(Top_index_doc):
        find.loc[i, 'Name'] = str(df['Name'][index])
        find.loc[i, 'Surah'] = str(df['Surah'][index])
        find.loc[i, 'Ayat'] = str(df['Ayat'][index])
        find.loc[i, 'Arabic'] = df['Arabic'][index]
        find.loc[i, 'EnglishTranslation'] = df['EnglishTranslation'][index]
        find.loc[i, 'Tafseer'] = df['Tafseer'][index]

    for j, simScore in enumerate(linear_similarities[:-(size+1):-1]):
        find.loc[j, 'Score'] = simScore

    return find
"""
path_to_home="main.html"
path_to_result="query_results.html"

@app.route("/", methods=["POST", "GET"])
@app.route("/home", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        QUERy = request.form["query"]

        return redirect(url_for("query_results", Query=QUERy))
    else:
        return render_template(path_to_home)

@app.route("/search<Query>")
def query_results(Query):
    print(Query)
    #print(SearchDocument(Query))
    
    out=[
        "The Cow | 2 | 15 | ٱللَّهُ يَسْتَهْزِئُ بِهِمْ وَيَمُدُّهُمْ فِي طُغْيَانِهِمْ يَعْمَهُونَ | Allah will throw back their mockery on them, and give them rope in their trespasses; so they will wander like blind ones (To and fro). | There is no commentary by Abul Maududi available for this verse",
     "The Cow | 2 | 15 | ٱللَّهُ يَسْتَهْزِئُ بِهِمْ وَيَمُدُّهُمْ فِي طُغْيَانِهِمْ يَعْمَهُونَ | Allah will throw back their mockery on them, and give them rope in their trespasses; so they will wander like blind ones (To and fro). | There is no commentary by Abul Maududi available for this verse",
     "The Cow | 2 | 15 | ٱللَّهُ يَسْتَهْزِئُ بِهِمْ وَيَمُدُّهُمْ فِي طُغْيَانِهِمْ يَعْمَهُونَ | Allah will throw back their mockery on them, and give them rope in their trespasses; so they will wander like blind ones (To and fro). | There is no commentary by Abul Maududi available for this verse",
     "The Cow | 2 | 15 | ٱللَّهُ يَسْتَهْزِئُ بِهِمْ وَيَمُدُّهُمْ فِي طُغْيَانِهِمْ يَعْمَهُونَ | Allah will throw back their mockery on them, and give them rope in their trespasses; so they will wander like blind ones (To and fro). | There is no commentary by Abul Maududi available for this verse",
     ]
    return render_template(path_to_result, query=Query , answer=out , home_url=url_for("home"))

