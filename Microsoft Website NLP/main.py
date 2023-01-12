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
from flask import Flask, render_template, request, redirect, url_for, Markup

app = Flask(__name__)
df = pd.read_csv("Quran.csv")

df = df[["SurahNo", "AyahNo", "EnglishTranslation", "Tafseer"]]

arabic = pd.read_csv("Arabic.csv")

df.rename(columns={"SurahNo": "Surah", "AyahNo": "Ayat"}, inplace=True)


df = arabic.merge(
    df,
    left_on=['Surah', 'Ayat'],
    right_on=['Surah', 'Ayat'],
    how='inner')


arabic = []
text = []
for index, row in df.iterrows():
    arabic.append(row['Arabic'])
    t = ""
    t += row['Name'] + "|" + str(row['Surah'])+"|"+str(row['Ayat']) + \
        "|" + row['EnglishTranslation'] + "|" + row['Tafseer']
    text.append(t)

# Putting data in variables

surah = []
surah_english = []
ayat = []
translation = []
tafseer = []
tafseer_ayat = []
curr = 1
for i in range(len(text)):
    if int(text[i].split("|")[1]) != curr:
        if len(ayat) > 0:
            surah.append(ayat)
            tafseer.append(tafseer_ayat)
            surah_english.append(translation)
        ayat = []
        tafseer_ayat = []
        translation = []
        curr += 1
    tafseer_ayat.append(text[i].split("|")[-1])
    ayat.append(arabic[i])
    translation.append(text[i].split("|")[-2])
surah.append(ayat)
tafseer.append(tafseer_ayat)
surah_english.append(translation)


quran = ""
for ayats in surah:
    for ayat in ayats:
        quran += ayat + "\n"

quran_english = ""
for ayats in surah_english:
    for ayat in ayats:
        quran_english += ayat + "\n"

tafseer_all = ""
for i in tafseer:
    for j in i:
        tafseer_all += j + "\n"


quran.split("\n")[0]

tafseer_all.split("\n")[0]

len_of_surahs = []
for i in surah:
    len_of_surahs.append(len(i))

len_of_ayats = []
for i in surah:
    for j in i:
        len_of_ayats.append(len(j))

len_of_tafseers = []
for i in tafseer:
    for j in i:
        len_of_tafseers.append(len(j))

# NLP

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


lower_case_quran = quran_english.lower()
lower_case_tafseer = tafseer_all.lower()


def clean_text(lower_case):
    # split text phrases into words
    words = nltk.word_tokenize(lower_case)

    # Create a list of all the punctuations we wish to remove
    punctuations = ['.', ',', '/', '!', '?', ';',
                    ':', '(', ')', '[', ']', '-', '_', '%']

    # Remove all the special characters
    punctuations = re.sub(r'\W', ' ', str(lower_case))

    stop_words = stopwords.words('english')

    # Getting rid of all the words that contain numbers in them
    w_num = re.sub('\w*\d\w*', '', lower_case).strip()

    # remove all single characters
    lower_case = re.sub(r'\s+[a-zA-Z]\s+', ' ', lower_case)

    # Substituting multiple spaces with single space
    lower_case = re.sub(r'\s+', ' ', lower_case, flags=re.I)

    # Removing non-english characters
    lower_case = re.sub(r'^b\s+', '', lower_case)

    # Return keywords which are not in stop words
    keywords = [
        word for word in words if not word in stop_words and word in punctuations and word in w_num]

    return keywords


# Lemmatize the words
wordnet_lemmatizer = WordNetLemmatizer()

lemmatized_word_quran = [wordnet_lemmatizer.lemmatize(
    word) for word in clean_text(lower_case_quran)]

clean_data_quran = ' '.join(lemmatized_word_quran)

# Lemmatize the words
wordnet_lemmatizer = WordNetLemmatizer()

lemmatized_word_tafseer = [wordnet_lemmatizer.lemmatize(
    word) for word in clean_text(lower_case_tafseer)]

clean_data_tafseer = ' '.join(lemmatized_word_tafseer)

df_clean_quran = pd.DataFrame([clean_data_quran])
df_clean_quran.columns = ['script']
df_clean_quran.index = ['quran']

df_clean_tafseer = pd.DataFrame([clean_data_tafseer])
df_clean_tafseer.columns = ['script']
df_clean_tafseer.index = ['tafseer']

#  Counting the occurrences of tokens and building a sparse matrix of documents x tokens.

corpus_quran = df_clean_quran.script
vect_quran = CountVectorizer(stop_words='english')

# Transforms the data into a bag of words
data_vect_quran = vect_quran.fit_transform(corpus_quran)

corpus_tafseer = df_clean_tafseer.script
vect_tafseer = CountVectorizer(stop_words='english')

# Transforms the data into a bag of words
data_vect_tafseer = vect_tafseer.fit_transform(corpus_tafseer)

feature_names_quran = vect_quran.get_feature_names()
data_vect_feat_quran = pd.DataFrame(
    data_vect_quran.toarray(), columns=feature_names_quran)
data_vect_feat_quran.index = df_clean_quran.index

feature_names_tafseer = vect_tafseer.get_feature_names()
data_vect_feat_tafseer = pd.DataFrame(
    data_vect_tafseer.toarray(), columns=feature_names_tafseer)
data_vect_feat_tafseer.index = df_clean_tafseer.index

data_quran = data_vect_feat_quran.transpose()

data_tafseer = data_vect_feat_tafseer.transpose()


top_dict_quran = {}
for c in data_quran.columns:
    top = data_quran[c].sort_values(ascending=False)
    top_dict_quran[c] = list(zip(top.index, top.values))

top_dict_tafseer = {}
for c in data_tafseer.columns:
    top = data_tafseer[c].sort_values(ascending=False)
    top_dict_tafseer[c] = list(zip(top.index, top.values))


word_count_dict = dict(top_dict_quran['quran'])
popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
popular_words_nonstop = [
    w for w in popular_words if w not in stopwords.words("english")]
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      max_words=1000,
                      width=1000, height=1000,
                      ).generate(" ".join(popular_words_nonstop))
plt.imshow(wordcloud, interpolation='bilinear')
fig = plt.gcf()
fig.set_size_inches(10, 12)
plt.axis('off')
plt.title("Top most common 1000 words from Quran", fontsize=20)
plt.tight_layout(pad=0)

word_count_dict = dict(top_dict_tafseer['tafseer'])
popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
popular_words_nonstop = [
    w for w in popular_words if w not in stopwords.words("english")]
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      max_words=1000,
                      width=1000, height=1000,
                      ).generate(" ".join(popular_words_nonstop))
plt.imshow(wordcloud, interpolation='bilinear')
fig = plt.gcf()
fig.set_size_inches(10, 12)
plt.axis('off')
plt.title("Top most common 1000 words from Tafseer", fontsize=20)
plt.tight_layout(pad=0)

word_count_dict = dict(top_dict_quran['quran'][:50])
popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
popular_words_nonstop = [
    w for w in popular_words if w not in stopwords.words("english")]
plt.figure(figsize=(10, 10))
plt.barh(range(50), [word_count_dict[w]
         for w in reversed(popular_words_nonstop[0:50])])
plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
plt.title("Top words")
plt.show()

word_count_dict = dict(top_dict_tafseer['tafseer'][:50])
popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
popular_words_nonstop = [
    w for w in popular_words if w not in stopwords.words("english")]
plt.figure(figsize=(10, 10))
plt.barh(range(50), [word_count_dict[w]
         for w in reversed(popular_words_nonstop[0:50])])
plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
plt.title("Top words")
plt.show()

# Sentiment Analysis

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Islam is the best religion!")

sia.polarity_scores(quran_english)

sia.polarity_scores(tafseer_all)

s_num = 1
for ayats in surah:
    s = ""
    for ayat in ayats:
        # print(ayat.split("|")[-1])
        s += ayat.split("|")[-1] + "\n"
    print(s_num, sia.polarity_scores(s))
    s_num += 1

s_num = 1
for ayats in tafseer:
    s = ""
    for ayat in ayats:
        # print(ayat.split("|")[-1])
        s += ayat.split("|")[-1] + "\n"
    print(s_num, sia.polarity_scores(s))
    s_num += 1

# TEXT SUMMARY

# Summary of Quran


def summarise(text):
    sentence_list = nltk.sent_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(
        7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


summarise(quran_english)

# Summary of Surahs

s_num = 1
for ayats in surah_english:
    s = ""
    for ayat in ayats:
        # print(ayat.split("|")[-1])
        s += ayat.split("|")[-1] + "\n"
    summary = summarise(s)
    if len(summary) < 5:
        print("SURAH NUMBER:", s_num, "SURAH:", s)
    else:
        print("SURAH NUMBER:", s_num, "SUMMARY:", summary)
    print()
    s_num += 1

# Summary of Tafaseer

s_num = 1
for ayats in tafseer:
    s = ""
    for ayat in ayats:
        # print(ayat.split("|")[-1])
        s += ayat.split("|")[-1] + "\n"
    summary = summarise(s)
    if len(summary) < 5:
        print("SURAH NUMBER:", s_num, "TAFSEER:", s)
    else:
        print("SURAH NUMBER:", s_num, "TAFSEER SUMMARY:", summary)
    print()
    s_num += 1

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
df.head()

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

    if size == 1:
        print("Query: ", query, ".  Surah:", find.loc[0, 'Name'], " Number:", find.loc[0, 'Surah'],
              " Ayat:", find.loc[0, 'Ayat'], " Tafseer: ", find.loc[0, 'Tafseer'], "\n")
    else:
        return find


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        QUERy = request.form["query"]
        return redirect(url_for("query_results", Query=QUERy))
    else:
        return render_template("main.html")


@app.route("/search<Query>")
def query_results(Query):
    out=Markup(SearchDocument(str(Query)).to_html())
    return render_template("query_results.html", query=Query , answer=out)

if __name__ == "__main__":
    app.run(debug=True)
