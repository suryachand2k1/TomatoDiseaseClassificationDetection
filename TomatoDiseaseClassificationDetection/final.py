from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
main = tkinter.Tk()
main.title("NLP Text Classification")
main.geometry("1300x1200")
def download_data():
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
	text.insert(END,"Downloaded Dataset\n")
	text.insert(END,"All Categories in the Dataset:\n"+str(twenty_train.target_names))
	return twenty_train

def countvect():
	twenty_train=download_data()
	text.delete('1.0', END)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	text.insert(END,"Fitted CountVectorizer\n")
	text.insert(END,"Training Shape from CountVectorizer:\n"+str(X_train_counts.shape))
	return X_train_counts

def tfidf():
	X_train_counts=countvect()
	text.delete('1.0', END)
	tfidf_transformer =TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	text.insert(END,"Fitted TF-IDF transformer\n")
	text.insert(END,"Training Shape from TF-IDF transformer:\n"+str(X_train_tfidf.shape))
	return X_train_tfidf

def stem():
    twenty_train=download_data()
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    text.delete('1.0', END)
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')   
    text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),('mnb', MultinomialNB(fit_prior=False))])
    text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
    predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
    acc=np.mean(predicted_mnb_stemmed == twenty_test.target)
    text.insert(END,"Accuracy from StemmedCountVectorizer:"+str(acc*100))

def pipe():
	#X_train_tfidf=tfidf()
	twenty_train=download_data()
	text.delete('1.0', END)
	text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	NB = text_clf.fit(twenty_train.data, twenty_train.target)
	text.insert(END,"Completed fitting of Multinomial Naive Bayes\n")
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
	predicted = text_clf.predict(twenty_test.data)
	acc=np.mean(predicted == twenty_test.target)
	text.insert(END,"Accuracy from Multinomial Naive Bayes:"+str(acc*100))
	return NB

def pipe1():
	twenty_train=download_data()
	text.delete('1.0', END)
	text_clf_sgd = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-sgd', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3,random_state=42))])
	sgd = text_clf_sgd.fit(twenty_train.data, twenty_train.target)
	
	text.insert(END,"Completed fitting of SGD Classifier\n")
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
	predicted_sgd = text_clf_sgd.predict(twenty_test.data)
	acc=np.mean(predicted_sgd == twenty_test.target)
	text.insert(END,"Accuracy from SGD Classifier:"+str(acc*100))
	return sgd

def Gridsearch_NB():
	twenty_train=download_data()
	NB=pipe()
	text.delete('1.0', END)
	parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
	gs_clf = GridSearchCV(NB, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
	text.insert(END,"After Tuning Parameters for Naive Bayes\n")
	text.insert(END,"Classifier Best Parameters:"+str(gs_clf.best_params_)+"\n")
	text.insert(END,"Naive Bayes Classifier Best Score:"+str(100*gs_clf.best_score_))
	

def Gridsearch_SGD():
	twenty_train=download_data()
	sgd=pipe1()
	text.delete('1.0', END)
	parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-sgd__alpha': (1e-2, 1e-3)}
	gs_clf = GridSearchCV(sgd, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
	text.insert(END,"After Tuning Parameters for SGD\n")
	text.insert(END,"Classifier Best Parameters:"+str(gs_clf.best_params_)+"\n")
	text.insert(END,"Tuned Sdg Best Score:"+str(100*gs_clf.best_score_))
	



	
font = ('times', 16, 'bold')
title = Label(main, text='NLP Text Classification')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
download = Button(main, text="Download & Categories", command=download_data)
download.place(x=700,y=100)
download.config(font=font1)

cvect = Button(main, text="CountVectorizer", command=countvect)
cvect.place(x=700,y=150)
cvect.config(font=font1)

tfvect = Button(main, text="TF-IDF", command=tfidf)
tfvect.place(x=700,y=200)
tfvect.config(font=font1)

stemmed = Button(main, text="StemmedCountVectorizer", command=stem)
stemmed.place(x=700,y=250)
stemmed.config(font=font1)

pipeline = Button(main, text="MultinomialNB", command=pipe)
pipeline.place(x=700,y=300)
pipeline.config(font=font1)

pipeline1 = Button(main, text="SGD Classifier", command=pipe1)
pipeline1.place(x=700,y=350)
pipeline1.config(font=font1)

grid = Button(main, text="Tuned Naive_Bayes", command=Gridsearch_NB)
grid.place(x=700,y=400)
grid.config(font=font1)

grid1 = Button(main, text="Tuned SGD", command=Gridsearch_SGD)
grid1.place(x=700,y=450)
grid1.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='pale turquoise')
main.mainloop()
