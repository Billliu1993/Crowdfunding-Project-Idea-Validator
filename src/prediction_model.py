# Essential packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
# Text cleaning packages
import nltk 
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# Text feature packages
import textstat
from lexical_diversity import lex_div as ld
# Word embedding
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec
# Modeling packages
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

class Kickstarter_Prediction_Model():

    def __init__(self):
        self.model = pickle.load(open('models/current_XGBClassifier.sav', 'rb'))
        self.dv = Doc2Vec.load("./models/doc2vec_model")

    @staticmethod
    def lem_words(text):
        text = text.split()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        return text

    @staticmethod
    def make_lower_case(text):
        return text.lower()

    @staticmethod
    def remove_stop_words(text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    @staticmethod
    def remove_punctuation(text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text

    def clean_message(self, message):
        message = self.make_lower_case(message)
        message = self.remove_stop_words(message)
        message = self.remove_punctuation(message)
        message = self.lem_words(message)
        return message

    def length(self, text):
        return len(text)

    def lex_div(self, text):

        token = ld.tokenize(text)
        return ld.ttr(token)

    def lex_entropy(self, text, base = 2.0):

        dct = dict.fromkeys(list(text))
        pkvec = [float(text.count(c)) / len(text) for c in dct]
        H = -sum([pk * math.log(pk) / math.log(base) for pk in pkvec])
        return H

    def lex_readability(self, text, mode = 'all'):

        if mode == 'all':
            fre_score = textstat.flesch_reading_ease(text)
            fog_index = textstat.gunning_fog(text)
            fkg_index = textstat.flesch_kincaid_grade(text)
            dcr_score = textstat.dale_chall_readability_score(text)
            text_standard = textstat.text_standard(text, float_output=True)
            return fre_score, fog_index, fkg_index, dcr_score, text_standard
        
        if mode == 'fre':
            fre_score = textstat.flesch_reading_ease(text)
            return fre_score
        
        if mode == 'fog':
            fog_index = textstat.gunning_fog(text)
            return fog_index
        
        if mode == 'fkg':
            fkg_index = textstat.flesch_kincaid_grade(text)
            return fkg_index
        
        if mode == 'dcr':
            dcr_score = textstat.dale_chall_readability_score(text)
            return dcr_score
        
        if mode == 'text_std':
            text_standard = textstat.text_standard(text, float_output=True)
            return text_standard

    def category_to_dummy(self, cat):
        cat_lst = ['art', 'comics', 'crafts', 'design', 'fashion', 'film & video', 'food', 'games', 'journalism',
                    'music', 'photography', 'publishing', 'technology', 'theater']
        cat_dummy = []

        for i in cat_lst:
            if i == cat:
                cat_dummy.append(1)
            else:
                cat_dummy.append(0)

        return cat_dummy

    def get_cust_features(self, submit):
        feature_lst = [float(submit.days_to_deadline.value), float(submit.goal_USD.value), 1]

        cleaned_text = self.clean_message(submit.description.value)

        feature_lst.append(self.length(cleaned_text))
        feature_lst.append(self.lex_div(cleaned_text))
        feature_lst.append(self.lex_entropy(cleaned_text))
        for i in self.lex_readability(cleaned_text):
            feature_lst.append(i)
        for j in self.category_to_dummy(submit.category.value):    
            feature_lst.append(j)

        return feature_lst

    def get_doctovec_features(self, text):
        message_array = self.dv.infer_vector(doc_words=text.split(" "), epochs=200)
        return message_array

    def make_prediction(self, submit):
        #model = pickle.load(open('models/current_RandomForestClassifier.sav', 'rb'))
        model = pickle.load(open('models/current_XGBClassifier.sav', 'rb'))

        cust_feature = self.get_cust_features(submit)
        d2v_feature = self.get_doctovec_features(submit.description.value)

        for i in d2v_feature:
            cust_feature.append(i)

        pred_class = model.predict([cust_feature])[0]
        if pred_class == 0:
            pred_label = 'FAIL'
        else:
            pred_label = 'SUCCESSFUL'

        pred_proba = model.predict_proba([cust_feature])[0][1]

        print("=" * 117)
        print('KICKSTARTER IDEA VALIDATOR RESULTS:')
        print("=" * 117)
        print('\nBased on your input, the forecasted results are:')
        print('=> Predicted results: {}'.format(pred_label))
        print('=> Successful probability: {:.0%}'.format(pred_proba))
        print("=" * 117)


