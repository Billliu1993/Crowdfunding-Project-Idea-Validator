import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from textwrap import wrap
import re
import math
import textstat

import matplotlib.pyplot as plt
from skimage import io

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse


class Kickstarter_Information_Retrieval_Model():

    def __init__(self):
        self.dv = Doc2Vec.load("./models/doc2vec_model")
        self.tf = pickle.load(open("models/tfidf_model.pkl", "rb"))
        self.svd = pickle.load(open("models/svd_model.pkl", "rb"))
        self.svd_feature_matrix = pickle.load(open("models/lsa_embeddings.pkl", "rb"))
        self.doctovec_feature_matrix = pickle.load(open("models/doctovec_embeddings.pkl", "rb"))
        self.df = pd.read_csv("data/model_data_newVar.csv")


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


    def get_message_tfidf_embedding_vector(self, message):
        message_array = self.tf.transform([message]).toarray()
        message_array = self.svd.transform(message_array)
        message_array = message_array[:,:].reshape(1, -1)
        return message_array


    def get_message_doctovec_embedding_vector(self, message):
        message_array = self.dv.infer_vector(doc_words=message.split(" "), epochs=200)
        message_array = message_array.reshape(1, -1)
        return message_array


    @staticmethod
    def get_similarity_scores(message_array, embeddings):
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix


    def get_ensemble_similarity_scores(self, message):
        message = self.clean_message(message)
        bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        bow_similarity = self.get_similarity_scores(bow_message_array, self.svd_feature_matrix)
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)

        ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
        ensemble_similarity.columns = ["semantic_similarity", "bow_similarity"]
        ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"] + ensemble_similarity["bow_similarity"])/2
        
        ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
        ensemble_similarity.reset_index(inplace = True)
        
        return ensemble_similarity

    def query_similar_kickstarter(self, message, cat):
        
        similar_kickstarter = self.get_ensemble_similarity_scores(message)
        category_df = self.df[['name', 'category_slug']]
        
        similar_proj = similar_kickstarter.merge(category_df, on = 'name', how = 'left')
        rec_df = similar_proj.loc[similar_proj['category_slug'] == cat,]
             
        return rec_df.head(3)

    def lex_readability(self, text, mode = 'fre'):

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

    def lex_entropy(self, text, base = 2.0):

        dct = dict.fromkeys(list(text))
        pkvec = [float(text.count(c)) / len(text) for c in dct]
        H = -sum([pk * math.log(pk) / math.log(base) for pk in pkvec])
        return H
    
    
    def get_similar_project(self, submit):
        recs = self.query_similar_kickstarter(submit.description.value, submit.category.value)
        clean_text = self.clean_message(submit.description.value)


        idea_val = [float(submit.days_to_deadline.value), 
                    float(submit.goal_USD.value), 
                    len(submit.days_to_deadline.value), 
                    self.lex_readability(submit.description.value), 
                    self.lex_entropy(clean_text)]
        titles = ['Funding days (days)', 'Funding amount (USD)', 'File length (words)', 'File readability (a.u.)', 'File richness (a.u.)']

        print('\nMost similar projects on Kickstarter are identified for further comparison:')
        print("=" * 117)

        for i in range(len(recs)):
            single_name = recs.name[i]
            single_kickstarter = self.df.query('name==@single_name')
            
            name = single_kickstarter.name.values[0]
            category = single_kickstarter.category_slug.values[0]
            state = single_kickstarter.binary_state.values[0]
            story = single_kickstarter.blurb.values[0]
            similarity = recs.ensemble_similarity[i]
            duration = single_kickstarter.days_to_deadline.values[0]
            amount = single_kickstarter.goal_USD.values[0]
            doc_length = single_kickstarter.length.values[0]
            doc_readability = single_kickstarter.fre_score.values[0]
            doc_richness = single_kickstarter.lex_entropy.values[0]
            url = single_kickstarter.URL.values[0]
            
           
            print("NAME: {}".format(name.upper()))
            print("DESCRIPTION: {}".format(story))
            print("\nSTATUS: {}".format(state.upper()))
            print("SIMILAR LEVEL: {:.0%}".format(similarity))
            print("DETAILED COMPARISON:")
            plt.figure(figsize=(16,5))
            
            
            reference_val = [duration, amount, doc_length, doc_readability, doc_richness]

            for i in range(1,6):

                idea = idea_val[i-1]
                ref = reference_val[i-1]
                title = titles[i-1]

                plt.subplot(1, 5, i)
                plt.bar(['Your idea', 'Kickstarter Project'], [idea, ref], width=0.5, color=['royalblue', 'darkorange'])
                plt.gca().set_title(title)
                plt.tick_params(axis='x', labelsize=10, labelrotation=45)
                plt.tick_params(axis='y', labelsize=10, labelrotation=90)

            plt.show()
            
            print("\nFOR MORE DETAILS: {}".format(url))
            print("=" * 117)
            


    
                
            
            


