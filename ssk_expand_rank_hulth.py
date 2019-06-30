import numpy as np
import string
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from shogun.Kernel import SubsequenceStringKernel
from modshogun import StringCharFeatures, RAWBYTE
import time
from textblob import TextBlob
from scipy import sparse
import os
from numpy import random
import re


class ssk_expand_rank_hulth:
    def __init__(self):
        self.english_stopwords = stopwords.words("english")
        self.translator = str.maketrans("", "", string.punctuation)

    def remove_punctuation(self, txt):
        return txt.translate(self.translator)

    def remove_stopwords(self, txt):
        words = word_tokenize(txt)
        return " ".join([w for w in words if w not in self.english_stopwords])

    def get_clean_text(self, txt):
        # convert to lower case
        txt = txt.lower()
        # Remove punctuation
        txt = self.remove_punctuation(txt)
        # Remove stopwords
        return self.remove_stopwords(txt)

    def get_cooccurrence_matrix_inputs(
        self, doc, window_size, doc_no, total_no_of_docs,
        vocabulary, data, row, col
    ):
        for pos, token in enumerate(doc):
            i = vocabulary.setdefault(token, len(vocabulary))
            start = max(0, pos - window_size)
            end = min(len(doc), pos + window_size + 1)
            for pos2 in range(start, end):
                if pos2 == pos:
                    continue
                j = vocabulary.setdefault(doc[pos2], len(vocabulary))
                for t in range(total_no_of_docs):
                    row[t].append(i)
                    col[t].append(j)
                    if t == doc_no:
                        data[t].append(1.0)
                    else:
                        data[t].append(0)
        return vocabulary, data, row, col

    def get_txt(self, path):
        f_handle = open(path)
        txt = f_handle.read()
        txt = re.sub(r'\s+', ' ', txt)
        return txt

    def run_for_all_docs(self):
        path = "automatic_keyphrase_extraction/Hulth2003/Training"
        no_of_docs_for_sample_space = 50
        # k for selecting k-nearest documents
        k = 8
        all_docs_list = []
        all_keywords_list = []
        doc_names = []
        self.doc_scores = []
        for file in os.listdir(path):
            if file.endswith(".abstr"):
                doc_file_path = os.path.join(path, file)
                filename = file.split('.')[0]
                doc_names.append(filename)
                keyword_filename = filename + '.uncontr'
                keywords_file_path = os.path.join(path, keyword_filename)
                all_docs_list.append(self.get_txt(doc_file_path))
                all_keywords_list.append(self.get_txt(keywords_file_path))
        all_docs_list_indices = list(range(len(all_docs_list)))
        doc_index = 0
        for i, doc in enumerate(all_docs_list):
            print("---")
            print("i : ", i)
            print("Doc : ", doc_names[i])
            all_docs_list_indices_dummy = all_docs_list_indices.copy()
            all_docs_list_indices_dummy.remove(i)
            random_indices = random.choice(
                np.array(all_docs_list_indices_dummy),
                no_of_docs_for_sample_space,
                replace=False
            )
            docs_list = [all_docs_list[j] for j in random_indices]
            keywords_list = [all_keywords_list[j] for j in random_indices]
            docs_list.insert(doc_index, doc)
            label = [
                keyword.strip().lower()
                for keyword in all_keywords_list[i].split(';')
            ]
            k_nearest_docs, k_nearest_doc_scores = self.get_k_nearest_docs(
                docs_list, doc_index, k, keywords_list
            )
            vocabulary, cooccurrence_matrix = self.get_cooccurrence_matrix(
                k_nearest_docs, k_nearest_doc_scores, k
            )
            global_affinity_matrix = self.get_global_affinity_matrix(
                k_nearest_doc_scores, cooccurrence_matrix
            )
            word_score_vector = self.get_word_score(
                vocabulary, global_affinity_matrix
            )
            phrases_list = self.extract_phrases(
                doc, vocabulary, word_score_vector, label
            )
            self.calculate_metrics(label, phrases_list)
        avg_doc_score = sum(self.doc_scores) / len(all_docs_list)
        print("Average Doc Score : ", avg_doc_score)

    def get_k_nearest_docs(self, docs_list, doc_index, k, keywords_list):
        cleaned_doc_list = []
        for doc in docs_list:
            cleaned_doc_list.append(self.get_clean_text(doc))
        features = StringCharFeatures(cleaned_doc_list, RAWBYTE)
        # print(dir(features))
        n = 7
        lambda_sym = 0.2
        sk = SubsequenceStringKernel(features, features, n, lambda_sym)
        sim_mat = sk.get_kernel_matrix()
        # print("sim_mat.shape : ", sim_mat.shape)
        similarity_scores_with_indices = [
            (sim_score, i)
            for i, sim_score in list(enumerate(sim_mat[doc_index]))
        ]
        k_nearest_docs_with_indices = sorted(
            similarity_scores_with_indices, reverse=True
        )[: k]
        # print("k_nearest_docs_with_indices : ", k_nearest_docs_with_indices)
        k_nearest_docs = [
            docs_list[i] for score, i in k_nearest_docs_with_indices
        ]
        k_nearest_doc_scores = np.array(
            [score for score, i in k_nearest_docs_with_indices]
        ).reshape(k, 1, 1)
        return k_nearest_docs, k_nearest_doc_scores

    def get_cooccurrence_matrix(self, k_nearest_docs, k_nearest_doc_scores, k):
        w = 10
        allowed_pos = ["JJ", "NN", "NNS", "NNP", "NNPS"]
        vocabulary = {}
        data = [[]] * k
        row = [[]] * k
        col = [[]] * k
        for doc_no, article in enumerate(k_nearest_docs):
            # convert to lower case
            article = article.lower()
            # Remove stopwords
            article = self.remove_stopwords(article)
            # retain words with pos in allowed_pos, exclude punctuation
            doc_with_nouns_and_adj_only = [
                candidate_word[0]
                for candidate_word in pos_tag(word_tokenize(article))
                if candidate_word[1] in allowed_pos
                and candidate_word[0] not in string.punctuation
            ]
            vocabulary, data, row, col = self.get_cooccurrence_matrix_inputs(
                doc_with_nouns_and_adj_only, w, doc_no, k,
                vocabulary, data, row, col
            )
        # print("len(vocabulary) : ", len(vocabulary))
        cooccurrence_matrix = np.zeros(
            shape=(k, len(vocabulary), len(vocabulary))
        )
        for t in range(k):
            cooccurrence_matrix[t] = sparse.coo_matrix(
                (data[t], (row[t], col[t]))
            ).toarray()
        # print("cooccurrence_matrix.shape", cooccurrence_matrix.shape)
        return vocabulary, cooccurrence_matrix

    def get_global_affinity_matrix(
        self, k_nearest_doc_scores, cooccurrence_matrix
    ):
        global_affinity_matrix = np.sum(
            k_nearest_doc_scores * cooccurrence_matrix, axis=0
        )
        # print("global_affinity_matrix.shape : ",
        #       global_affinity_matrix.shape)
        # normalizing global_affinity_matrix along row
        row_sum = global_affinity_matrix.sum(axis=1)
        global_affinity_matrix = (
            global_affinity_matrix / row_sum[:, np.newaxis]
        )
        return global_affinity_matrix

    def get_word_score(self, vocabulary, global_affinity_matrix):
        # damping factor
        mu = 0.85
        vocabulary_length = len(vocabulary)
        word_score_vector_prev = np.ones(vocabulary_length)
        e_vector = np.ones(vocabulary_length)
        word_score_vector = (
            mu * np.matmul(
                word_score_vector_prev, global_affinity_matrix
            ) + ((1 - mu) / vocabulary_length) * e_vector
        )
        while np.sum((word_score_vector - word_score_vector_prev) ** 2) < 0.01:
            word_score_vector = (
                mu * np.matmul(
                    word_score_vector_prev, global_affinity_matrix
                ) + ((1 - mu) / vocabulary_length) * e_vector
            )
            word_score_vector_prev = word_score_vector
        # print("word_score_vector.shape : ", word_score_vector.shape)
        return word_score_vector

    def extract_phrases(self, doc, vocabulary, word_score_vector, label):
        blob = TextBlob(doc.lower())
        doc_phrases = blob.noun_phrases
        phrases_with_scores = []
        for phrase in doc_phrases:
            words_list = word_tokenize(phrase.strip())
            phrases_with_scores.append(
                (
                    sum(
                        [
                            word_score_vector[i]
                            for i in [
                                vocabulary[word]
                                for word in words_list
                                if word in vocabulary.keys()
                            ]
                        ]
                    ),
                    phrase
                )
            )
        phrases_with_scores = sorted(phrases_with_scores, reverse=True)
        m = 10  # no of top phrases to be considered
        phrases_list = []
        for x in phrases_with_scores:
            if x[1] not in phrases_list:
                phrases_list.append(x[1])
            if len(phrases_list) >= m:
                break
        print("===LABEL===")
        print(len(label))
        for x in label:
            print(x)
        print("===TOP PHRASES===")
        for phrase in phrases_list:
            print(phrase)
        return phrases_list

    def calculate_metrics(self, label, phrases_list):
        no_of_positives = 0
        for x in label:
            for y in phrases_list:
                if x in y:
                    no_of_positives += 1
                    break
        if len(phrases_list) > 0:
            score = no_of_positives / len(phrases_list)
        else:
            score = 0
        print("Score : ", score)
        self.doc_scores.append(score)


def run_ssk_expand_rank_hulth():
    ssk_expand_rank_hulth_obj = ssk_expand_rank_hulth()
    ssk_expand_rank_hulth_obj.run_for_all_docs()


start = time.time()
run_ssk_expand_rank_hulth()
end = time.time()
print("Execution time : ", end - start)
