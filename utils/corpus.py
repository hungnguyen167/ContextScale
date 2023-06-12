import nltk
import numpy as np
import math
from scipy import spatial
import time
from sys import stdin
from datetime import datetime

class Corpus(object):
	"""description of class"""
	def __init__(self, documents, docpairs = None):
		print("Loading corpus, received: " + str(len(documents)) + " docs.")
		self.docs_raw = [d[1] for d in documents]
		self.docs_names = [d[0] for d in documents]
		self.punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", u"``", "''", "\"", "'", "-", "$" ]
		self.doc_pairs = docpairs
		self.results = {}

	def tokenize(self, stopwords = None, freq_treshold = 5):
		self.stopwords = stopwords
		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Preprocessing corpus...", flush = True)
		self.docs_tokens = [[tok.strip() for tok in nltk.word_tokenize(doc) if tok.strip() not in self.punctuation and len(tok.strip()) > 2] for doc in self.docs_raw]
		#self.docs_tokens = [[tok.strip() for tok in nltk.word_tokenize(doc)] for doc in self.docs_raw]

        
		self.freq_dicts = []
		if self.stopwords is not None:
			for i in range(len(self.docs_tokens)):
				self.docs_tokens[i] = [tok.strip() for tok in self.docs_tokens[i] if tok.strip().lower() not in self.stopwords]	
					
	def build_occurrences(self):
		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building vocabulary...", flush = True)
		self.vocabulary = {} 
		for dt in self.docs_tokens:
			for t in dt:
				if t not in self.vocabulary:
					self.vocabulary[t] = len(self.vocabulary)

		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building coocurrence matrix...", flush = True)
		self.occurrences = np.ones((len(self.docs_tokens), len(self.vocabulary)), dtype = np.float32)
		cnt = 0
		for i in range(len(self.docs_tokens)):
			cnt += 1
			print(str(cnt) + "/" + str(len(self.docs_tokens)))
			for j in range(len(self.docs_tokens[i])):
				word = self.docs_tokens[i][j]
				self.occurrences[i][self.vocabulary[word]] += 1
		if np.isnan(self.occurrences).any():
			raise ValueError("NaN in self.occurrences")

	def set_doc_positions(self, positions):
		for i in range(len(self.docs_names)):
			self.results[self.docs_names[i]] = positions[i]