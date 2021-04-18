from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# Commented out IPython magic to ensure Python compatibility.
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
# %matplotlib inline
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

import sys
assert sys.version_info.major == 3
from nltk.corpus import stopwords 

project_name = "MyMuseum"
net_id = "Madeleine Chang: mmc337, Shefali Janorkar: skj28, Esther Lee: esl86, Yvette Hung: yh387, Tiffany Zhong: tz279"

def tokenize(text):
  temp = re.split('[^a-z]', text.lower())
  words = []
  for w in temp:
    if w != "": words.append(w)
  return words

#load traveler ratings
rating_file = open("traveler_rating_USonly.json")
rating_loaded = json.load(rating_file)
ratings = {}
for r in rating_loaded:
	ratings[r] = []
	for i in rating_loaded[r][::-1]:
		s = i.replace(',', '')
		ratings[r].append(int(s))

#create list of museums and inverted
museums = list(ratings.keys())
inv_museums = {m:v for (v,m) in enumerate(museums)}

#create a TFIDF matrix
def already_tok(d):
	return d

# key = museum, value = index
museum_to_index = {}

i = 0
for m in museums:
	museum_to_index[m] = i
	i+=1

# key = index, value = museum
index_to_museum = {v:k for k,v in museum_to_index.items()}

# get cosine similarity
def get_cos_sim(mus1, mus2, input_doc_mat, 
								museum_to_index=museum_to_index):
	#cos_sim = 0
	#q_array = input_doc_mat[museum_to_index[mus1]]
	#d_array = input_doc_mat[museum_to_index[mus2]]
	#
	#num = np.dot(q_array, d_array)
	#
	#denom = np.linalg.norm(q_array)  * np.linalg.norm(d_array)    
	# ADDED 1, does it affect the result?
	#cos_sim = num / (denom + 1)
	#		
	#return cos_sim

  v1 = input_doc_mat[museum_to_index[mus1]]
  v2 = input_doc_mat[museum_to_index[mus2]]
  vec1 = np.array(v1)
  vec2 = np.array(v2)
  
  normvec1 = np.linalg.norm(vec1)
  normvec2 = np.linalg.norm(vec2)
  
  n = np.dot(vec1, vec2)
  m = np.dot(normvec1, normvec2)
  
  return n/(m+1)

# construct cosine similarity matrix
def build_museum_sims_cos(num_museums, input_doc_mat, index_to_museum=index_to_museum, museum_to_index=museum_to_index, input_get_sim_method=get_cos_sim):
	#sim_mat = np.zeros((num_museums, num_museums))
	#for i in (index_to_museum):
	#		for j in (index_to_museum):
	#				sim_mat[i][j] = input_get_sim_method(index_to_museum[i], index_to_museum[j], input_doc_mat)
	#		
	#return sim_mat

	cosmat = np.zeros((len(input_doc_mat), len(input_doc_mat)))
	for i in range(len(input_doc_mat)):
		for j in range(len(input_doc_mat)):
			if (i==j): cosmat[i][j] = 1.0
			else:
				mus1 = index_to_museum[i]
				mus2 = index_to_museum[j]
				cosmat[i][j] = input_get_sim_method(mus1, mus2, input_doc_mat, museum_to_index)
	return cosmat

# find top n museums
def get_top_n(museum, n, cosine_mat):
	n = n + 1
	museum_index = museum_to_index[museum]
	museum_row = cosine_mat[museum_index]
	top_n_temp = np.argpartition(museum_row, -n)[-n:]
	top_n_temp = top_n_temp[np.argsort(museum_row[top_n_temp])][::-1]
	top_n = []
	for i in top_n_temp:
		if(i != museum_index):
			top_n.append(i)
	return top_n

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query

	# query = "dinosaur"

		tok_query = tokenize(query)

		#print(museums[0])
		#print(inv_museums["Chicago Children's Museum"])

		#load tags
		tag_clouds_file = open("tag_clouds_USonly.json")
		tag_clouds_file_loaded = json.load(tag_clouds_file)
		tags = {}
		for r in tag_clouds_file_loaded:
			tags[r] = []
			for i in tag_clouds_file_loaded[r]:
				s = i.replace(',', '')
				s=s.lower()
				s = re.sub(r'[^\w\s]', '', s)
				tags[r].append(s)

		#print(tags["Chicago Children's Museum"])

		#load review titles
		review_quote_file = open("review_quote_USonly.json")
		review_quote_file_loaded = json.load(review_quote_file)
		review_titles = {}
		for r in review_quote_file_loaded:
			review_titles[r] = []
			for i in review_quote_file_loaded[r]:
				s = i.replace(',', '')
				s=s.lower()
				s = re.sub(r'[^\w\s]', '', s)
				review_titles[r].append(s)

		#print(review_titles["The Mariners' Museum & Park"])

		#load review content
		review_content_file = open("review_content_USonly.json")
		review_content_file_loaded = json.load(review_content_file)
		review_content = {}
		for r in review_content_file_loaded:
			review_content[r] = []
			for i in review_content_file_loaded[r]:
				s = i.replace(',', '')
				s=s.lower()
				s=re.sub('\n','',s)
				s=re.sub('\xa0','',s)
				s = re.sub(r'[^\w\s]', '', s)
				review_content[r].append(s)
		#print(review_content["The Mariners' Museum & Park"])

		# #clear review titles of questions
		# for p in review_titles:
		#   for q in review_titles[p]:
		#     if "?" in q: review_titles[p].remove(q)

		#print(review_titles["The Mariners' Museum & Park"])

		all_stopwords = stopwords.words('english')

		tok_tags = {}
		for m in museums:
			tok_tags[m] = []
			for t in tags[m]:
				for i in tokenize(t):
						if i not in all_stopwords:
								tok_tags[m].append(i)

		# tokenize review content
		tok_review = {}
		for m in museums:
			tok_review[m] = []
			for t in review_content[m]:
				for i in tokenize(t):
						if i not in all_stopwords:
							tok_review[m].append(i)

		# possibly clean up the tokens

		#create dict with all info for each museum
		museum_info = {}
		for m in museums:
			museum_info[m] = {'ratings': ratings[m], 'tags': tags[m], 'tokenized tags': tok_tags[m], 'review titles': review_titles[m], 'review content': review_content[m], 'tokenized content': tok_review[m]}

		# print(museum_info["Gettysburg Heritage Center"])
		l = len(museum_info)
		museum_info[query] = {'ratings': [1, 1, 1, 1, 1], 'tags': tok_query, 'tokenized tags': tok_query, 'review titles': tok_query, 'review content': tok_query, 'tokenized content': tok_query}
		museums.append(query)
		museum_to_index[query] = l
		index_to_museum[l] = query

		# min df originally 10
		tfidf_vec = TfidfVectorizer(min_df = 1, max_df = 0.8, max_features = 5000, analyzer = "word", tokenizer = already_tok, preprocessor = already_tok, token_pattern=None)

		# V rough algo
		# Find the top museums based on tags and reviews
		# find cosine similarity matrices: one for tags and one for reviews (TEMP, may switch to rocchio)
		# multiply 2 cosine similarity matrices: one for tokenized tags and reviews (what about review title?)
		# this code block takes quite a while to run, optimize it?


		# tf-idf matrices
		tfidf_mat_tags = tfidf_vec.fit_transform(museum_info[m]['tokenized tags'] for m in museums).toarray()
		tfidf_mat_reviews = tfidf_vec.fit_transform(museum_info[m]['tokenized content'] for m in museums).toarray()

		# cosine matrices
		# should I add 1 to these matrices?
		num_museums = len(museums)
		tags_cosine = build_museum_sims_cos(num_museums, tfidf_mat_tags)
		reviews_cosine = build_museum_sims_cos(num_museums, tfidf_mat_reviews)


		# higher = similar
		# tags and reviews weighted equally here, but can be changed
		multiplied = np.multiply(tags_cosine, reviews_cosine)

		top_5 = get_top_n(query, 5, multiplied)

		# TODO
		# 1. If time allows, narrow down location based on latitude and longitude
		# 2. Not include query

		# print("Top 20 Matches for The Mariners' Museum & Park\n")
		# i = 1
		# for t in top_5:
		# 	print(str(i) + ': ' + index_to_museum[t])
		# 	i+=1
		top_5_museums = []
		for i in top_5:
			top_5_museums.append(index_to_museum[i])

		data = top_5_museums


		del museums[-1]
		del museum_info[query]
		del museum_to_index[query]
		del index_to_museum[l]

	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



