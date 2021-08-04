#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Load libraries
import re
import time
import operator

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
import pandas as pd
import time 

import sparse_dot_topn.sparse_dot_topn as ct
low_memory=False
import sys
import os


# In[32]:


class StringMatch():
    '''
    The StringMatch class, which cleans the data and performs the fast fuzzy merge
    '''
    
    def __init__(self, source_names, target_names):
        self.source_names = source_names
        self.target_names = target_names
        self.ct_vect      = None
        self.tfidf_vect   = None
        self.vocab        = None
        self.sprse_mtx    = None
        
        
    def tokenize(self, analyzer='char_wb', n=3):
        '''
        Tokenizes the list of strings, based on the selected analyzer

        :param str analyzer: Type of analyzer ('char_wb', 'word'). Default is trigram
        :param str n: If using n-gram analyzer, the gram length
        '''
        # Create initial count vectorizer & fit it on both lists to get vocab
        self.ct_vect = CountVectorizer(analyzer=analyzer, ngram_range=(n, n))
        self.vocab   = self.ct_vect.fit(self.source_names + self.target_names).vocabulary_
        
        # Create tf-idf vectorizer
        self.tfidf_vect  = TfidfVectorizer(vocabulary=self.vocab, analyzer=analyzer, ngram_range=(n, n))
        
        
    def match(self, ntop=1, lower_bound=0, output_fmt='df'):
        '''
        Main match function. Default settings return only the top candidate for every source string.
        
        :param int ntop: The number of top-n candidates that should be returned
        :param float lower_bound: The lower-bound threshold for keeping a candidate, between 0-1.
                                   Default set to 0, so consider all canidates
        :param str output_fmt: The output format. Either dataframe ('df') or dict ('dict')
        '''
        self._awesome_cossim_top(ntop, lower_bound)
        
        if output_fmt == 'df':
            match_output = self._make_matchdf()
        elif output_fmt == 'dict':
            match_output = self._make_matchdict()
            
        return match_output
        
        
    def _awesome_cossim_top(self, ntop, lower_bound):
        '''  '''
        # To CSR Matrix, if needed
        A = self.tfidf_vect.fit_transform(self.source_names).tocsr()
        B = self.tfidf_vect.fit_transform(self.target_names).transpose().tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        self.sprse_mtx = csr_matrix((data,indices,indptr), shape=(M,N))
    
    
    def _make_matchdf(self):
        ''' Build dataframe for result return '''
        # CSR matrix -> COO matrix
        cx = self.sprse_mtx.tocoo()

        # COO matrix to list of tuples
        match_list = []
        for row,col,val in zip(cx.row, cx.col, cx.data):
            match_list.append((row, self.source_names[row], col, self.target_names[col], val))

        # List of tuples to dataframe
        colnames = ['Row Idx', 'Title', 'Candidate Idx', 'Candidate Title', 'Score']
        match_df = pd.DataFrame(match_list, columns=colnames)

        return match_df

    
    def _make_matchdict(self):
        ''' Build dictionary for result return '''
        # CSR matrix -> COO matrix
        cx = self.sprse_mtx.tocoo()

        # dict value should be tuple of values
        match_dict = {}
        for row,col,val in zip(cx.row, cx.col, cx.data):
            if match_dict.get(row):
                match_dict[row].append((col,val))
            else:
                match_dict[row] = [(col, val)]

        return match_dict   


# In[56]:


os.chdir(r"C:\Users\rcedxk23\Downloads\2003")

directory = r"C:\Users\rcedxk23\Downloads\2003" 
filelist = os.listdir(directory)
print(filelist)


# In[82]:


os.chdir(r"C:\Users\rcedxk23\Downloads\2021")

directory = r"C:\Users\rcedxk23\Downloads\2021" 
#code directory, change between datasets
filelist = os.listdir(directory)
print(filelist)
final_name_list = ['2021_1_match', '2021_2_match', '2021_3_match', '2021_4_match']
#merge file names as needed, also change between datasets 


t1 = time.time()

file_length = len(filelist)

for i in range(file_length):
    os.chdir(r"C:\Users\rcedxk23\Downloads\2021")
    #contract data set 
    contract_data = pd.read_csv(filelist[i])
    #convert recipient_parent_name to a string 
    contract_data['recipient_parent_name'] = contract_data['recipient_parent_name'].map(str)
    #clean dataset, the compustat data
    os.chdir(r"C:\Users\rcedxk23\Downloads")
    clean = pd.read_csv('2021_compustat.csv')
    #make sure company names are strings 
    clean['conm'] = clean['conm'].map(str)
    
    #make the contract data into a list 
    contract_list = contract_data.recipient_parent_name.tolist()
    #make the company data into a list 
    clean_list = clean.conm.tolist()
    
    #create an instance of the string match class with the messy data list and the clean data list 
    #namematch = StringMatch(clean_list, contract_list)
    namematch = StringMatch(contract_list, clean_list)
    #make the tokens of the names 
    namematch.tokenize()
    #create the final dataframe 
    #check the time
  
    match_df = namematch.match()
    
    path = r"C:\Users\rcedxk23\Downloads\2021_matches"
    output_file = os.path.join(path,f'{final_name_list[i]}.csv')

    match_df.to_csv(output_file, index=False)
    #save the match df as a csv 

    
t = time.time()-t1
print("SELFTIMED:", t)


# In[ ]:





# In[10]:


# #make the contract data into a list 
# contract_list = contract_data.recipient_parent_name.tolist()
# #make the company data into a list 
# clean_list = clean.conm.tolist()


# In[14]:


# #create an instance of the string match class with the messy data list and the clean data list 
# #namematch = StringMatch(clean_list, contract_list)
# namematch = StringMatch(contract_list, clean_list)
# #make the tokens of the names 
# namematch.tokenize()
# #create the final dataframe 
# #check the time
# t1 = time.time()
# match_df = namematch.match()
# t = time.time()-t1
# print("SELFTIMED:", t)


# In[20]:


match_df


# In[19]:


#save the match df as a csv 
match_df.to_csv("hi.csv")


# In[ ]:




