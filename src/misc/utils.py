import json
import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from copy import copy, deepcopy
from torch.nn.utils.rnn import pad_sequence
#--------------------------------------------------------------------------------------------------------------------------------------------
def load_json(file_path: str):
  with open(file_path,'r') as f:
    return json.loads(f.read())
#--------------------------------------------------------------------------------------------------------------------------------------------
  


#--------------------------------------------------------------------------------------------------------------------------------------------  
def squad_json_to_df(file_path: str) -> pd.DataFrame:
  
  # Import the json
  squad_json = load_json(file_path = file_path)
  df = pd.json_normalize(squad_json,
                        record_path= ['data','paragraphs','qas'],#path to the json containing answers {text is the answer and answer_start is the starting pos in the story}
                        meta= [['data','paragraphs','context'],])# The story on which question are asked

  df = df.explode('answers') #Adds a row for each answer {text,answer_start}

  #Split answer into text and answer_start columns
  df[["answer"]] = df[["answers"]].applymap(lambda x: x['text'], na_action='ignore')
  df[["answer_start"]] = df[["answers"]].applymap(lambda x: x['answer_start'], na_action='ignore')
  del df['answers']

  #Adds a row for each plausible answer {text,answer_start}. They are answers when an answer is not possible
  df = df.explode('plausible_answers') 

  #Split plausible answer into text and answer columns
  df[["plausible_answer"]] = df[["plausible_answers"]].applymap(lambda x: x['text'],na_action='ignore')
  df[["plausible_answer_start"]] = df[["plausible_answers"]].applymap(lambda x: x['answer_start'],na_action='ignore')
  del df['plausible_answers']

  # Removing duplicates
  df = df.drop_duplicates()

  # Renaming columns
  # old name = new name
  dict = {'data.paragraphs.context': 'context',
          'is_impossible': 'impossible'
          }
  df.rename(columns=dict,inplace=True)

  df = df.astype({"question": str})
  df = df.astype({"context": str})

  return df
#--------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------
def pad_characters(char_list_batch, padding_idx=None, word_max_len=None):
    """
    'char_list_batch' is a list of lists of lists, in which both sub-lists and sub-sub-lists have different length.
    Each item of the uter list corresponds to a word in a sentence,
    while each nested list corresponds to the characters that word is made of.
    """
    #if word_max_len is not defined, find it from the data 
    if not word_max_len:
        word_max_len = 0
        for batch in char_list_batch: #batch is a list of words (a sentence)
            for word in batch:
                word_max_len = max(word_max_len, len(word))

    #extend each word (list of char indexes to have the same length)
    for i, batch in enumerate(char_list_batch): #batch is a list of words (a sentence)
        for word in batch:
            word.extend([padding_idx] * (word_max_len - len(word)))
        #convert the batch in a torch tensor
        char_list_batch[i] = torch.tensor(batch)

    #now pad each batch element to have the same size (i.e., number of words)
    char_list_batch = pad_sequence(char_list_batch, batch_first=True, padding_value=padding_idx)
    
    return char_list_batch
#----------------------------------------------------------------------------------------------------------------------------------------------------

def pad_pos_indexes(pos_indexes_batch, padding_idx, pos_indexes_max_len=None):
    if not pos_indexes_max_len:
        pos_indexes_max_len = 0
        for indexes in pos_indexes_batch: #batch is a list of indexes
                pos_indexes_max_len = max(pos_indexes_max_len, len(indexes))
    
    for i, indexes in enumerate(pos_indexes_batch):
        indexes.extend([padding_idx] * (pos_indexes_max_len - len(indexes)))#
        pos_indexes_batch[i] = torch.tensor(indexes)
    
    return pos_indexes_batch


#----------------------------------------------------------------------------------------------------------------------------------------------------
def GloVe_embedding_matrix(glove_file, special_tokens=None):
    #load the file, save each row as a list in which the first item is the word and the
    #others are the vector values
    print("Parsing GloVe file....")
    glove_matrix = []
    try:
        with open(glove_file, "r",encoding='utf8') as f:
            #parse each line
            for line in tqdm(f):
                line = line.replace(r"\n", "")
                line = line.split()
                glove_matrix.append([float(value) for value in line[1:]])
    except Exception as e:
        print(e)

    # print(len(glove_matrix))

    embedding_dim = len(glove_matrix[0])

    ### Create the embedding matrix
    if special_tokens:
        embedding_matrix = torch.cat((torch.rand((len(special_tokens), embedding_dim)),
                                    torch.tensor(glove_matrix)), dim=0)
    else:
        embedding_matrix = torch.tensor(glove_matrix)
    
    return embedding_matrix
#----------------------------------------------------------------------------------------------------------------------------------------------------
