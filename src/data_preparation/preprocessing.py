"""
For what concerns text preprocessing we first `lowercased` all the words in the dataset.
We also decided to remove `special characters` (i.e., punctuation, emoji, ...) and `extra spaces` from contexts and questions.
However, given the task of retrieving answer from the context, we decided not to remove `stop words` and `digits`.
Indeed they can be part of the answer and they can also provide to the model useful insights, especially for what regards the questions (e.g., why, where, who, ...).
An important note is that while processing the text we kept track of the indexes of all the characters in the original context
in order to be able to recontruct the predicted answer from the start and end indexes provided as model outputs.

To this aim we implemented custom preprocessing functions for each each preprocessing step. The functioning of such functions is resumed in the following example:
```
          context = Hi. World
remaining_context = [123456789]
```
If the preprocessing step removes '.' from the context we also remove the index 3 from the remaining_context obtaining
```
          context = Hi World
remaining_context = [12456789]
```

Following the work by Chen et al. https://aclanthology.org/P17-1171.pdf, we used the `en_core_web_sm` pre-trained English language model from SpaCy to perform tokenization and to enrich the dataset with additional linguistic features from text. In particular, for each token, we decided to extract the following:
- Lemma
- Part-of-speech (`POS`) tag
- Named entity (`ENT`) tag <br>

The choice of using lemmas instead of simple tokens is to have a better correspondence with the words in the pre-trained embeddings of GloVe. Moreover, the rationale behind combining them to additional linguistic features is to enhance the model performances in terms of text comprehension, providing extra information about phrashes structure and syntax.

Finally, we store the original context, questions and answers, the associated lemmas, `POS` and `ENT` tags, the start and end points of answer in the context, the position of each lemma in the original context, and many other auxiliary features in separated dataframes for training, validation and test.
        
"""
import ast
import json
import math
import re
import string

import pandas as pd
import spacy
from spacy.lang.en import English
from tqdm.notebook import tqdm

nlp = English()

class DataLocations:
    """Utiliy class to save location of the initial and final data"""
    def __init__(self, init, mid, final, is_test, use_full_spacy_pipe):
        self.init = init
        self.mid = mid
        self.final = final
        self.is_test = is_test
        self.use_full_spacy_pipe = use_full_spacy_pipe

class Colour:
   """Colors used to print in the terminal"""
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[33m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class NLP_preprocessing:
    """Contains functions to preprocess the dataset"""
    dtypes = {
        'question': 'str',
        'lemmas': 'object',
        'id': 'str',
        'context':'str',
        'original_context':'str',
        'answer': 'str',
        'plausible_answer': 'str',
        'answer_start': 'float64',
        'plausible_answer_start': 'float64',
        'origin_context':'str'}


    def remove_char_list(self,edit_column, remaining_chars, char_set):
        rchars = remaining_chars.copy()
        to_del = []
        new_edit_column = ""
        for i, c in enumerate(edit_column):
            if c in char_set:
                to_del.append(i)
            else:
                new_edit_column += c
        for index in sorted(to_del, reverse=True):
            del rchars[index]
        return new_edit_column, rchars


    def strip_start_end(self,edit_column, remaining_chars):
        stripped = edit_column.strip()
        start = edit_column.find(stripped)
        end = start+len(stripped)
        return stripped, remaining_chars[start:end]

    def remove_char(self,edit_column,remaining_chars,ch):
        rchars = remaining_chars.copy()
        to_del = []
        new_edit_column = ""
        for i, c in enumerate(edit_column):
            if c == ch:
                to_del.append(i)
            else:
                new_edit_column += c
            
        for index in sorted(to_del, reverse=True):
            del rchars[index]
        return new_edit_column,rchars

    def remove_space_and_char(self,edit_column,remaining_chars):
        rchars = remaining_chars.copy()
        new_edit_column = ""
        to_del = []
        # Loop over each character in the string
        for i in range(len(edit_column)):
            # If the character is not a space or it is the first character of the string,
            # append it to the result string
            if edit_column[i] != " " or i == 0 or edit_column[i-1] != " ":
                new_edit_column += edit_column[i]
            # If the character is a space and the previous character was a space,
            # append a single space to the result string
            elif edit_column[i-1] == " ":
                to_del.append(i-1)
                
        for index in sorted(to_del, reverse=True):
            del rchars[index]
        return new_edit_column,rchars

    def lower_same_size(self,edit_column):
        r"""The loweing function changes the string lenght in some cases eg. İ->i̇ so we keep the lenght the same.
        i̇ occupies more than one space in this example"""
        new_edit_column = ""
        for i, c in enumerate(edit_column):
            new_edit_column += c.lower()[0]
        return new_edit_column

    def find_closest_lower_number(self,num, lst):
        left, right = 0, len(lst) - 1
        closest = None
        while left <= right:
            mid = (left + right) // 2
            if lst[mid] < num:
                closest = lst[mid]
                left = mid + 1
            else:
                right = mid - 1
        return closest,left


    def find_start_token(self,original_start,starts):
        r"""
        Finds the token that matches the start position in the original context,
        - if the start is not the first char of a word returns -2
        """
        if math.isnan(original_start):
            return float('nan')
        else:
            original_start = int(original_start)
        try:
            return starts.index(original_start)
        except Exception as e:
            return -2 


    def find_end_token(self,original_end,ends,answer):
        r"""
        Finds the token that matches the end position in the original context,
        - if the end is the last char of a word or inside the word returns the word
        - if the end is on a space it will return the previous word       
        """
        if math.isnan(original_end):
            return float('nan')
        else:
            original_end = int(original_end)-1
        try:
            return ends.index(original_end)
        except Exception as e:
            value,index = self.find_closest_lower_number(original_end,ends)
            return index

    def compute_end(self,start, answer):
        if math.isnan(start):
            return float('nan') 
        else:
            return start+len(answer)

    def correct_pos(self,original_char_pos,preprocessed_pos):
        retpos = []
        for p in preprocessed_pos:
            retpos.append(original_char_pos[p])
        return [original_char_pos[p] for p in preprocessed_pos]

    def tryload(self,val):
        try:
            loads = json.loads(str(val.replace("'",'"')))
            if isinstance(loads, list):
                return loads
        except:
            return 0
        
    def preprocess_before_lemmatization(self,data):
        data = data.reset_index(drop=True)
        data.drop('Unnamed: 0', axis=1, inplace=True)
        #Create list of caracters not deleted from context
        data['remaining_context'] = data.context.progress_apply(lambda x:list(range(len(x))))
        data['origin_context'] = data.context
        #lower and remove quotes
        data['question'] = data.question.apply(lambda x: re.sub("'", '', x).lower())
        data[['context','remaining_context']] = data[['context','remaining_context']].progress_apply(lambda x: self.remove_char(*x,"'"), axis=1,result_type="expand")
        data['context'] = data.context.progress_apply(lambda x: self.lower_same_size(x))
        data['answer'] = data.answer.str.lower()
        # The other columns contain Nans so they have to be treated differently
        data['answer'] = data.answer.str.replace("'", '')
        data['answer'] = data.answer.str.lower()
        data['plausible_answer'] = data.plausible_answer.str.replace("'", '')
        data['plausible_answer'] = data.plausible_answer.str.lower()

        # In english there are many composite words separated by '-' (e.g., baby-sitter). Replace '-' with a space
        data['question'] = data.question.apply(lambda x: re.sub("-", ' ', x))
        data['context'] = data.context.apply(lambda x: re.sub("-", ' ', x))
        # The other columns contain Nans so they have to be treated differently
        data['answer'] = data.answer.str.replace("-", ' ')
        data['plausible_answer'] = data.plausible_answer.str.replace("-", ' ')
        #remove special chars
        exclude = set(string.punctuation) #set of all special chars

        data['question'] = data.question.apply(lambda x: ''.join(ch for ch in x if ch not in exclude) if isinstance(x, str) else None)
        # print(len(data['context'].values[0]),data['context'].values[0])
        data[['context','remaining_context']] = data[['context','remaining_context']].progress_apply(lambda x: self.remove_char_list(*x,exclude), axis=1,result_type="expand")
        # print(len(data['context'].values[0]),data['context'].values[0])
        data['answer'] = data.answer.apply(lambda x: ''.join(ch for ch in x if ch not in exclude) if isinstance(x, str) else None)
        data['plausible_answer'] = data.plausible_answer.apply(lambda x: ''.join(ch for ch in x if ch not in exclude) if isinstance(x, str) else None)

        # remove extra spaces
        data['question'] = data.question.apply(lambda x: x.strip() if isinstance(x, str) else None)
        data['question'] = data.question.apply(lambda x: re.sub(" +", " ", x) if isinstance(x, str) else None)
        data[['context','remaining_context']] = data[['context','remaining_context']].progress_apply(lambda x: self.strip_start_end(*x), axis=1,result_type="expand")
        data[['context','remaining_context']] = data[['context','remaining_context']].progress_apply(lambda x: self.remove_space_and_char(*x), axis=1,result_type="expand")
        data['answer'] = data.answer.apply(lambda x: x.strip() if isinstance(x, str) else None)
        data['answer'] = data.answer.apply(lambda x: re.sub(" +", " ", x) if isinstance(x, str) else None)
        data['plausible_answer'] = data.plausible_answer.apply(lambda x: x.strip() if isinstance(x, str) else None)
        data['plausible_answer'] = data.plausible_answer.apply(lambda x: re.sub(" +", " ", x) if isinstance(x, str) else None)

        return data

    def lemmatized_question(self,data,full_pipe):
        docs = list(data.question.values)
        nlp = spacy.load("en_core_web_sm")
        all_lemmas = []
        all_qpos = []
        all_qent = []
        for doc in tqdm(nlp.pipe(docs,
                                    batch_size=16, 
                                    n_process=2,
                                    disable=["parser", "ner", "textcat"] if not full_pipe else ["parser","textcat"]
                                    ), total=len(docs)): 
            lemmas = []
            pos = []
            ent = []
            for token in doc:
                lemmas.append(token.lemma_)
                if full_pipe:
                    pos.append(token.pos_)
                    ent.append(token.ent_type_ if token.ent_type_ else "")

                all_lemmas.append(lemmas)
                if full_pipe:
                    all_qpos.append(pos)
                    all_qent.append(ent)

        return  all_lemmas,all_qpos,all_qent

    def lemmatized_context(self,data, full_pipe):
        docs = list(data.context.values)
        nlp = spacy.load("en_core_web_sm")
        all_lemmas = []
        all_starts = []
        all_ends = []
        all_pos = []
        all_ent = []
        for doc in tqdm(nlp.pipe(docs,
                                batch_size=16, 
                                n_process=2,
                                disable=["parser", "ner", "textcat"] if not full_pipe else ["parser","textcat"]
                                ), total=len(docs)):
            lemmas = []
            idx_start = []
            idx_end = []
            pos = []
            ent = []
            for token in doc:
                lemmas.append(token.lemma_)
                idx_start.append(token.idx)
                idx_end.append(token.idx + len(token.text) - 1)
                if full_pipe:
                    pos.append(token.pos_)
                    ent.append(token.ent_type_ if token.ent_type_ else "")

            all_lemmas.append(lemmas)
            all_starts.append(idx_start)
            all_ends.append(idx_end)
            if full_pipe:
                all_pos.append(pos)
                all_ent.append(ent)

        return all_lemmas,all_starts,all_ends,all_pos,all_ent

    def preprocess_and_lemmatize(self,data, full_pipe):
        data = self.preprocess_before_lemmatization(data)

        data1 = data[['context']].copy().drop_duplicates() #Removing duplicate context to speed up execution
        all_lemmas,all_starts,all_ends,all_pos,all_ent = self.lemmatized_context(data1,full_pipe)

        data1['lemmas'] = all_lemmas
        data1['original_start_points'] = all_starts
        data1['original_end_points'] = all_ends
        if full_pipe:
            data1['context_pos'] = all_pos
            data1['context_ent'] = all_ent
            data = data.merge(data1,on='context')
            
            data1 = data[['id','question']].copy().drop_duplicates()  #Removing duplicate context to speed up execution

        all_questions, all_qpos, all_qent = self.lemmatized_question(data1,full_pipe)
        data1['question'] = all_questions
        if full_pipe:
            data1['question_pos'] = all_qpos
            data1['question_ent'] = all_qent
            del data['question']
            data = data.merge(data1,on='id')


            data['original_start_points'] = data[['remaining_context', 'original_start_points']].progress_apply(lambda x: self.correct_pos(*x), axis=1)
            data['original_end_points'] = data[['remaining_context', 'original_end_points']].progress_apply(lambda x: self.correct_pos(*x), axis=1)

            return data

    def update_lemmas_start_end(self,dataset_loc):
        #Loading data
        data = pd.read_csv(dataset_loc.mid,dtype=self.dtypes)
        data.drop('Unnamed: 0', axis=1, inplace=True)


        if dataset_loc.is_test:
            #Read the arrays for dev
            data['remaining_context'] = data['remaining_context'].progress_apply(ast.literal_eval)
            data["lemmas"] = data["lemmas"].progress_apply(ast.literal_eval)
            data['original_start_points'] = data['original_start_points'].progress_apply(ast.literal_eval)
            data['original_end_points'] = data['original_end_points'].progress_apply(ast.literal_eval)
        else:
            #Read the arrays for train and val
            data['remaining_context'] = data['remaining_context'].progress_apply(lambda x: json.loads(x))
            data['original_start_points'] = data['original_start_points'].progress_apply(lambda x: json.loads(x))
            data['original_end_points'] = data['original_end_points'].progress_apply(lambda x: json.loads(x))
            data['lemmas'] = data['lemmas'].progress_apply(lambda x: self.tryload(x))
            data = data[data['lemmas']!=0]
            pass

        if dataset_loc.is_test:
            data['context_pos'] = data['context_pos'].progress_apply(ast.literal_eval)
            data['context_ent'] = data['context_ent'].progress_apply(ast.literal_eval)
            data['question_pos'] = data['question_pos'].progress_apply(ast.literal_eval)
            data['question_ent'] = data['question_ent'].progress_apply(ast.literal_eval)

        data = data.astype({'answer': 'str','plausible_answer': 'str'})
        data = data.fillna({'answer': ''})

        #Save original starting and ending positions
        data['original_answer_start'] = data['answer_start']
        data['original_plausible_answer_start'] = data['plausible_answer_start']
        data['original_answer_end'] = data[['original_answer_start','answer']].apply(lambda x: self.compute_end(*x), axis=1)
        data['original_plausible_answer_end'] = data[['original_plausible_answer_start','plausible_answer']].apply(lambda x: self.compute_end(*x), axis=1)

        #Remove answer start and end token
        data['answer_start'] = data[['original_answer_start','original_start_points']].progress_apply(lambda x: self.find_start_token(*x), axis=1)
        data['plausible_answer_start'] = data[['original_plausible_answer_start','original_start_points']].progress_apply(lambda x: self.find_start_token(*x), axis=1)

        data['answer_end'] = data[['original_answer_end','original_end_points','answer']].progress_apply(lambda x: self.find_end_token(*x), axis=1)
        data['plausible_answer_end'] = data[['original_plausible_answer_end','original_end_points','plausible_answer']].progress_apply(lambda x: self.find_end_token(*x), axis=1)

        if not dataset_loc.is_test:
            #Remove answers with start not on the start of a word
            data = data[data['answer_start'] != -2]
            data = data[data['plausible_answer_start'] != -2]

        del data['remaining_context']
        return data


def comparePhrases(data):
    r"""
    Input
    ---
    A dataframe containing preprocessed data

    Output
    ---
    Samples from the dataset in the following format:
    1. Orignal context witout preprocessing
    2. End char of each lemma
    3. Start char of each lemma
    4. lemmas aligned with the original context
    
    example::
        
        .------------------
        | Answer_start: 33, Answer: overhead wires, Token_start: 33
        +-----------------
        | Most electrification systems use overhead wires, but third rail is an option up to about 1,200 V.
        |    3               19      27  31       40    46   51    57   62 65 68     75 78 81    87    9395         
        | 0    5               21      29  33       42     49  53    59   64 67 70     77 80 83    89    95             
        | most electrification system  use overhead wire   but third rail be an option up to about 1200  v                                                                                          
        ------------------
    """
    for index,row in data.iterrows():
        context = row.origin_context
        lemmas = row.lemmas
        starts = row.original_start_points
        ends = row.original_end_points
        slemmas = " "*(len(context)+10)
        sstarts = " "*(len(context)+10)
        sends = " "*(len(context)+10)
        answer = row.answer
        if math.isnan(row.original_answer_start):
            continue
        answer_start = int(row.original_answer_start)
        answer_start_tk = int(row.answer_start)
        answer_end_tk = int(row.answer_end)
        answer_end = int(row.original_answer_end)
        token_start = starts[answer_start_tk]
        token_end = ends[answer_end_tk]+1

        print('.-----------------')
        print(f'| Answer_start: {answer_start}, Answer: {Colour.GREEN+answer+Colour.END}, Token_start: {token_start}')

        
        for i,lemma in enumerate(lemmas):
            lemmas_start_pos = starts[i]
            lemmas_end_pos = ends[i]
            # print(len(starts),len(ends),len(lemmas))
            
            starts_right_pos = lemmas_start_pos + len(str(starts[i]))   
            end_right_pos = lemmas_end_pos + len(str(ends[i]))

            slemmas = slemmas[:lemmas_start_pos]+lemma+slemmas[lemmas_end_pos:]
            sstarts = sstarts[:lemmas_start_pos]+str(starts[i])+sstarts[starts_right_pos:]
            sends = sends[:lemmas_end_pos]+str(ends[i])+sends[end_right_pos:]

            context = context[:answer_start]+Colour.GREEN+context[answer_start:answer_end]+Colour.END+context[answer_end:]
            
            slemmas = slemmas[:token_start]+Colour.GREEN+slemmas[token_start:token_end]+Colour.END+slemmas[token_end:]

            print('+-----------------')
            print('|',context)
            print('|',sends)
            print('|',sstarts)
            print('|',slemmas)
            print('------------------')

def preprocess_all(print_example:bool = False):
    """Loads, preprocess and saves the train/test/validation splits of our dataset"""
    tqdm.pandas()

    train_locs = DataLocations('dataset/train_df.csv', 
                           'dataset/mid_processed_data_train.csv',
                           'dataset/train_data_frame_preprocessed.csv', False, True)

    val_locs = DataLocations('dataset/val_df.csv', 
                            'dataset/mid_processed_data_val.csv',
                            'dataset/val_data_frame_preprocessed.csv', False, True)

    dev_locs = DataLocations('dataset/dev_df.csv',
                            'dataset/mid_processed_data_dev.csv',
                            'dataset/dev_data_frame_preprocessed.csv', True, True)

    preprocessor = NLP_preprocessing() 
    
    ### Get lemmas, POS and ENT tags
    # Load the dataset that we previously saved in the .csv files
    data_train = pd.read_csv(train_locs.init)
    data_val = pd.read_csv(val_locs.init)
    data_test = pd.read_csv(dev_locs.init)

    # Preprocess and spacy pipeline
    data_train = preprocessor.preprocess_and_lemmatize(data_train, train_locs.use_full_spacy_pipe)
    data_val =  preprocessor.preprocess_and_lemmatize(data_val, val_locs.use_full_spacy_pipe)
    data_test = preprocessor.preprocess_and_lemmatize(data_test, dev_locs.use_full_spacy_pipe)

    ### Prepare and save dataframes
    # Save mid processed dataset after applying spacy pipeline
    data_train.to_csv(train_locs.mid)
    data_val.to_csv(val_locs.mid)
    data_test.to_csv(dev_locs.mid)

    # Load the mid processed dataset and update start and end position of each lemma to match the ones in the original context
    data_train = preprocessor.update_lemmas_start_end(train_locs)
    data_val = preprocessor.update_lemmas_start_end(val_locs)
    data_test = preprocessor.update_lemmas_start_end(dev_locs)

    # # save final dataset
    data_train.to_csv(train_locs.final)
    data_val.to_csv(val_locs.final)
    data_test.to_csv(dev_locs.final)

    ### An example of preprocessed data
    if print_example:
        comparePhrases(data_train[0:10])

