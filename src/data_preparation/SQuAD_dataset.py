import pandas as pd
import numpy as np
import os
import json
import string
import torch
import ast
from spacy import glossary
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Iterable, List, Tuple, Dict
from misc.utils import pad_characters, GloVe_embedding_matrix, pad_pos_indexes


#--------------------------------------------------------------------------------------------------------------------------------------------
def get_squad_dataframe(
        df_path: str
) -> pd.DataFrame:
    """
    Load the preprocessed SQuAD 2.0 dataset together with additional features (POS tags, ENT).
    Gather all these data and return them in a Pandas dataframe. 

    Parameters:
    -----------
    df_path: (str)
        The path where the preprocessed dataframe is stored.

    Returns:
    --------
    out_df: (pd.Dataframe)
        A dataframe containing the preprocessed SQuAD 2.0 data and the additional features.
    """
    # load the preprocessed dataset
    df = pd.read_csv(df_path)

    df = df.astype({'answer':           'str',
                    'plausible_answer': 'str',
                    'id':               'str',
                    'origin_context':   'str',
                    'impossible':       'bool'})

    #retain only the needed columns
    to_keep = [
        'id', 'impossible', 'question', 'origin_context', 'lemmas', 
        'answer', 'answer_start', 'answer_end', 'context',
        'plausible_answer', 'plausible_answer_start', 'plausible_answer_end',
        'original_start_points', 'original_end_points',
        'context_pos', 'context_ent', 'question_pos', 'question_ent'
    ]
    out_df = df[to_keep].copy()

    #rename some columns to avoid confusion
    out_df["preprocessed_context"] = out_df["context"].copy()
    out_df = out_df.drop(columns="context")
    out_df["context"] = out_df["lemmas"].copy()
    out_df = out_df.drop(columns="lemmas")

    tqdm.pandas(desc='1/8 loading questions')
    out_df['question'] = out_df['question'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='2/8 loading contexts')
    out_df["context"] = out_df["context"].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='3/8 loading original_start_points')
    out_df['original_start_points'] = out_df['original_start_points'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='4/8 loading original_end_points')
    out_df['original_end_points'] = out_df['original_end_points'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='5/8 loading contexts POS tags')
    out_df['context_pos'] = out_df['context_pos'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='6/8 loading questions POS tags')
    out_df['question_pos'] = out_df['question_pos'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='7/8 loading context ENT tags')
    out_df['context_ent'] = out_df['context_ent'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas(desc='8/8 loading question ENT tags')
    out_df['question_ent'] = out_df['question_ent'].progress_apply(lambda x: ast.literal_eval(x))
    tqdm.pandas()
    
    entities = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",     "ORDINAL", "CARDINAL"]
    ent_to_idx = dict(zip(entities, np.arange(3, len(entities) + 3)))
    
    #group the pos indexes according to the tag value
    def get_pos_indexes(lst):
        index_dict = {}
        for i, value in enumerate(lst):
            if value not in index_dict:
                index_dict[value] = [i]
            else:
                index_dict[value].append(i)
        return index_dict
    
    def get_ent_indexes(lst): # ['<PAD>','<OOV>', '<SOS>']
        ent_indexes = [2] 
        for ent in lst:
            if ent == "":
                ent_indexes.append(1)
            else:
                ent_indexes.append(ent_to_idx[ent])
        return ent_indexes
    
    out_df['context_pos_grouped_idxs'] = out_df['context_pos'].progress_apply(lambda x: get_pos_indexes(x))
    out_df['question_pos_grouped_idxs'] = out_df['question_pos'].progress_apply(lambda x: get_pos_indexes(x))
    
    out_df['context_ent_idxs'] = out_df['context_ent'].progress_apply(lambda x: get_ent_indexes(x))
    out_df['question_ent_idxs'] = out_df['question_ent'].progress_apply(lambda x: get_ent_indexes(x))
    
    return out_df
#--------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------
class Vocabulary:
    """
    A `Vocabulary` object stores information about the alphabet and the special tokens considered for the task.
    In addition, it computes the word frequencies in the given text data, allowing removal of unusual ones.
    Finally, it provides mapping from word and single characters to embedding indexes (e.g., GloVe).
    """
    def __init__(
            self,
            special_tokens: Optional[Iterable] = None,
            alphabet:  Optional[Iterable] = None
    ):
        '''
        Parameters:
        -----------
        special tokens: (Optional[Iterable], default=['<PAD>', '<OOV>', '<SOS>'])
            a collection of the special tokens to include in the vocabulary. 
        
        alphabet: (Optional[Iterable], default=[string.ascii_lowercase, string.digits])
            a collection of the characters to include in the character vocabulary.
        '''

        #check that the user has inserted at least the necessary special tokens
        #if not add them
        if alphabet is None:
            alphabet = string.ascii_lowercase + string.digits
        if special_tokens is None:
            special_tokens = ['<PAD>','<OOV>', '<SOS>']
        necessary_special_tokens = ['<PAD>','<OOV>', '<SOS>']
    
        if special_tokens is None:
            special_tokens = necessary_special_tokens
        else:
            for st in necessary_special_tokens:
                if st not in special_tokens:
                    special_tokens.append(st)
        self.special_tokens = special_tokens

        #initiate a dictionary for the word & character frequencies
        self.word_frequencies = {}
        self.char_frequencies = {}
        
        #initiate a dictionary that maps words/characters to GloVe indexes (and viceversa)
        # n.b. the index refer to the row in the Glove embedding matrix associated to that word
        self.word_to_idx = {}
        self.idx_to_word = {}

        #initiate a dictionary that maps characters to indexes in the embedding matrix (and viceversa)
        #n.b. this dictionary is fixed and given by the alphabet we are using
        self.char_to_idx = {'<PAD>': 0,'<OOV>': 1,'<SOS>': 2,'<EOS>': 3}
        self.char_to_idx.update(dict(zip(alphabet, np.arange(4, len(alphabet)+4))))
        self.idx_to_char = {idx: char for char, idx in enumerate(self.char_to_idx)}


    def __len__(self):
        return len(self.word_frequencies)
    

    def __getitem__(self, idx):
        return list(self.word_to_idx.keys())[idx]
    

    def build_word_vocabulary(
            self, 
            sentence_list: List[List[str]], 
            output_dir: Optional[str] = None
    ) -> None:
        """
        Create a dictionary whose keys are the words contained in the sentence_list and the 
        values are the associated frequencies/counts.

        Parameters:
        -----------
        sentence_list: (List[List[str]])
            A list of sentences, in which each sentence is represented as a list of words. 

        output_dir: (Optional[str] = None)
            The path to the output directory.
        """
        for sentence in tqdm(sentence_list, desc="Building word vocabulary"):
            for word in sentence:
                self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1

        #save the global vocabulary with the frequencies
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, "word_vocabulary.json"), 'w', encoding='UTF-8') as f:
                json.dump(self.word_frequencies, f)


    def build_char_vocabulary(
            self, 
            sentence_list: List[List[str]], 
            output_dir: Optional[str] = None
    ) -> None:
        """
        Create a dictionary whose keys are the characters contained in the words in sentence_list and the 
        values are the associated frequencies/counts.

        Parameters:
        -----------
        sentence_list: (List[List[str]])
            A list of sentences, in which each sentence is represented as a list of words. 

        output_dir: (Optional[str] = None)
            The path to the output directory.
        """
        for sentence in tqdm(sentence_list, desc="Building char vocabulary"):
            for word in sentence:
                for char in word:
                    self.char_frequencies[char] = self.char_frequencies.get(char, 0) + 1

        #save the global vocabulary with the frequencies
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, "char_vocabulary.json"), 'w', encoding='UTF-8') as f:
                json.dump(self.char_frequencies, f)


    def load_word_vocabulary(
            self, 
            path: str
    ) -> None:
        """
        Load pre-computed word frequency dictionary. 

        path: (str)
            The path to the vocabulary (word_frequencies) file.
        """
        try:
            with open(path, 'r') as f:
                self.word_frequencies = json.load(f)
        except:
            raise FileNotFoundError("No such file found at {}".format(path))
    

    def load_char_vocabulary(self, path):
        """
        Load pre-computed char frequency dictionary. 
        
        path: (str)
            The path to the vocabulary (char_frequencies) file.
        """
        try:
            with open(path, 'r') as f:
                self.char_frequencies = json.load(f)
        except:
            raise FileNotFoundError("No such file found at {}".format(path))
        
    
    def reduce_word_vocabulary(
            self, 
            max_size: int,
            freq_threshold: int, 
    ) -> None:
        """
        Reduce the current size of the Vocabulary to `max_size`, removing less frequent words.
        
        Parameters:
        -----------
        max_size: (int)
            If vocabulary size is greater than this value, the vocabulary is reduced to `max_size` 
            removing words with lower frequency.

        freq_threshold: (int) 
            All the words whose frequency is below this threshold are remove from the vocabulary.
        """
        # check that frequencies dictionary is properly loaded
        if not self.word_frequencies:
            raise ValueError("You first need to build a vocabulary or to load it from file!")

        #limit vocab by removing low freq words
        self.word_frequencies = {word: freq for word, freq in self.word_frequencies.items() if freq >= freq_threshold}
        
        #limit vocab to the max_size specified
        self.word_frequencies = dict(sorted(self.word_frequencies.items(), key=lambda x: -x[1])[:(max_size-len(self.special_tokens))])


    def get_GloVe_indexes(
            self, 
            GloVe_file_path: str
    ) -> None:
        """
        Load GloVe embeddings file, filter words that are present in our vocabulary, and create a map between such words
        and the row index in the GloVe file.

        Parameters:
        -----------
        GloVe_file_path: (str)
            Path to the location in which the file containing the GloVe embeddings is stored. 
        """
        # check that frequencies dictionary is properly loaded
        if not self.word_frequencies:
            raise ValueError("You first need to build a word vocabulary or load it from file!")

        # Add to the dictionary all the vocabulary words with the <OOV> index
        oov_idx = self.special_tokens.index('<OOV>')
        self.word_to_idx.update(dict(zip(self.word_frequencies.keys(), 
                                         np.ones(len(self.word_frequencies.keys()), dtype=int)*oov_idx)))
        
        # Associate ad-hoc indexes to the special tokens
        for idx, token in enumerate(self.special_tokens):
            self.word_to_idx[token] = idx

        num_special_tokens=len(self.special_tokens)

        # load the GloVe file containing the embeddings and update word_to_idx dict
        with open(GloVe_file_path, "r",encoding='utf-8') as f:
            for i, line in enumerate(f):
                glove_word = line.split(" ")[0] 
                if glove_word in self.word_frequencies.keys():
                    self.word_to_idx[glove_word] = i + num_special_tokens

        self.idx_to_word = {idx: word for word, idx in enumerate(self.word_to_idx)}
    

    def word_numericalize(
            self, 
            word_list: List[str], 
            add_SOS: Optional[bool] = False, 
            add_EOS: Optional[bool] = False
    ) -> List[int]:
        """
        Map words/tokens to the index associated to their GloVe embedding.

        Paramters:
        ----------
        word_list: (List[str])
            A list of words/tokens.
        
        add_SOS: (Optional[bool] = False) 
            If `True`, the token <SOS> is appended at the start of the sentence.

        add_EOS: (Optional[bool] = False)
            If `True`, the token <EOS> is appended at the end of the sentence.

        Returns:
        --------
        numericalized_text: (List[int])
            A list of indexes associated to the embedding vectors
        """
        numericalized_text = []
        
        if add_SOS:
            numericalized_text.append(self.word_to_idx['<SOS>'])
        
        for token in word_list:
            if token in self.word_to_idx.keys():
                numericalized_text.append(self.word_to_idx[token])
            else: #out-of-vocab words are represented by OOV token index
                numericalized_text.append(self.word_to_idx['<OOV>'])
        
        if add_EOS:
            numericalized_text.append(self.word_to_idx['<EOS>'])       
                
        return numericalized_text
    

    def char_numericalize(
            self, 
            word_list: List[str], 
            add_SOS: Optional[bool] = False, 
            add_EOS: Optional[bool] = False
    ) -> List[int]:
        """
        Map single characters to indexes.

        Paramters:
        ----------
        word_list: (List[str])
            A list of words/tokens.
        
        add_SOS: (Optional[bool] = False) 
            If `True`, the token <SOS> is appended at the start of the sentence.

        add_EOS: (Optional[bool] = False)
            If `True`, the token <EOS> is appended at the end of the sentence.

        Returns:
        --------
        numericalized_text: (List[int])
            A list of indexes correspondent to the input characters.
        """
        numericalized_text = []
        
        if add_SOS:
            numericalized_text.append([self.char_to_idx['<SOS>']])
        
        for token in word_list:
            numericalized_token = []
            for char in token:
                if char in self.char_to_idx.keys():
                    numericalized_token.append(self.char_to_idx[char])
                else: #out-of-vocab char are represented by OOV token index
                    numericalized_token.append(self.char_to_idx['<OOV>'])
            numericalized_text.append(numericalized_token)
        
        if add_EOS:
            numericalized_text.append([self.char_to_idx['<EOS>']])
                
        return numericalized_text
#------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------------
class Train_Dataset(Dataset):
    """
    Parameters:
    -----------
    df: (pd.Dataframe)
        The training dataframe
    vocab_freq_threshold: (int)
        All the words whose frequency is below this threshold are remove from the vocabulary.
    vocab_max_size: (int)
        If vocabulary size is greater than this value, the vocabulary is reduced to `vocab_max_size` removing words with lower frequency.
    special_tokens: (Iterable[str] = None)
        A collection of special tokens to include in the vocabulary.
    vocabulary: (Optional[Vocabulary] = None) 
        A Vocabulary object created for this dataset
    glove_file: Optional[str] = r"..\glove.6B\glove.6B.50d.txt"
        The path to the GloVe embeddings file.
    """
    def __init__(
            self, 
            df: pd.Dataframe, 
            vocab_freq_threshold: int, 
            vocab_max_size: int, 
            special_tokens: Iterable[str] = None, 
            vocabulary: Optional[Vocabulary] = None,
            glove_file: Optional[str] = r"..\glove.6B\glove.6B.50d.txt"
    ):
    
        self.df = df
        
        #extract useful data from the dataframe
        self.impossibles = self.df["impossible"]
        self.contexts = self.df["context"]
        self.questions = self.df["question"]
        self.ids = self.df["id"]
        self.answer_starts = self.df["answer_start"]
        self.answer_ends = self.df["answer_end"]
        self.pl_answer_starts = self.df["plausible_answer_start"]
        self.pl_answer_ends = self.df["plausible_answer_end"]
        self.original_start_points = self.df["original_start_points"]
        self.original_end_points = self.df["original_end_points"]
        self.context_pos_grouped_idxs = self.df['context_pos_grouped_idxs']
        self.question_pos_grouped_idxs = self.df['question_pos_grouped_idxs']
        self.context_pos = self.df['context_pos']
        self.question_pos = self.df['question_pos']
        
        self.question_ent_idxs = self.df['question_ent_idxs']
        self.context_ent_idxs = self.df['context_ent_idxs']
        
        
        self.tag_to_idx = dict(zip(glossary.GLOSSARY.keys(), np.arange(3, len(glossary.GLOSSARY.keys()) + 3)))

        #load/create the vocabulary 
        if isinstance(vocabulary, str): #path to vocabulary to load
            self.vocabulary = Vocabulary(special_tokens)
            self.vocabulary.load_word_vocabulary(vocabulary)
            self.vocabulary.build_char_vocabulary(pd.concat([self.contexts, self.questions], axis=0))
        elif isinstance(vocabulary, Vocabulary): #vocabulary already created
            self.vocabulary = vocabulary
        else:
            self.vocabulary = Vocabulary(special_tokens)
            self.vocabulary.build_word_vocabulary(pd.concat([self.contexts, self.questions], axis=0))
            self.vocabulary.build_char_vocabulary(pd.concat([self.contexts, self.questions], axis=0))

        #clean vocabulary according to input parameters
        self.vocabulary.reduce_word_vocabulary(vocab_freq_threshold, vocab_max_size)

        #compute the glove indexes
        self.vocabulary.get_GloVe_indexes(glove_file)

        #get glove matrix to compute POS embeddings
        self.glove_matrix = GloVe_embedding_matrix(glove_file, special_tokens=None)

        
    def __len__(self):
        return len(self.df)
    

    def get_POS_embeddings(
            self, 
            numericalized_text: Iterable[int], 
            pos_idxs_dict: Dict[str, List[int]], 
            pos_vals: List[str]
    ) -> Tuple[torch.Tensor[float], List[int]]:
        """
        Compute embedding vectors for the POS tags. Those embedding are computed as the average of the 
        embedding vectors of all the words that are associated to a certain POS tag. 
        
        Parameters:
        -----------
        numericalized_text: (Iterable[int])
            A list of indexes associated to the tokens in the text.

        pos_idxs_dict: (Dict[str, List[int]])
            A dictionary whose keys are the POS tags present in the text and values are lists of
            positions in which they are present in the text

        pos_vals: (List[str])
            The list on unique POS tags values present in the text.

        Returns:
        --------
        embedding_matrix: (torch.Tensor[float])
            A matrix whose rows are the embedding vector associated to the POS tags.

        numericalized_tags: (List[int])
            A list on indexes which POS tags are mapped to.
        """

        temp_embedding_matrix = torch.empty((len(self.tag_to_idx.keys())+2, self.glove_matrix.shape[1]))
        numericalized_tags = []
        i = 2 # ['<PAD>','<OOV>', '<SOS>'] not oov here
        temp_embedding_matrix[0, :] = torch.randn_like(self.glove_matrix[0, :]) # pad
        temp_embedding_matrix[1, :] = torch.randn_like(self.glove_matrix[1, :]) # oov
        temp_embedding_matrix[2, :] = torch.randn_like(self.glove_matrix[3, :]) # sos
        
        embedding_matrix = torch.empty((len(pos_vals)+1, self.glove_matrix.shape[1]))
        
        numericalized_tags.append(2)
        embedding_matrix[0,:] = temp_embedding_matrix[2, :] 
        
        for tag, idxs in pos_idxs_dict.items():
            curr_glove_vectors = self.glove_matrix[numericalized_text[idxs], :]
            temp_embedding_matrix[self.tag_to_idx[tag], :] = curr_glove_vectors.mean(dim=0)
            
        for idx, pos in enumerate(pos_vals):
            numericalized_tags.append(self.tag_to_idx[pos])
            embedding_matrix[idx+1, :] = temp_embedding_matrix[self.tag_to_idx[pos], :]
        
        return embedding_matrix, numericalized_tags
    

    def __getitem__(self, index):
        #extract the items at the current index
        impossible = self.impossibles[index]
        context_text = self.contexts[index]
        question_text = self.questions[index]
        id = self.ids[index]
        original_start_points = self.original_start_points[index]
        original_end_points = self.original_end_points[index]
        context_pos_idx_dict = self.context_pos_grouped_idxs[index]
        question_pos_idx_dict = self.question_pos_grouped_idxs[index]
        context_pos = self.context_pos[index]
        question_pos = self.question_pos[index]
        context_ent_idxs = self.context_ent_idxs[index]
        question_ent_idxs = self.question_ent_idxs[index]
            
        #map context and question tokens to indexes
        numericalized_context = self.vocabulary.word_numericalize(context_text, add_SOS=True)
        numericalized_question = self.vocabulary.word_numericalize(question_text, add_SOS=True)
        
        #map context and question characters to indexes
        numericalized_context_chars = self.vocabulary.char_numericalize(context_text, add_SOS=True)
        numericalized_question_chars = self.vocabulary.char_numericalize(question_text, add_SOS=True)     

        #retrieve answer start and end 
        if impossible: #plausible answer -> set endopoints to 0
            answer_start, answer_end = 0.0, 0.0
        else: #return actual endpoints
            answer_start, answer_end = self.answer_starts[index]+1, self.answer_ends[index]+1

        #convert numerical attributes to tensor
        numericalized_context = torch.tensor(numericalized_context)
        numericalized_question = torch.tensor(numericalized_question)
        answer_start = torch.tensor(answer_start)
        answer_end = torch.tensor(answer_end)

        #get POS embedding matrices
        context_pos_embedding_matrix, context_pos_tags = self.get_POS_embeddings(numericalized_context, context_pos_idx_dict, context_pos)
        question_pos_embedding_matrix, question_pos_tags = self.get_POS_embeddings(numericalized_question, question_pos_idx_dict,question_pos)
        #context_pos_tags, question_pos_tags = torch.tensor(list(context_pos_tags)), torch.tensor(list(question_pos_tags))

        # return numericalized_context, numericalized_question, id, answer_start, answer_end, original_start_points, original_end_points
        return {
            "context": numericalized_context,
            "question": numericalized_question, 
            "context_chars": numericalized_context_chars,
            "question_chars": numericalized_question_chars,
            "id": id, 
            "ans_start": answer_start, 
            "ans_end": answer_end, 
            "start_points": original_start_points, 
            "end_points": original_end_points,
            "context_pos_embedding_matrix": context_pos_embedding_matrix,
            "question_pos_embedding_matrix": question_pos_embedding_matrix,
            "context_pos_tags": context_pos_tags,
            "question_pos_tags": question_pos_tags,
            "context_ent_idxs": context_ent_idxs,
            "question_ent_idxs": question_ent_idxs
        }
#--------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------
class Validation_Dataset:
    """
    Parameters:     
    -----------
    df: (pd.Dataframe)
        The training dataframe
    vocabulary: (Optional[Vocabulary] = None) 
        A Vocabulary object created for this dataset
    glove_file: Optional[str] = r"..\glove.6B\glove.6B.50d.txt"
        The path to the GloVe embeddings file.
    
    NOTE: in the implementation we use the same vocabulary used for the training dataset
    """
    def __init__(
            self, 
            df: pd.DataFrame, 
            vocabulary: Vocabulary, 
            glove_file: str
    ):
        self.df = df
        self.vocabulary = vocabulary
        
        #extract useful data from the dataframe
        self.impossibles = self.df["impossible"]
        self.contexts = self.df["context"]
        self.questions = self.df["question"]
        self.ids = self.df["id"]
        self.answer_starts = self.df["answer_start"]
        self.answer_ends = self.df["answer_end"]
        self.pl_answer_starts = self.df["plausible_answer_start"]
        self.pl_answer_ends = self.df["plausible_answer_end"]
        self.original_start_points = self.df["original_start_points"]
        self.original_end_points = self.df["original_end_points"]
        self.context_pos_grouped_idxs = self.df['context_pos_grouped_idxs']
        self.question_pos_grouped_idxs = self.df['question_pos_grouped_idxs']
        self.context_pos = self.df['context_pos']
        self.question_pos = self.df['question_pos']
        self.question_ent_idxs = self.df['question_ent_idxs']
        self.context_ent_idxs = self.df['context_ent_idxs']

        #get glove matrix to compute POS embeddings
        self.glove_matrix = GloVe_embedding_matrix(glove_file, special_tokens=None)
        self.tag_to_idx = dict(zip(glossary.GLOSSARY.keys(), np.arange(2, len(glossary.GLOSSARY.keys()) + 2)))


    def __len__(self):
        return len(self.df)
    
    def get_POS_embeddings(
            self, 
            numericalized_text: Iterable[int], 
            pos_idxs_dict: Dict[str, List[int]], 
            pos_vals: List[str]
    ) -> Tuple[torch.Tensor[float], List[int]]:
        """
        Compute embedding vectors for the POS tags. Those embedding are computed as the average of the 
        embedding vectors of all the words that are associated to a certain POS tag. 
        
        Parameters:
        -----------
        numericalized_text: (Iterable[int])
            A list of indexes associated to the tokens in the text.

        pos_idxs_dict: (Dict[str, List[int]])
            A dictionary whose keys are the POS tags present in the text and values are lists of
            positions in which they are present in the text

        pos_vals: (List[str])
            The list on unique POS tags values present in the text.

        Returns:
        --------
        embedding_matrix: (torch.Tensor[float])
            A matrix whose rows are the embedding vector associated to the POS tags.

        numericalized_tags: (List[int])
            A list on indexes which POS tags are mapped to.
        """

        temp_embedding_matrix = torch.empty((len(self.tag_to_idx.keys())+2, self.glove_matrix.shape[1]))
        numericalized_tags = []
        i = 2 # ['<PAD>','<OOV>', '<SOS>'] not oov here
        temp_embedding_matrix[0, :] = torch.randn_like(self.glove_matrix[0, :]) # pad
        temp_embedding_matrix[1, :] = torch.randn_like(self.glove_matrix[1, :]) # oov
        temp_embedding_matrix[2, :] = torch.randn_like(self.glove_matrix[3, :]) # sos
        
        embedding_matrix = torch.empty((len(pos_vals)+1, self.glove_matrix.shape[1]))
        
        numericalized_tags.append(2)
        embedding_matrix[0,:] = temp_embedding_matrix[2, :] 
        
        for tag, idxs in pos_idxs_dict.items():
            curr_glove_vectors = self.glove_matrix[numericalized_text[idxs], :]
            temp_embedding_matrix[self.tag_to_idx[tag], :] = curr_glove_vectors.mean(dim=0)
            
        for idx, pos in enumerate(pos_vals):
            numericalized_tags.append(self.tag_to_idx[pos])
            embedding_matrix[idx+1, :] = temp_embedding_matrix[self.tag_to_idx[pos], :]
        
        return embedding_matrix, numericalized_tags
        

    def __getitem__(self,index):
        #extract the items at the current index
        impossible = self.impossibles[index]
        context_text = self.contexts[index]
        question_text = self.questions[index]
        id = self.ids[index]
        original_start_points = self.original_start_points[index]
        original_end_points = self.original_end_points[index]
        context_pos_idx_dict = self.context_pos_grouped_idxs[index]
        question_pos_idx_dict = self.question_pos_grouped_idxs[index]
        context_pos = self.context_pos[index]
        question_pos = self.question_pos[index]
        context_ent_idxs = self.context_ent_idxs[index]
        question_ent_idxs = self.question_ent_idxs[index]
            
        #map context and question tokens to indexes
        numericalized_context = self.vocabulary.word_numericalize(context_text, add_SOS=True)
        numericalized_question = self.vocabulary.word_numericalize(question_text, add_SOS=True)

        #map context and question characters to indexes
        numericalized_context_chars = self.vocabulary.char_numericalize(context_text, add_SOS=True)
        numericalized_question_chars = self.vocabulary.char_numericalize(question_text, add_SOS=True)

        #retrieve answer start and end 
        if impossible: #plausible answer -> set endopoints to 0
            answer_start, answer_end = 0.0, 0.0
        else: #return actual endpoints
            answer_start, answer_end = self.answer_starts[index]+1, self.answer_ends[index]+1
        
        #convert numerical attributes to tensor
        numericalized_context = torch.tensor(numericalized_context)
        numericalized_question = torch.tensor(numericalized_question)
        answer_start = torch.tensor(answer_start)
        answer_end = torch.tensor(answer_end)

        #get POS embedding matrices
        context_pos_embedding_matrix, context_pos_tags = self.get_POS_embeddings(numericalized_context, context_pos_idx_dict, context_pos)
        question_pos_embedding_matrix, question_pos_tags = self.get_POS_embeddings(numericalized_question, question_pos_idx_dict,question_pos)
        # context_pos_tags, question_pos_tags = torch.tensor(list(context_pos_tags)), torch.tensor(list(question_pos_tags))

        # return numericalized_context, numericalized_question, id, answer_start, answer_end, original_start_points, original_end_points
        return {
            "context": numericalized_context,
            "question": numericalized_question, 
            "context_chars": numericalized_context_chars,
            "question_chars": numericalized_question_chars,
            "id": id, 
            "ans_start": answer_start, 
            "ans_end": answer_end, 
            "start_points": original_start_points, 
            "end_points": original_end_points,
            "context_pos_embedding_matrix": context_pos_embedding_matrix,
            "question_pos_embedding_matrix": question_pos_embedding_matrix,
            "context_pos_tags": context_pos_tags,
            "question_pos_tags": question_pos_tags,
            "context_ent_idxs": context_ent_idxs,
            "question_ent_idxs": question_ent_idxs
        }
#-------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------
class MyCollate:
    """
    Class that provides methods to collate instances of the previously defined Datasets into batches,
    specifically applying padding to allow storing data in tensors.
    """
    def __init__(
            self, 
            word_pad_idx: int, 
            char_pad_idx: int
    ):
        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx    
        
    def __call__(self, batch):
        #get all indexed sentences of the batch
        context = [item["context"] for item in batch] 
        question = [item["question"] for item in batch]
        
        context_chars = [item["context_chars"] for item in batch] 
        question_chars = [item["question_chars"] for item in batch] 
        
        context_pos_tags = [item["context_pos_tags"] for item in batch]
        question_pos_tags = [item["question_pos_tags"] for item in batch]
        
        context_pos_emb_mat = [item["context_pos_embedding_matrix"] for item in batch]
        question_pos_emb_mat = [item["question_pos_embedding_matrix"] for item in batch]
        
        context_ent_idxs = [torch.Tensor(item["context_ent_idxs"]) for item in batch]
        question_ent_idxs = [torch.Tensor(item["question_ent_idxs"]) for item in batch]
        
        #pad them using pad_sequence method from pytorch. 
        context = pad_sequence(context, batch_first=False, padding_value = self.word_pad_idx)  
        question = pad_sequence(question, batch_first=False, padding_value = self.word_pad_idx)
        
        context_chars = pad_characters(context_chars, self.char_pad_idx)
        question_chars = pad_characters(question_chars, self.char_pad_idx)
        
        context_pos_tags = pad_pos_indexes(context_pos_tags, padding_idx = self.word_pad_idx)
        question_pos_tags = pad_pos_indexes(question_pos_tags, padding_idx = self.word_pad_idx)
        
        context_ent_idxs = pad_sequence(context_ent_idxs, padding_value = self.word_pad_idx, batch_first=True)
        question_ent_idxs = pad_sequence(question_ent_idxs, padding_value = self.word_pad_idx, batch_first=True)
        
        context_pos_emb_mat = pad_sequence(context_pos_emb_mat, batch_first=True)
        question_pos_emb_mat = pad_sequence(question_pos_emb_mat, batch_first=True)
        
        #get the other features that doesn't need to be padded
        id = [item["id"] for item in batch]
        ans_start = [item["ans_start"] for item in batch]
        ans_end = [item["ans_end"] for item in batch]
        start_points = [item["start_points"] for item in batch]
        end_points = [item["end_points"] for item in batch]
        
        
        return {
            "context": context,
            "question": question, 
            "context_chars": context_chars,
            "question_chars": question_chars,
            "id": id, 
            "ans_start": ans_start, 
            "ans_end": ans_end, 
            "start_points": start_points, 
            "end_points": end_points,
            "context_pos_emb_mat": context_pos_emb_mat,
            "question_pos_emb_mat": question_pos_emb_mat,
            "context_pos_tags": context_pos_tags,
            "question_pos_tags": question_pos_tags,
            "context_ent_idxs": context_ent_idxs,
            "question_ent_idxs": question_ent_idxs
        }
#----------------------------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------------------------------
def get_train_loader(
        dataset: Dataset, 
        batch_size: int, 
        num_workers: Optional[int] = 0, 
        shuffle: Optional[bool] = True, 
        pin_memory: Optional[bool] = True
):
    word_pad_idx = dataset.vocabulary.word_to_idx['<PAD>']
    char_pad_idx = dataset.vocabulary.char_to_idx['<PAD>']
    loader = DataLoader(dataset, 
                        batch_size = batch_size, 
                        num_workers = num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory, 
                        collate_fn = MyCollate(word_pad_idx, char_pad_idx))
    return loader
#----------------------------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------------------------------
def get_valid_loader(        
        dataset: Dataset, 
        batch_size: int, 
        num_workers: Optional[int] = 0, 
        shuffle: Optional[bool] = True, 
        pin_memory: Optional[bool] = True
):
    word_pad_idx = dataset.vocabulary.word_to_idx['<PAD>']
    char_pad_idx = dataset.vocabulary.char_to_idx['<PAD>']
    loader = DataLoader(dataset, 
                        batch_size = batch_size, 
                        num_workers = num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory, 
                        collate_fn = MyCollate(word_pad_idx, char_pad_idx))
    return loader
#----------------------------------------------------------------------------------------------------------------------------------------------------

