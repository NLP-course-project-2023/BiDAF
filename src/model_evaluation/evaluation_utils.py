
#  ---- IMPORT LIBRARIES ---- #

import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


#  ------------------------ #


def process_logits(logits):
  """ Logits normalization and softmax application to pass from logits to probabilities 

      Input: logits (`np.ndarray`): (1, N)
      Output: probabilities (`np.ndarray`): (1, N)
  """
  # Normalize logits 
  logits = logits - logits.max(axis=-1, keepdims=True)

  # Apply the softmax 
  exp_logits = np.exp(logits)
  probabilities = exp_logits / exp_logits.sum()

  return probabilities


#  ------------------------ #


def check_spans(start_idxs, end_idxs):
  
  # Check that all spans are meaningful: start < end
  check1 = np.all(start_idxs <= end_idxs)

  return check1 


#  ------------------------ #

def predict(model, data_loader, device):    
  """ Get model raw predictions. 

      Output: 
      predictions = [{
                    'id': Sample identifier
                    'start': Individual start logits for each token (1, context_len)
                    'end': Individual end logits for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }, ... ] 
      
  """

  results = []
  model.eval()

  with torch.no_grad():
      eval_bar = tqdm(data_loader)
      for e in eval_bar:
          cw_idxs = e['context'].T
          qw_idxs  = e['question'].T
          cc_idxs  = e['context_chars']
          qc_idxs = e['question_chars']
          id = e['id']
          y1 = e['ans_start']
          y2 = e['ans_end']
          start_points = e['start_points']
          end_points = e['end_points']
          
          # Setup for forward
          cw_idxs = cw_idxs.to(device)
          qw_idxs = qw_idxs.to(device)
          batch_size = cw_idxs.size(0)

          # Forward
          log_p1, log_p2 = model(cw_idxs, qw_idxs)
          y1, y2 = torch.stack(y1),torch.stack(y2)
          y1, y2 = torch.nan_to_num(y1), torch.nan_to_num(y2)
          y1, y2 = y1.to(device), y2.to(device)
          y1, y2 = y1.to(torch.int64), y2.to(torch.int64)


          for i in range((batch_size)):
            results.append({
                'id':id[i],
                'start':log_p1[i,:],
                'end':log_p2[i,:], 
                'start_points': start_points[i], # start reference in char over the context for each token 
                'end_points': end_points[i], # end reference in char over the context for each token
                'token_target_start': y1[i]-1, # answer start in token based on our tokenization
                'token_target_end': y2[i]-1 # answer end in token based on our tokenization
                })

          # Log info
          eval_bar.update(1)
          eval_bar.set_postfix(NLL = max(y1))

  return results

#  ------------------------ #
    
def predict_original(model, data_loader, device):    
  """ Get model predictions. 

      Output: 
      predictions = [{
                    'id': Sample identifier
                    'start': Individual start logp for each token (1, context_len)
                    'end': Individual end logp for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }, ... ] 
      
  """

  results = []
  model.eval()

  with torch.no_grad():
      eval_bar = tqdm(data_loader)
      for e in eval_bar:
          cw_idxs = e['context'].T
          qw_idxs  = e['question'].T
          cc_idxs  = e['context_chars']
          qc_idxs = e['question_chars']
          id = e['id']
          y1 = e['ans_start']
          y2 = e['ans_end']
          start_points = e['start_points']
          end_points = e['end_points']
          
          # Setup for forward
          cw_idxs = cw_idxs.to(device)
          qw_idxs = qw_idxs.to(device)
          batch_size = cw_idxs.size(0)

          # Forward
          log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
          y1, y2 = torch.stack(y1),torch.stack(y2)
          y1, y2 = torch.nan_to_num(y1), torch.nan_to_num(y2)
          y1, y2 = y1.to(device), y2.to(device)
          y1, y2 = y1.to(torch.int64), y2.to(torch.int64)


          for i in range((batch_size)):
            results.append({
                'id':id[i],
                'start':log_p1[i,:],
                'end':log_p2[i,:], 
                'start_points': start_points[i], # start reference in char over the context for each token 
                'end_points': end_points[i], # end reference in char over the context for each token
                'token_target_start': y1[i]-1, # answer start in token based on our tokenization
                'token_target_end': y2[i]-1 # answer end in token based on our tokenization
                })

          # Log info
          eval_bar.update(1)
          eval_bar.set_postfix(NLL = max(y1))

  return results

#  ------------------------ #

def predict_pro(model, data_loader, device):    
  """ Get model predictions. 

      Output: 
      predictions = [{
                    'id': Sample identifier
                    'start': Individual start logp for each token (1, context_len)
                    'end': Individual end logp for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }, ... ] 
      
  """

  results = []
  model.eval()

  with torch.no_grad():
      eval_bar = tqdm(data_loader)
      for e in eval_bar:
          cw_idxs = e['context'].T
          qw_idxs  = e['question'].T
          cc_idxs  = e['context_chars']
          qc_idxs = e['question_chars']
          id = e['id']
          y1 = e['ans_start']
          y2 = e['ans_end']
          start_points = e['start_points']
          end_points = e['end_points']

          c_pos = e['context_pos_emb_mat']
          q_pos = e['question_pos_emb_mat']

          ce_idxs = e['context_ent_idxs']
          qe_idxs = e['question_ent_idxs']

          
          # Setup for forward
          cw_idxs = cw_idxs.to(device)
          qw_idxs = qw_idxs.to(device)
          batch_size = cw_idxs.size(0)

          # Forward
          log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, c_pos, q_pos, ce_idxs, qe_idxs)
          y1, y2 = torch.stack(y1),torch.stack(y2)
          y1, y2 = torch.nan_to_num(y1), torch.nan_to_num(y2)
          y1, y2 = y1.to(device), y2.to(device)
          y1, y2 = y1.to(torch.int64), y2.to(torch.int64)


          for i in range((batch_size)):
            results.append({
                'id':id[i],
                'start':log_p1[i,:],
                'end':log_p2[i,:], 
                'start_points': start_points[i], # start reference in char over the context for each token 
                'end_points': end_points[i], # end reference in char over the context for each token
                'token_target_start': y1[i]-1, # answer start in token based on our tokenization
                'token_target_end': y2[i]-1 # answer end in token based on our tokenization
                })

          # Log info
          eval_bar.update(1)
          eval_bar.set_postfix(NLL = max(y1))

  return results

#  ------------------------ #


def get_answer_span(start, end, max_answer_len = -1):
  """ Retrieve the answer from the probabilities vectors. 

      Input:
        start (`np.ndarray`): Individual start probabilities for each token (context_len)
        end (`np.ndarray`): Individual end probabilities for each token (context_len)
        max_answer_len (int): upper limit for number of tokens in the provided answer

      Output: 
        joint_probability: joint probability matrix containing at (i,j) the s(i)*e(j)
        start_idxs: start token of the span with max probability
        end_idxs: end token of the span with max probability
        max_prob: max probability over all valid spans 
        no_answer_prob: probability of impossible answer

  """
  # Set the maximum length of the answer to the length of the context if no specified
  if max_answer_len < 0: max_answer_len = start.shape[-1]

  # Get the probability to have no answer
  no_answer_prob = start[0] * end[0]

  # Remove the 0-index probability and to exclude it from the max retrieval over the valid tokens
  start[0] = end[0] = 0.0

  # Exapnd start and end as column and row vectors 
  #     start:  (context_len, 1)
  #     end:   (1, context_len)
  start = np.expand_dims(start, -1)
  end = np.expand_dims(end, 0)

  # Compute pairwise probabilities
  #     joint_probability: (context_len, context_len)
  joint_probability = np.matmul(start, end)

  # Remove pair with end < start 
  joint_probability = np.triu(joint_probability)

  # Remove pair with end - start > max_answer_len
  joint_probability = np.tril(joint_probability, max_answer_len - 1)

  # Take the pair (i, j) that maximizes the joint probabilities and that max value
  max_in_row = np.max(joint_probability, axis=1)
  max_in_col = np.max(joint_probability, axis=0)
  start_idxs = np.argmax(max_in_row)
  end_idxs = np.argmax(max_in_col)
  max_prob = joint_probability[start_idxs, end_idxs]

  if not check_spans(start_idxs, end_idxs): 
    print("----- CHECK KO -------")
    print(f"> Start: {start_idxs}")
    print(f"> End: {end_idxs}")
    print(f"> Max: {max_prob}")
    print("Setting start_idxs, end_idxs and max_prob to zero")
    print()
    start_idxs = 0
    end_idxs = 0
    max_prob = 0.0

  return joint_probability, start_idxs, end_idxs, max_prob, no_answer_prob


# ------------------------ #


def process_predictions(predictions, to_numpy=True, max_answer_len = -1):
  """ Process model predictions to get the predicted answer as token span. 
      
      > Input must be a list of N dictionaries in the format:

      predictions = [{
                    'id': Sample identifier
                    'start': Individual start logits for each token (1, context_len)
                    'end': Individual end logits for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }, ... ] 

      > Output will be a list of N dictionaries in the format:

      results = [{
                    'id': Sample identifier
                    'start_idxs': start token of the span with max probability
                    'end_idxs': end token of the span with max probability
                    'max_prob': max probability over all valid spans 
                    'no_answer_prob': probability to have no answer
                    'start_prob': Individual start probabilities for each token (1, context_len)
                    'end_prob': Individual end probabilities for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }, ... ]
  """

  results = []

  eval_bar = tqdm(predictions)
  for sample in eval_bar:

    start_logits = sample['start']
    end_logits = sample['end']

    if to_numpy:
      start_logits = start_logits.cpu()
      end_logits = end_logits.cpu()
      start_logits = np.array(start_logits)
      end_logits = np.array(end_logits)


    start_prob = process_logits(start_logits)
    end_prob = process_logits(end_logits)

    joint_probability, start_idxs, end_idxs, max_prob, no_answer_prob = get_answer_span(start_prob, end_prob, max_answer_len)

    results.append({  'id': sample['id'],
                      # 'joint_probability': joint_probability,
                      'start_idxs': start_idxs,
                      'end_idxs': end_idxs,
                      'max_prob': max_prob, 
                      'no_answer_prob': no_answer_prob,
                      'start_prob': start_prob,
                      'end_prob': end_prob,
                      # 'start_logits': start_logits,
                      # 'end_logits': end_logits,
                      'start_points': sample['start_points'],
                      'end_points': sample['end_points'],
                      'token_target_start': sample['token_target_start'],
                      'token_target_end': sample['token_target_end']
                  })

  return results


#  ------------------------ #



def retrieve_answers(results, df):
  """ Retrieve textual answers from spans in tokens. 

      Input: 
      results = [{
                    'id': Sample identifier
                    'start_idxs': start token of the span with max probability
                    'end_idxs': end token of the span with max probability
                    'max_prob': max probability over all valid spans 
                    'no_answer_prob': probability to have no answer,
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    ... }, ... ]

      Output: 
      answers = [{
                    'id': Sample identifier
                    'answer_score': score to have predicted correctly the answer
                    'no_answer_score': score to have no answer
                    'predicted_answer': the predicted answer as a string
                    'predicted_start_token_oncontext': start token of the predicted answer's span on tokenized context (-1 if no answer predicted)
                    'predicted_end_token_oncontext': end token of the predicted answer's span on tokenized context(-1 if no answer predicted)
                    'target_start_token_oncontext': start token of the correct answer's span on tokenized context (-1 if impossible answer)
                    'target_end_token_oncontext': end token of the correct answer's span on tokenized context (-1 if impossible answer)
                    'target_answer': correct answer 
                    'target_plausible': correct plausible answer
                    }, ... ]
  """

  answers = []

  eval_bar = tqdm(results)
  for sample in eval_bar:

    target_start_token = sample['token_target_start']
    target_end_token = sample['token_target_end']
    predicted_start_token = sample['start_idxs']
    predicted_end_token = sample['end_idxs']
    start_points = sample['start_points']
    end_points = sample['end_points']
    answer_prob = sample['max_prob']
    no_answer_prob = sample['no_answer_prob']
    id = sample['id']


    # TO BE REMOVED 
    if predicted_start_token == 0 or predicted_end_token == 0: 
      print("\n ----- WARNING: PREDICTED 0 ---- \n")
      print("Sample:    ", sample['id'][0])
      predicted_start_token_oncontext = 0
      predicted_end_token_oncontext = 0 

    else: 
      # Get the token reference without the 0-index case
      predicted_start_token_oncontext = predicted_start_token-1
      predicted_end_token_oncontext = predicted_end_token-1

      # Get the char reference 
      predicted_start_char = start_points[predicted_start_token_oncontext]
      predicted_end_char = end_points[predicted_end_token_oncontext]

    provide_answer = True
    # Inspired by Threshold-based answerable verification (TAV2) - Neural Question Answering on SQuAD 2.0 (Y. Zhou)
    score_no_answer = no_answer_prob - answer_prob
    if score_no_answer > answer_prob: provide_answer = False

    # Retrieve from the dataset the information
    x = df[df['id'] == id]
    context = x['origin_context'].values[0]
    answer = context[predicted_start_char:predicted_end_char+1]

    answers.append({
        'id': id,
        'answer_score': answer_prob,
        'no_answer_score': score_no_answer,
        'predicted_answer': answer, 
        'predicted_start_token_oncontext': predicted_start_token_oncontext if provide_answer else -1,
        'predicted_end_token_oncontext': predicted_end_token_oncontext if provide_answer else -1, 
        'target_start_token_oncontext': target_start_token,
        'target_end_token_oncontext': target_end_token,
        'target_answer': x['answer'].values[0],
        'target_plausible': x['plausible_answer'].values[0]   
    })

  return answers


# ------------------------ #


def eval_results(answers):
  """ Prepare the results for the official evaluation script (evaluate.py).

  Input: 
      answers = [{
                    'id': Sample identifier
                    'answer_score': score to have predicted correctly the answer
                    'no_answer_score': score to have no answer
                    'predicted_answer': the predicted answer as a string
                    'predicted_start_token_oncontext': start token of the predicted answer's span on tokenized context (-1 if no answer predicted)
                    'predicted_end_token_oncontext': end token of the predicted answer's span on tokenized context(-1 if no answer predicted)
                    'target_start_token_oncontext': start token of the correct answer's span on tokenized context (-1 if impossible answer)
                    'target_end_token_oncontext': end token of the correct answer's span on tokenized context (-1 if impossible answer)
                    'target_answer': correct answer 
                    'target_plausible': correct plausible answer
                    }, ... ]
  
  Output: 
      answers_eval = { id : predicted_answer  , ...    }
      NB: if the question is predicted as impossible, the answer is ''

  """

  answers_eval = {}

  eval_bar = tqdm(answers)
  for sample in eval_bar:

      id = sample['id']
      score_no_answer = sample['no_answer_score']
      answer_score = sample['answer_score']
      answer = sample['predicted_answer']

      provide_answer = True
      if score_no_answer > answer_score: provide_answer = False

      answers_eval[id] = answer if provide_answer else '' 

  return answers_eval



# ------------------------ #



def get_graphics(sample_result, df):
  """ Get visualizations about model performances.

    Input:
    sample_result = {
                    'id': Sample identifier
                    'start_idxs': start token of the span with max probability
                    'end_idxs': end token of the span with max probability
                    'max_prob': max probability over all valid spans 
                    'no_answer_prob': probability to have no answer,
                    'start_prob': Individual start probabilities for each token (1, context_len)
                    'end_prob': Individual end probabilities for each token (1, context_len)
                    'start_points': start reference in char over the context for each token 
                    'end_points': end reference in char over the context for each token
                    'token_target_start': answer start in token based on our tokenization
                    'token_target_end': answer end in token based on our tokenization
                    }
    
  """
  start_scores = sample_result['start_prob'][1:]
  end_scores = sample_result['end_prob'][1:]
  sample_id = sample_result['id']
  tokens = df[df['id']==sample_id]['context'].values[0]
  answer = df[df['id']==sample_id]['answer'].values[0]
  plausible_answer = df[df['id']==sample_id]['plausible_answer'].values[0]
  token_target_start = sample_result['token_target_start'].cpu()
  token_target_end = sample_result['token_target_end'].cpu()

  # Consider probability vectors without padding
  padd = len(start_scores)-len(tokens)
  start_scores = start_scores[:-padd]
  end_scores = end_scores[:-padd]

  # We'll use the tokens as the x-axis labels. In order to do that, they all need
  # to be unique, so we'll add the token index to the end of each one.
  token_labels = []
  for (i, token) in enumerate(tokens):
      token_labels.append('{:} - {:>2}'.format(token, i))

  # Store the tokens and scores in a DataFrame. 
  # Each token will have two rows, one for its start score and one for its end
  # score. The "marker" column will differentiate them. A little wacky, I know.
  scores = []
  for (i, token_label) in enumerate(token_labels):

      # Add the token's start score as one row.
      scores.append({'token_label': token_label, 
                    'score': start_scores[i],
                    'marker': 'start'
                    })
      
      # Add  the token's end score as another row.
      scores.append({'token_label': token_label, 
                    'score': end_scores[i],
                    'marker': 'end'
                    })
      
  df = pd.DataFrame(scores)

  # Draw a grouped barplot to show start and end scores for each word.
  # The "hue" parameter is where we tell it which datapoints belong to which
  # of the two series.
  g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                  kind="bar", height=6, aspect=4)
  
  # Report the textual correct answer 
  g.ax.set_title(f"Sample: {sample_id}\nAnswer: {answer}\nPlausible Answer: {plausible_answer}")
  
  # Create a colored background for the bin
  t_start_pos = token_target_start - 0.5
  t_end_pos = token_target_end + 0.5
  g.ax.axvspan(t_start_pos, t_end_pos, facecolor='green', alpha=0.2)



  # Turn the xlabels vertical.
  g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

  # Turn on the vertical grid to help align words to scores.
  g.ax.grid(True)


# ------------------------ #


