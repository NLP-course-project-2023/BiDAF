import re
import string
from collections import Counter
from data_preparation.SQuAD_dataset import Vocabulary
import torch
import numpy as np
import json
import logging
import os
import queue
import shutil
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from types import SimpleNamespace
from typing import Tuple, List, Union, Any, Dict

# --------------------------------------------------------------------------------------------------------------------------------------------
def create_vocabulary(sentences: List[str]) -> Vocabulary:
    """
    Create a vocabulary from the given sentences.

    Args:
        sentences (List[str]): List of sentences to construct vocabulary from.

    Returns:
        Vocabulary: An object of Vocabulary containing the word and character vocabularies.
    """

    vocab = Vocabulary()
    vocab.build_word_vocabulary(sentences)
    vocab.build_char_vocabulary(sentences)

    return vocab


# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------
def save_args(args: Any, path: str) -> None:
    """
    Save the arguments to a JSON file.

    Args:
        args (Any): Arguments object or dictionary.
        path (str): Path to the file where arguments should be saved.
    """
    with open(path, 'w') as f:
        json.dump(args, f, indent = 4, sort_keys = True)


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def get_args(path: str) -> SimpleNamespace:
    """

        Parse the arguments from a JSON file and return as a SimpleNamespace object.

        Args:
            path (str): Path to the JSON file containing the arguments.

        Returns:
            SimpleNamespace: Parsed arguments.


         DESC OF SOME ARGS

         'name':        'test',  # Name to identify training or test run
         'max_ans_len': 15,  # Maximum length of a predicted answer
         'num_workers': 4,  # Number of sub-processes to use per data loader
         'save_dir':    './save/',  # Base directory for saving information
         'batch_size':  64,  # Batch size per GPU. Scales automatically when multiple GPUs are available
         'hidden_size': 100,  # Number of features in encoder hidden layers
         'num_visuals': 10,  # Number of examples to visualize in TensorBoard.
         'load_path':   None  # Path to load as a model checkpoint.

     """

    # Parse JSON into an object with attributes corresponding to dict keys
    with open(path) as f:
        args = json.load(f, object_hook = lambda d: SimpleNamespace(**d))

    if args.best_metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.best_metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.best_metric_name}"')

    return args


# --------------------------------------------------------------------------------------------------------------------------------------------
def get_save_dir(base_dir: str, name: str, training: bool, id_max: int = 100) -> Tuple[str, int]:
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir, uid

    raise RuntimeError('Too many save directories created with the same name')


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def get_available_devices() -> Tuple[torch.device, List[int]]:
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def get_logger(log_dir: str, name: str) -> logging.Logger:
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt = '%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt = '%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def load_model(model: nn.Module, checkpoint_path: str, return_step: bool = True) -> Union[nn.Module, Tuple[nn.Module, int]]:
    """Load model parameters from disk.

    Args:
        model (torch.nn.
        Parallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    # device = f"cuda:{gpu_ids[0] if gpu_ids else 'cpu'}"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ckpt_dict = torch.load(checkpoint_path, map_location = device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

def visualize(tbx: Any, pred_dict: Dict[str, str], eval_path: str, step: int, split: str, num_visuals: int) -> None:
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size = num_visuals, replace = False)

    with open(eval_path, 'r') as eval_file:
        eval_dict = json.load(eval_file)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']

        gold = answers[0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag = f'{split}/{i + 1}_of_{num_visuals}',
                     text_string = tbl_fmt,
                     global_step = step)


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def evaluate(model: nn.Module, data_loader: Any, device: torch.device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (Any): DataLoader providing the data.
        device (torch.device): Device (CPU/GPU) to use for computation.

    Returns:
        Tuple[OrderedDict, Dict[str, str]]: Results of the evaluation and the predictions dictionary.
    """
    nll_meter = AverageMeter()

    model.eval()
    pred_dict = {}
    l = len(data_loader.dataset)
    print(l)
    # with open(eval_file, 'r') as fh:
    #    gold_dict = json.load(fh)
    with torch.no_grad():
        eval_bar = tqdm(data_loader)
        for e in eval_bar:
            cw_idxs = e['context'].T
            qw_idxs = e['question'].T
            cc_idxs = e['context_chars']
            qc_idxs = e['question_chars']
            ce_idxs = e['context_ent_idxs']
            qe_idxs = e['question_ent_idxs']
            id = e['id']
            y1 = e['ans_start']
            y2 = e['ans_end']
            start_points = e['start_points']
            end_points = e['end_points']

            if model.module.model_type == 'pro':
                c_pos = c_pos.to(device)
                q_pos = q_pos.to(device)
                ce_idxs = ce_idxs.to(device)
                qe_idxs = qe_idxs.to(device)

            if model.module.model_type != 'base':
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)

            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if model.module.model_type == 'original':
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            elif model.module.model_type == 'base':
                log_p1, log_p2 = model(cw_idxs, qw_idxs)

            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, c_pos, q_pos, ce_idxs, qe_idxs)

            # log_p1, log_p2 =  log_p1.to(torch.float32) , log_p2.to(torch.float32)
            y1, y2 = torch.stack(y1), torch.stack(y2)
            y1, y2 = torch.nan_to_num(y1), torch.nan_to_num(y2)
            y1, y2 = y1.to(device), y2.to(device)
            y1, y2 = y1.to(torch.int64), y2.to(torch.int64)

            loss1 = nn.NLLLoss()
            loss = loss1(log_p1, y1) + loss1(log_p2, y2)
            loss_val = loss.item()
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            # p1, p2 = log_p1.exp(), log_p2.exp()
            # starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            eval_bar.update(1)
            eval_bar.set_postfix(NLL = nll_meter.avg)

            """preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)"""

    model.train()

    # results = eval_dicts(gold_dict, pred_dict)
    """results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)"""

    results_list = [('NLL', nll_meter.avg)]
    results = OrderedDict(results_list)

    return results, pred_dict


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def metric_max_over_ground_truths(metric_fn: Any, prediction: str, ground_truths: List[str]) -> float:
    """
    Calculate the maximum metric value over all ground truths.

    Args:
        metric_fn (function): A function that computes the desired metric.
        prediction (str): The predicted answer string.
        ground_truths (List[str]): A list of ground truth answer strings.

    Returns:
        float: The maximum metric value over all ground truth answers.
    """
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by converting to lowercase, removing punctuation,
    articles, and extra whitespace.

    Args:
        s (str): The answer string to normalize.

    Returns:
        str: The normalized answer string.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def get_tokens(s: str) -> List[str]:
    """
    Tokenize the string after normalizing.

    Args:
        s (str): The string to tokenize.

    Returns:
        List[str]: List of tokens from the normalized string.
    """
    if not s:
        return []
    return normalize_answer(s).split()


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def compute_em(a_gold: str, a_pred: str) -> int:
    """
    Compute exact match between the gold answer and the prediction.

    Args:
        a_gold (str): The ground truth answer.
        a_pred (str): The predicted answer.

    Returns:
        int: 1 if the answers match exactly, 0 otherwise.
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def compute_f1(a_gold: str, a_pred: str) -> float:
    """
    Compute F1 score between the gold answer and the prediction.

    Args:
        a_gold (str): The ground truth answer.
        a_pred (str): The predicted answer.

    Returns:
        float: The F1 score between the gold and predicted answers.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def compute_avna(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute answer vs. no-answer accuracy.

    Args:
        prediction (str): The predicted answer.
        ground_truths (List[str]): A list of ground truth answers.

    Returns:
        float: 1 if both prediction and ground truth are either both answers or both no-answers, 0 otherwise.
    """
    return float(bool(prediction) == bool(ground_truths))


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def eval_dicts(gold_dict: Dict[str, Any], pred_dict: Dict[str, str]) -> Dict[str, float]:
    """
    Evaluate the predictions against the ground truth and compute various metrics.

    Args:
        gold_dict (Dict[str, Any]): Dictionary containing the ground truth answers.
        pred_dict (Dict[str, str]): Dictionary containing the predicted answers.

    Returns:
        Dict[str, float]: Dictionary with metrics ('EM', 'F1', 'AvNA') and their respective values.
    """
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]['answers']
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        avna += compute_avna(prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    eval_dict['AvNA'] = 100. * avna / total

    return eval_dict


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        """Reset meter."""
        self.__init__()

    def update(self, val: float, num_samples: int = 1) -> None:
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model: nn.Module, num_updates: int) -> None:
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model: nn.Module) -> None:
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model: nn.Module) -> None:
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir: str, max_checkpoints: int, metric_name: str,
                 maximize_metric: bool = False, log: Union[None, Any] = None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val: float) -> bool:
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message: str) -> None:
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step: int, model: nn.Module, metric_val: float, device: torch.device) -> None:
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name':  model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step':        step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
def train(logger: Any,
          eval_steps: int,
          train_loader: Any,
          val_loader: Any,
          num_epochs: int,
          num_train_samples: int,
          device: torch.device,
          optimizer: torch.optim.Optimizer,
          scheduler: Any,
          ema: EMA,
          step: int,
          model: nn.Module,
          tbx: Any,
          saver: CheckpointSaver,
          max_grad_norm: float,
          val_eval_file: str,
          best_metric_name: str,
          max_ans_len: int) -> None:
    """
    Train a model for a specified number of epochs.

    Args:
        logger (Any): Logger instance for tracking training progress.
        eval_steps (int): Number of steps to evaluate and save checkpoints.
        train_loader (Any): DataLoader instance for the training dataset.
        val_loader (Any): DataLoader instance for the validation dataset.
        num_epochs (int): Total number of epochs to train.
        num_train_samples (int): Number of training samples.
        device (torch.device): Device to which the model is assigned (e.g., 'cuda' or 'cpu').
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (Any): Learning rate scheduler.
        ema (EMA): Exponential moving average utility.
        step (int): Current step of the training.
        model (nn.Module): Model instance to be trained.
        tbx (Any): TensorBoard writer.
        saver (CheckpointSaver): Utility to save model checkpoints.
        max_grad_norm (float): Maximum gradient norm (for gradient clipping).
        val_eval_file (str): Path to the validation evaluation file.
        best_metric_name (str): Name of the metric to determine the best model.
        max_ans_len (int): Maximum length of the predicted answer.

    Returns:
        None: This function doesn't return anything; it trains the model in-place.
    """

    # Train
    logger.info('Training...')
    steps_till_eval = eval_steps
    epoch = step // num_train_samples
    while epoch != num_epochs:
        epoch += 1
        logger.info(f'Starting epoch {epoch}...')
        with torch.enable_grad():
            progress_bar = tqdm(train_loader)
            for i, e in enumerate(progress_bar):
                cw_idxs = e['context'].T
                qw_idxs = e['question'].T
                cc_idxs = e['context_chars']
                qc_idxs = e['question_chars']
                c_pos = e['context_pos_emb_mat']
                q_pos = e['question_pos_emb_mat']
                ce_idxs = e['context_ent_idxs']
                qe_idxs = e['question_ent_idxs']
                id = e['id']
                y1 = e['ans_start']
                y2 = e['ans_end']
                start_points = e['start_points']
                end_points = e['end_points']
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)

                if model.module.model_type != 'base':
                    cc_idxs = cc_idxs.to(device)
                    qc_idxs = qc_idxs.to(device)

                if model.module.model_type == 'pro':
                    c_pos = c_pos.to(device)
                    q_pos = q_pos.to(device)
                    ce_idxs = ce_idxs.to(device)
                    qe_idxs = qe_idxs.to(device)

                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                if model.module.model_type == 'original':
                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                elif model.module.model_type == 'base':
                    log_p1, log_p2 = model(cw_idxs, qw_idxs)
                else:
                    log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs, c_pos, q_pos, ce_idxs, qe_idxs)

                # log_p1, log_p2 =  log_p1.to(torch.float32) , log_p2.to(torch.float32)
                y1, y2 = torch.stack(y1), torch.stack(y2)
                y1, y2 = torch.nan_to_num(y1), torch.nan_to_num(y2)
                y1, y2 = y1.to(device), y2.to(device)
                y1, y2 = y1.to(torch.int64), y2.to(torch.int64)

                loss1 = nn.NLLLoss()
                loss = loss1(log_p1, y1) + loss1(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step = i * batch_size
                progress_bar.update(1)
                progress_bar.set_postfix(epoch = epoch,
                                         NLL = loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = eval_steps

                    # Evaluate and save checkpoint
                    logger.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, val_loader, device,
                                                  val_eval_file,
                                                  max_ans_len)
                    saver.save(step, model, results[best_metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    logger.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    logger.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    """visualize(tbx,
                              pred_dict = pred_dict,
                              eval_path = val_eval_file,
                              step = step,
                              split = 'dev',
                              num_visuals = num_visuals)"""
# --------------------------------------------------------------------------------------------------------------------------------------------


