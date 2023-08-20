import numpy as np
import random
import torch
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import json
from tensorboardX import SummaryWriter
import train_utils
import SQuAD_dataset
from model.BIDAF import BiDAF
import utils
import pandas as pd
import os
import string

base_dir = ""
training_run_name = ""


def main(args):

    """
    Main training routine for the given model and dataset.
    """

    # Set up logging and devices
    args.save_dir, uid = train_utils.get_save_dir(args.save_dir, args.name, training=True)
    log = train_utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = train_utils.get_available_devices()
    log.info(f'Args: {json.dumps(vars(args), indent = 4, sort_keys = True)}')
    
    config_save_dir = os.path.join(args.save_dir, f'config_{uid}.json')
    train_utils.save_args(args.__dict__, config_save_dir)
    
    
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get saver
    saver = train_utils.CheckpointSaver(args.save_dir,
                            max_checkpoints=args.max_checkpoints,
                            metric_name=args.best_metric_name,
                            maximize_metric=args.maximize_metric,
                            log = log)

    # Get data loader
    log.info('Building dataset...')

    train_df = SQuAD_dataset.get_squad_dataframe(args.train_df_file)
    val_df = SQuAD_dataset.get_squad_dataframe(args.val_df_file)

    #create vocabulary using train and validation data
    all_text = pd.concat([train_df['context'], train_df['question'], val_df['context'], val_df['question']])
    vocab = SQuAD_dataset.Vocabulary()
    vocab.build_word_vocabulary(all_text)
    vocab.build_char_vocabulary(all_text)

    #create training and validation datasets
    train_dataset = SQuAD_dataset.Train_Dataset(
        df= train_df,
        vocabulary=vocab,
        vocab_freq_threshold=args.vocab_freq_treshold,
        vocab_max_size=args.vocab_max_size,
        glove_file=args.glove_file
    )

    val_dataset =  SQuAD_dataset.Validation_Dataset(
        df=val_df,
        vocabulary=train_dataset.vocabulary,
        glove_file=args.glove_file
    )

    train_loader = SQuAD_dataset.get_train_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle_data,
        pin_memory=args.pin_memory
    )

    val_loader = SQuAD_dataset.get_valid_loader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = args.shuffle_data,
        pin_memory = args.pin_memory
    )

    # Get embeddings
    log.info('Loading embeddings...')
    print(train_dataset.vocabulary.special_tokens)
    word_vectors = utils.GloVe_embedding_matrix(
        glove_file = args.glove_file, 
        special_tokens = train_dataset.vocabulary.special_tokens
    )

    # Get model
    log.info('Building model...')
    alphabet = string.ascii_lowercase + string.digits
    model = BiDAF(model_type = args.model_type,
                  word_vectors = word_vectors,
                  hidden_size = args.hidden_size,
                  drop_prob = args.drop_prob,
                  char_channel_width=args.char_channel_width,
                  char_embed_dim=args.char_embed_dim,
                  alphabet_size=len(alphabet)
                  )
    
    model.emb.embed.weight.requires_grad = False
    model = torch.nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = train_utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()
    ema = train_utils.EMA(model, args.ema_decay)

    # Get optimizer and scheduler
    par = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = optim.Adadelta(par, args.lr,
                               weight_decay = args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    train_utils.train(
        logger = log,
        step = step,
        eval_steps = args.eval_steps,
        train_loader = train_loader,
        num_epochs = args.num_epochs,
        device = device,
        optimizer = optimizer,
        scheduler = scheduler,
        ema = ema,
        model = model,
        tbx = tbx,
        num_train_samples = len(train_dataset),
        val_loader = val_loader,
        saver = saver,
        max_grad_norm = args.max_grad_norm,
        val_eval_file = args.val_eval_file,
        best_metric_name = args.best_metric_name,
        max_ans_len = args.max_ans_len,
        num_visuals = args.num_visuals
    )



if __name__ == "__main__":
    # args = train_utils.get_args("code/code-for-run/config.json")

    TRAINING_DIR = ""
    
    # Set WD
    os.chdir(TRAINING_DIR)

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Read JSON configuration file')
    parser.add_argument('--config', help='Path to the config JSON file')
    arguments = parser.parse_args()

    # Read the JSON file
    args = train_utils.get_args(arguments.config)

    main(args)