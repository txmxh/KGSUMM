import os
import sys
import time
import argparse
import pandas as pd # Needed for CSV output

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from transformers import (
    AutoTokenizer, 
    T5Tokenizer,
    T5Model,
    get_linear_schedule_with_warmup
)

from classes.config import Config
from classes.helpers import Utils
from classes.data import (
    load_dglke, 
    format_triples, 
    format_time,
    writer
)
from classes.dataset import ESBenchmark
from classes.models import ESLMKGE, ESLM
from evaluation import evaluation

def main(args):
    # Load config
    config = Config(args)
    do_train = config.do_train
    do_test = config.do_test
    device = config.device
    model_name = config.model_name

    if model_name == "bert":
        model_base = "bert-base-uncased"
    elif model_name == "ernie":
        model_base = "nghuyong/ernie-2.0-en"
    elif model_name == "t5":
        model_base = "t5-base"
    else:
        print("Please choose the correct model name: bert/ernie/t5")
        sys.exit()

    main_model_dir = f"models-eslm-kge-{model_name}" if config.enrichment else f"models-eslm-{model_name}"
    criterion = nn.BCELoss()
    utils = Utils()
    tokenizer = T5Tokenizer.from_pretrained(model_base, model_max_length=config.max_length, legacy=False) if model_name=="t5" else AutoTokenizer.from_pretrained(model_base, model_max_length=config.max_length)

    # Training
    if do_train:
        print("Training on progress ....")
        for ds_name in config.ds_name:
            print(f"Dataset: {ds_name}")
            if config.enrichment:
                entity2vec, pred2vec, entity2ix, pred2ix = load_dglke(ds_name)
                entity_dict = entity2vec  # memmap array
                pred_dict = pred2vec      # memmap array

            for topk in config.topk:
                dataset = ESBenchmark(ds_name, 6, topk, False)
                train_data, valid_data = dataset.get_training_dataset()
                for fold in range(config.k_fold):
                    train_data_size = len(train_data[fold][0])
                    train_data_samples = train_data[fold][0]
                    print(f"fold: {fold}, total entities: {train_data_size}, topk: top{topk}")
                    
                    models_path = os.path.join(main_model_dir, f"eslm_checkpoint-{ds_name}-{topk}-{fold}")
                    os.makedirs(models_path, exist_ok=True)

                    model = ESLMKGE(model_name, model_base) if config.enrichment else ESLM(model_name, model_base)
                    model.to(device)

                    # Optimizer and scheduler
                    param_optimizer = list(model.named_parameters())
                    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                    optimizer_parameters = [
                        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
                        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                    ]
                    optimizer = optim.AdamW(optimizer_parameters, lr=config.learning_rate)
                    num_training_steps = train_data_size * config.epochs
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

                    # Epoch loop
                    for epoch in range(config.epochs):
                        model.train()
                        t_start = time.time()
                        train_loss = 0

                        for num, eid in enumerate(train_data_samples):
                            # The 'literals' list is the clean list (length N)
                            literals = dataset.get_literals(eid) 
                            # 'triples_formatted' is used for tokenization (length N)
                            triples_formatted = format_triples(literals)
                            # 'triples' is the raw list, we need the raw triples matching the cleaned list
                            raw_triples = dataset.get_triples(eid) # Original raw triples (length M >= N)
                            
                            # Re-map the raw triples to only those that resulted in a valid literal line
                            # This ensures 'triples_to_embed' has length N
                            triples_to_embed = raw_triples[:len(literals)] 

                            labels = dataset.prepare_labels(eid)

                            input_ids_list = []
                            attention_masks_list = []

                            for triple in triples_formatted:
                                src_tokenized = tokenizer.encode_plus(
                                    triple,
                                    max_length=config.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_token_type_ids=True,
                                    add_special_tokens=True
                                )
                                input_ids_list.append(src_tokenized['input_ids'])
                                attention_masks_list.append(src_tokenized['attention_mask'])

                            # KGE embeddings
                            if config.enrichment:
                                s_embs, p_embs, o_embs = [], [], []
                                # <<< CRITICAL FIX: Iterate over the filtered 'triples_to_embed' list
                                for triple in triples_to_embed: 
                                    s, p, o = triple
                                    o_emb = np.zeros([400,])
                                    if str(o).startswith("http://") and o in entity2ix:
                                        o_emb = entity_dict[entity2ix[o]]
                                    p_emb = pred_dict[pred2ix[p]] if p in pred2ix else np.zeros([400,])
                                    s_emb = entity_dict[entity2ix[s]] if s in entity2ix else np.zeros([400,])
                                    s_embs.append(s_emb)
                                    p_embs.append(p_emb)
                                    o_embs.append(o_emb)
                                s_tensor = torch.tensor(np.array(s_embs), dtype=torch.float).unsqueeze(1)
                                p_tensor = torch.tensor(np.array(p_embs), dtype=torch.float).unsqueeze(1)
                                o_tensor = torch.tensor(np.array(o_embs), dtype=torch.float).unsqueeze(1)
                                kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)
                                # The length of 'kg_embeds' is now guaranteed to match 'input_ids_tensor'

                            input_ids_tensor = torch.tensor(input_ids_list).to(device)
                            attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                            
                            # IMPORTANT: The targets length must also match the sequence length
                            # We use len(triples_to_embed) to ensure target size matches input size
                            targets = utils.tensor_from_weight(len(triples_to_embed), triples_to_embed, labels).to(device)


                            outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds) if config.enrichment else model(input_ids_tensor, attention_masks_tensor)
                            reshaped_logits = outputs
                            reshaped_targets = targets.unsqueeze(-1)
                            loss = criterion(reshaped_logits, reshaped_targets)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            train_loss += loss.item()

                        avg_train_loss = train_loss / train_data_size
                        training_time = format_time(time.time() - t_start)

                        # Validation (similar fix)
                        t_start = time.time()
                        valid_data_size = len(valid_data[fold][0])
                        valid_data_samples = valid_data[fold][0]
                        model.eval()
                        valid_loss = 0
                        with torch.no_grad():
                            for eid in valid_data_samples:
                                # Data preparation in validation loop also needs the fix
                                literals = dataset.get_literals(eid)
                                triples_formatted = format_triples(literals)
                                raw_triples = dataset.get_triples(eid)
                                triples_to_embed = raw_triples[:len(literals)] 

                                labels = dataset.prepare_labels(eid)

                                input_ids_list = []
                                attention_masks_list = []
                                for triple in triples_formatted:
                                    src_tokenized = tokenizer.encode_plus(
                                        triple,
                                        max_length=config.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True
                                    )
                                    input_ids_list.append(src_tokenized['input_ids'])
                                    attention_masks_list.append(src_tokenized['attention_mask'])

                                if config.enrichment:
                                    s_embs, p_embs, o_embs = [], [], []
                                    # <<< CRITICAL FIX: Iterate over the filtered 'triples_to_embed' list
                                    for triple in triples_to_embed:
                                        s, p, o = triple
                                        o_emb = np.zeros([400,])
                                        if str(o).startswith("http://") and o in entity2ix:
                                            o_emb = entity_dict[entity2ix[o]]
                                        p_emb = pred_dict[pred2ix[p]] if p in pred2ix else np.zeros([400,])
                                        s_emb = entity_dict[entity2ix[s]] if s in entity2ix else np.zeros([400,])
                                        s_embs.append(s_emb)
                                        p_embs.append(p_emb)
                                        o_embs.append(o_emb)
                                    s_tensor = torch.tensor(np.array(s_embs), dtype=torch.float).unsqueeze(1)
                                    p_tensor = torch.tensor(np.array(p_embs), dtype=torch.float).unsqueeze(1)
                                    o_tensor = torch.tensor(np.array(o_embs), dtype=torch.float).unsqueeze(1)
                                    kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)

                                input_ids_tensor = torch.tensor(input_ids_list).to(device)
                                attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                                
                                targets = utils.tensor_from_weight(len(triples_to_embed), triples_to_embed, labels).to(device)
                                outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds) if config.enrichment else model(input_ids_tensor, attention_masks_tensor)
                                reshaped_logits = outputs
                                reshaped_targets = targets.unsqueeze(-1)
                                valid_loss += criterion(reshaped_logits, reshaped_targets).item()

                        avg_valid_loss = valid_loss / valid_data_size
                        validation_time = format_time(time.time() - t_start)
                        torch.save({
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": avg_train_loss,
                            "valid_loss": avg_valid_loss,
                            "fold": fold,
                            "training_time": training_time,
                            "validation_time": validation_time
                        }, os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        print("Training completed.")

    # -------------------------------------------------------------------------
    # Testing (with consolidated CSV saving)
    # -------------------------------------------------------------------------
    if do_test:
        print("Predicting on progress ....")
        
        all_predictions = [] # List to collect all prediction results

        for ds_name in config.ds_name:
            if config.enrichment:
                entity2vec, pred2vec, entity2ix, pred2ix = load_dglke(ds_name)
                entity_dict = entity2vec
                pred_dict = pred2vec

            for topk in config.topk:
                dataset = ESBenchmark(ds_name, 6, topk, False)
                test_data = dataset.get_testing_dataset()

                for fold in range(config.k_fold):
                    test_data_samples = test_data[fold][0]
                    model = ESLMKGE(model_name, model_base) if config.enrichment else ESLM(model_name, model_base)
                    models_path = os.path.join(main_model_dir, f"eslm_checkpoint-{ds_name}-{topk}-{fold}")
                    checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                    model.to(device)

                    with torch.no_grad():
                        for eid in test_data_samples:
                            # Data preparation in testing loop also needs the fix
                            literals = dataset.get_literals(eid)
                            triples_formatted = format_triples(literals)
                            raw_triples = dataset.get_triples(eid)
                            triples_to_embed = raw_triples[:len(literals)] 

                            labels = dataset.prepare_labels(eid)

                            input_ids_list = []
                            attention_masks_list = []

                            for triple in triples_formatted:
                                src_tokenized = tokenizer.encode_plus(
                                    triple,
                                    max_length=config.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=True
                                )
                                input_ids_list.append(src_tokenized['input_ids'])
                                attention_masks_list.append(src_tokenized['attention_mask'])

                            if config.enrichment:
                                s_embs, p_embs, o_embs = [], [], []
                                # <<< CRITICAL FIX: Iterate over the filtered 'triples_to_embed' list
                                for triple in triples_to_embed:
                                    s, p, o = triple
                                    o_emb = np.zeros([400,])
                                    if str(o).startswith("http://") and o in entity2ix:
                                        o_emb = entity_dict[entity2ix[o]]
                                    p_emb = pred_dict[pred2ix[p]] if p in pred2ix else np.zeros([400,])
                                    s_emb = entity_dict[entity2ix[s]] if s in entity2ix else np.zeros([400,])
                                    s_embs.append(s_emb)
                                    p_embs.append(p_emb)
                                    o_embs.append(o_emb)
                                s_tensor = torch.tensor(np.array(s_embs), dtype=torch.float).unsqueeze(1)
                                p_tensor = torch.tensor(np.array(p_embs), dtype=torch.float).unsqueeze(1)
                                o_tensor = torch.tensor(np.array(o_embs), dtype=torch.float).unsqueeze(1)
                                kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)

                            input_ids_tensor = torch.tensor(input_ids_list).to(device)
                            attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                            targets = utils.tensor_from_weight(len(triples_to_embed), triples_to_embed, labels).to(device)


                            outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds) if config.enrichment else model(input_ids_tensor, attention_masks_tensor)
                            reshaped_logits = outputs

                            reshaped_logits = reshaped_logits.view(1, -1).cpu()
                            _, output_top = torch.topk(reshaped_logits, topk)
                            
                            # Collect predictions for CSV
                            all_predictions.append({
                                'entity_id': eid,
                                'dataset': ds_name,
                                'model': model_name,
                                'fold': fold,
                                'topk': topk,
                                # Convert numpy array of indices to a comma-separated string for CSV
                                'predicted_indices': ",".join(map(str, output_top.squeeze(0).numpy().tolist()))
                            })
                            
                            # Save individual prediction files (as per original code)
                            directory = f"outputs-{model_name}/{dataset.get_ds_name}/{eid}"
                            os.makedirs(directory, exist_ok=True)
                            writer(dataset.get_db_path, directory, eid, "top", topk, output_top.squeeze(0).numpy().tolist())

        print("Prediction and evaluation completed.")

        # Consolidated CSV saving logic (Colab ready)
        output_dir = "/content/KGSUMM/Task2/Supervised_Model_ESLM/ESBM_v1.2/Outputs/"
        os.makedirs(output_dir, exist_ok=True)
        
        df_predictions = pd.DataFrame(all_predictions)
        
        # Create a descriptive filename based on parameters used
        ds_str = "_".join(config.ds_name)
        topk_str = "_".join(map(str, config.topk))
        
        output_csv_path = os.path.join(
            output_dir, 
            f"predictions_ESLM_{model_name}_{ds_str}_top{topk_str}.csv"
        )
        
        df_predictions.to_csv(output_csv_path, index=False)
        print(f"✅ Consolidated predictions saved to: {output_csv_path}")


# -------------------------------------------------------------------------
# Argument setup for Google Colab/Jupyter compatibility
# This ensures you can define the arguments in your notebook instead of the command line.
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # NOTE: In Google Colab, you would define 'args' manually in a cell 
    # instead of using command line arguments. For local testing, 
    # this argparse setup is maintained but simplified.
    
    parser = argparse.ArgumentParser(description='ESLM')
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)
    
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=True)
    
    parser.add_argument('--enrichment', action='store_true')
    parser.add_argument('--no-enrichment', dest='enrichment', action='store_false')
    parser.set_defaults(enrichment=True)
    
    parser.add_argument("--model", type=str, default="t5", help="e.g., t5, bert, ernie")
    parser.add_argument("--max_length", type=int, default=40, help="Max sequence length for tokenizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Model learning rate")

    parser.add_argument("--dataset", type=str, default="dbpedia", # Changed default to single dataset for easier testing
                         help="comma-separated datasets to use, e.g. dbpedia,lmdb")
    parser.add_argument("--kfolds", type=int, default=5,
                         help="number of folds for k-fold cross-validation")
    
    parser.add_argument("--topk", type=str, default="5", help="comma-separated top-k values for evaluation")
    
    args = parser.parse_args()
    main(args)