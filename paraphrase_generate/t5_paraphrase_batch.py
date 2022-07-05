import torch
from transformers import T5ForConditionalGeneration,T5TokenizerFast
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import GPUtil
import re, regex
import json, sys, regex
import argparse
import logging
import glob
import os
from tqdm import tqdm, trange
import pandas as pd
import torch.nn as nn

global device, device_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_ids = GPUtil.getAvailable(limit = 4)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = self.data.content
        self.labels = self.data.label
        self.max_len = max_len
        self.tweet_ids = self.data.tweet_id

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        label = str(self.labels[index])
        tw_id = str(self.tweet_ids[index])
        
        input_st = "paraphrase: " + comment_text + " </s>"
        encoding = self.tokenizer.encode_plus(input_st,pad_to_max_length=True, 
                                         return_tensors="pt", max_length=self.max_len,
                                        truncation=True)

        ids = encoding['input_ids']
        mask = encoding['attention_mask']

        return {
            'org_text': comment_text,
            'input_ids': ids,
            'attention_mask': mask,
            'label': label,
            'tweet_id': tw_id
        }

def regular_encode(file_path, tokenizer, shuffle=True, num_workers = 1, batch_size=64, maxlen = 32, mode = 'train'):
    
    # if we are in train mode, we will load two columns (i.e., text and label).
    if mode == 'train':
        # Use pandas to load dataset
        df = pd.read_csv(file_path, delimiter='\t',header=0, names=['tweet_id', 'label', 'content'], encoding='utf-8', quotechar=None, quoting=3)
    
    # if we are in predict mode, we will load one column (i.e., text).
    elif mode == 'predict':
        df = pd.read_csv(file_path, delimiter='\t',header=0, names=['tweet_id','content'])
    else:
        print("the type of mode should be either 'train' or 'predict'. ")
        return
        
    print("{} Dataset: {}".format(file_path, df.shape))
    
    custom_set = CustomDataset(df, tokenizer, maxlen)
    
    dataset_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}

    batch_data_loader = DataLoader(custom_set, **dataset_params)
    
    return batch_data_loader


def tweet_normalizer(txt):
    # remove duplicates
    temp_text = regex.sub("(USER\s+)+","USER ", txt)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    temp_text = re.sub("[\r\n\f\t]+","",temp_text)
    temp_text = re.sub(r"\s+"," ", temp_text)
    temp_text = regex.sub("(USER\s+)+","USER ", temp_text)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    
    return temp_text

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--file_name", default="train.tsv", type=str, required=True,
                        help="type")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--model_name_or_path", default='UBC-NLP/ptsm_t5_paraphraser', type=str, required=True,
                    help="Path to pre-trained model or shortcut name")
    
    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size GPU/CPU.")

    parser.add_argument("--num_workers", default=1, type=int,
                        help="Total number of num_workers.")

    parser.add_argument("--num_return", default=10, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_k", default=50, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Total number of generated paraphrases per tweet.")


    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 0 if device=="cpu" else torch.cuda.device_count()

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)  
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name_or_path)

    if torch.cuda.is_available():
        if n_gpu == 1:
            model = model.to(device)
        else:
            torch.backends.cudnn.benchmark = True
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model

    train_file = os.path.join(args.data_dir, args.file_name)

    train_dataloader = regular_encode(train_file, tokenizer, batch_size=args.batch_size, maxlen = args.max_seq_length)

    task_name = args.data_dir.split("/")[-1]
    
    output_file = os.path.join(args.output_dir, task_name+"_paraphrase.json")

    model.eval()
    for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, attention_masks = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        labels = batch["label"]
        tweet_ids = batch["tweet_id"]
        org_texts = batch["org_text"]

        beam_outputs = model.generate(
                input_ids=input_ids.squeeze(1), attention_mask=attention_masks.squeeze(1),
                do_sample=True,
                max_length=args.max_seq_length,
                top_k=args.top_k,
                top_p=args.top_p,
                early_stopping=True,
                num_return_sequences=args.num_return
            )

        beam_outputs = beam_outputs.cpu()
        org_inputs = input_ids.squeeze(1).cpu().numpy().tolist()

        final_outputs = [tokenizer.decode(x, skip_special_tokens=True,clean_up_tokenization_spaces=True) for x in beam_outputs]

        output_lines = []
        for ind in range(len(labels)):
            org_tweet = tokenizer.decode(org_inputs[ind], skip_special_tokens=True,clean_up_tokenization_spaces=True)
            org_tweet = tweet_normalizer(org_tweet.replace("paraphrase: ", ""))

            paraphrases = final_outputs[ind * args.num_return : (ind+1) * args.num_return]
            paraphrases = [tweet_normalizer(x) for x in paraphrases]
            
            paraphrases = [x for x in paraphrases if x.lower() != org_tweet.lower()]
            paraphrases = list(set(paraphrases))
            
            output_all = {}
            output_all["tweet_id"] = tweet_ids[ind]
            output_all["original_tweet"] = org_texts[ind]
            output_all["label"] = labels[ind]
            
            output_all["paraphrase"] = paraphrases
            
            output_lines.append(json.dumps(output_all)+"\n")
            
        with open(output_file, "a") as out_f:
            out_f.writelines(output_lines)
    

if __name__ == "__main__":
    main()
