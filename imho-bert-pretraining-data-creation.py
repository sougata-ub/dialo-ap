""" Code to format and create data for pre-training Ampersand classifiers """
import math
import pickle
import pandas as pd
from tqdm import tqdm
import re
from transformers import BertTokenizerFast, RobertaTokenizerFast
import transformers
import argparse
import json
import numpy as np

transformers.logging.set_verbosity_error()


def match_imo(text):
    return re.search(r"\bimo\b", text.lower()) is not None or re.search(r"\bimho\b", text.lower()) is not None


def get_tokenizer(tokenizer_name):
    if "bert" in tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        print("Bert Tokenizer Loaded!")
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        print("Roberta Tokenizer Loaded!")
    return tokenizer


def init_tokenized_dict():
    tokenized_dict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'nsp_labels': []}
    return tokenized_dict


def create_training_data(file_path, tokenizer, pre_processed_file_name, shard_files):

    tokenized_dict = init_tokenized_dict()

    imho = pd.read_csv(file_path, header=None, on_bad_lines='skip', sep="\t")
    imho = imho.dropna()
    imho.columns = ["sentence_a", "sentence_b"]
    print("IMHO Shape:\n\n", imho.shape)

    sentence_a, sentence_b, labels = [], [], []
    for ix, row in tqdm(imho.iterrows()):
        s_a, s_b = row["sentence_a"], row["sentence_b"]
        lbl = 0
        if match_imo(s_a) and not match_imo(s_b):
            lbl = 1
        s_a = s_a.replace("imo", " ").replace("imho", " ").replace("IMO", " ").replace("IMHO", " ").strip()
        s_b = s_b.replace("imo", " ").replace("imho", " ").replace("IMO", " ").replace("IMHO", " ").strip()
        sentence_a.append(s_a)
        sentence_b.append(s_b)
        labels.append(lbl)

    st = 0
    k = 50000
    n_iter = math.ceil(len(sentence_a) / k)
    shard_size = math.ceil(len(sentence_a) / (7 if shard_files else 1))
    shard_counter, ix = 0, 1

    for _ in tqdm(range(n_iter)):
        inputs = tokenizer(sentence_a[st:st + k], sentence_b[st:st + k], max_length=200, truncation=True)

        tokenized_dict["input_ids"].extend(inputs.input_ids)
        tokenized_dict["token_type_ids"].extend(inputs.token_type_ids)
        tokenized_dict["attention_mask"].extend(inputs.attention_mask)
        tokenized_dict["nsp_labels"].extend(labels[st:st + k])

        shard_counter += len(inputs.input_ids)
        st += k

        if shard_counter > shard_size:
            print("Saving Shard!")
            fname = pre_processed_file_name.replace(".pkl", "_shard_"+str(ix)+".pkl")
            pickle.dump(tokenized_dict, open(fname, "wb"))
            print("Shard saved to:", fname)
            tokenized_dict = init_tokenized_dict()
            shard_counter = 0
            ix += 1

    if len(tokenized_dict["input_ids"]) > 0:
        print("Saving residual", len(tokenized_dict["input_ids"]), "data points.")
        fname = pre_processed_file_name.replace(".pkl", "_shard_" + str(ix) + ".pkl")
        pickle.dump(tokenized_dict, open(fname, "wb"))

    # pickle.dump(tokenized_dict, open(pre_processed_file_name, "wb"))
    # print("Saved Tokenized Data to Path:", pre_processed_file_name)


def create_qr_data(raw_file_path, tokenizer, fname):
    with open(raw_file_path) as f:
        quote_lm_data = [json.loads(i) for i in f.readlines()]

    qr_pairs = [[i["target"], i["callout"]] for i in quote_lm_data]
    print("Dataset size:", len(qr_pairs))

    all_idx = list(range(len(qr_pairs)))
    training_pairs = []
    for ix, i in tqdm(enumerate(qr_pairs)):
        random_idx = np.random.choice(all_idx)
        if ix != random_idx:
            training_pairs.append([i[0], qr_pairs[random_idx][1], 0])
        training_pairs.append([i[0], i[1], 1])

    print("All pair size:", len(training_pairs))

    sentence_a, sentence_b, labels = [], [], []
    for pair in tqdm(training_pairs):
        sentence_a.append(pair[0].strip())
        sentence_b.append(pair[1].strip())
        labels.append(pair[-1])

    st, k = 0, 50000
    n_iter = math.ceil(len(sentence_a) / k)
    tokenized_dict = init_tokenized_dict()

    for _ in tqdm(range(n_iter)):
        inputs = tokenizer(sentence_a[st:st + k], sentence_b[st:st + k], max_length=200, truncation=True)
        tokenized_dict["input_ids"].extend(inputs.input_ids)
        tokenized_dict["token_type_ids"].extend(inputs.token_type_ids)
        tokenized_dict["attention_mask"].extend(inputs.attention_mask)
        tokenized_dict["nsp_labels"].extend(labels[st:st + k])
        st += k

    pickle.dump(tokenized_dict, open(fname, "wb"))


def main():
    base_transformer = "bert-base-uncased"
    raw_file_path = "/home/argumentation/imho+context.tsv"
    pre_processed_file_name = "/home/argumentation/reddit-argument-parser/data/imho+context-" + base_transformer + "_tokenized_dict.pkl"
    shard_files = "true"
    dataset = "imho"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_transformer", type=str, help="Pretrained transformer.", default=base_transformer)
    parser.add_argument("--raw_file_path", type=str, help="raw_file_path", default=raw_file_path)
    parser.add_argument("--pre_processed_file_name", type=str, help="pre_processed_file_name",
                        default=pre_processed_file_name)
    parser.add_argument("--shard_files", type=str, help="shard_files",
                        default=shard_files)
    parser.add_argument("--dataset", type=str, help="dataset", default=dataset)

    argv = parser.parse_args()
    base_transformer = argv.base_transformer
    raw_file_path = argv.raw_file_path
    pre_processed_file_name = argv.pre_processed_file_name
    shard_files = True if argv.shard_files == "true" else False
    dataset = argv.dataset

    tokenizer = get_tokenizer(base_transformer)
    if dataset == "imho":
        create_training_data(raw_file_path, tokenizer, pre_processed_file_name, shard_files)
    else:
        create_qr_data(raw_file_path, tokenizer, pre_processed_file_name)
    print("Raw data tokenized and saved!\n")


if __name__ == '__main__':
    main()
