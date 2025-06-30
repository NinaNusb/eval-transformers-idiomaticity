
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
import numpy as np
import json
import pandas as pd
from itertools import chain
import random
import spacy
import csv
import random
from contextlib import closing
import pickle
import re

from statistic_tests import compute_similarity
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda:0"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE) 

bert_large_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_large_model = BertModel.from_pretrained('bert-large-uncased').to(DEVICE) 

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base').to(DEVICE)

roberta_large_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_large_model = RobertaModel.from_pretrained('roberta-large').to(DEVICE)

multi_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
multi_model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(DEVICE)

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")



###------------------------------------- Preprocessing -------------------------------------###

def drop_rows_with_single_meaning(data_df):
    # Group df by mwe
    groups = data_df.groupby('id')
    indices_to_drop = []
    
    for _, group in groups:
        # Check if there is at least one idiomatic meaning or minimum two unique values in the 'fine_grained' column or the literal_meaning in fine_grained meaning
        if (group["_1"].unique() == 'None' and group["_2"].unique() == 'None' and group["_3"].unique() == 'None') or len(group['fine_grained'].unique()) < 2 or group['literal_meaning'].unique() not in group['fine_grained'].unique():
            # If not, add the indices of the rows in this group to the list
            indices_to_drop.extend(group.index)
    
    # Drop the rows with the indices stored in indices_to_drop
    data_df = data_df.drop(indices_to_drop)
    
    return data_df, indices_to_drop


def find_sublist(sublist, full_list, print_ignored=False):
    
    # Added for roberta
    for i in range(len(full_list) - len(sublist) + 1): # make sure there are enough tokens remaining in the sentence to potentially match entire tokenized mwe      
       
        if full_list[i:i+len(sublist)] == sublist:
            return i
    if print_ignored:
        print(f"Didn't find start_id for {sublist} in sentence: {full_list}.")
    return None


def normalize_tokens(tokens, tokenizer=None):
    """
    Normalize tokens by removing common tokenizer-specific prefixes like Ġ, ##, or #.
    Works across tokenizers such as RoBERTa, BERT, etc.
    """
    return [re.sub(r"^(Ġ|##|#)+", "", tok) for tok in tokens]


def retrieve_preprocessed_df(file,print_ignored, tokenizer= bert_tokenizer, words_to_add=None):
    with open(file, "r") as file:
        data = json.load(file)

    columns = zip(*chain(*chain(*data.values())))
    column_names = ["id", "mwe", "literal_meaning", "_1", "_2", "_3", "proper", "meta", "0/1", "fine_grained", "prior", "sentence", "after", "source"]
    data_df = pd.DataFrame({name: col for name, col in zip(column_names, columns)})

    # Drop irrelevant columns
    data_df = data_df.drop(["proper", "meta", "source"], axis=1)
    init_len = len(data_df)

    # Drop proper nouns and metaphors
    data_df = data_df[~(data_df['fine_grained'].isin(['Proper Noun', 'Meta Usage']))]
    after_type_filter_len = len(data_df)
    if print_ignored: 
        print(f"We drop {init_len - after_type_filter_len} ProperNoun and MetaUsage rows out of {init_len}.")

    # Drop MWEs that don't have both a compositional and a non-compositional meaning
    data_df, dropped_rows = drop_rows_with_single_meaning(data_df)
    after_meaning_filter_len = len(data_df)
    if print_ignored:
        print(f"Number of dropped single-meaning-mwe rows: {after_type_filter_len - after_meaning_filter_len}")

    # For position testing purpose
    if words_to_add is not None:
        data_df['sentence'] = data_df['sentence'].apply(lambda x: words_to_add + x)

    # Tokenize sentences and mwe
    data_df['prior_tokens'] = data_df['prior'].apply(lambda x: tokenizer.tokenize(x))
    data_df['sentence_tokens'] = data_df['sentence'].apply(lambda x: tokenizer.tokenize(x))
    data_df['after_tokens'] = data_df['after'].apply(lambda x: tokenizer.tokenize(x))
    data_df['tokenized_mwe'] = data_df['mwe'].apply(lambda x: x.split())  # Split the MWE string into tokens


    # Normalize sentence tokens to get rid of prefixes (e.g., Ġ)
    data_df['normalized_sentence_tokens'] = data_df['sentence_tokens'].apply(lambda tokens: normalize_tokens(tokens, tokenizer))
    
    # Drop MWEs not found in normalized sentence
    before_mwe_sublist_filter_len = len(data_df)
    def mwe_in_sentence(row, print_ignored=False):
        return find_sublist(row['mwe'].split(), [t.lower() for t in row['normalized_sentence_tokens']], print_ignored) is not None

    print("Only MWEs that are tokenized like row['mwe'].split() have been retrieved at this step. \
     Specifically, we kept MWEs split into 2 tokens, in the singular form, and no different form from the MWE string ('running' != 'run'). \
     To see the ignored rows, set print_ignored=True.")

    data_df = data_df[data_df.apply(mwe_in_sentence, axis=1)]
    after_mwe_sublist_filter_len = len(data_df)
    if print_ignored:
        print(f"{before_mwe_sublist_filter_len - after_mwe_sublist_filter_len} rows dropped where MWE not found as sublist in sentence.")
    
    # Count remaining sentences
    if print_ignored:
        total_sentences = sum(len(data_df[col]) for col in ['prior', 'sentence', 'after'])
        print(f"There are {total_sentences:,} remaining sentences in the dataset.")


    data_df = data_df.reset_index(drop=True)

    return data_df

###------------------------------------- Encoding -------------------------------------###

def encode_paragraphs(paragraphs_batches, layer_id, tokenizer= bert_tokenizer, model= bert_model):
    """
    Function that returns the unpadded encoded paragraph batches at a given layer, with the output layer normalized.

    Args:
    - paragraphs_batches: a list of batches of paragraphs to be encoded
    - layer_id: the index of the layer in the model's hidden states to use for encoding
    """
    unpadded_encoded_tokens = []
    model.eval()

    for batch in paragraphs_batches:
        inputs = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Remove padding tokens using attention_mask
        layer = outputs.hidden_states[layer_id]   # Shape (batch_size, seq_length, embedding_size)
        batch_without_padding = layer[inputs.attention_mask.bool()]

        unpadded_encoded_tokens.append(batch_without_padding)

    return unpadded_encoded_tokens 



def save_model_outputs(model_outputs, checkpoint, file_path):
    # From Nazanin
    """
    Saves the attention weights and hidden states to a pickle file.

    - model_outputs: (Dict) Dictionary containing 
        1. the masked token identity, 
        2. the indices of the masked tokens in each sentence of the batch, 
        3. the decoded masked sequences (str), 
        4. attentions
        5. hidden states 
    - checkpoint: (str) The training step.
    - file_path: (str) The path to the pickle file.
    """
    try:
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = []

    for batch_i in model_outputs:
        masked_token_identity = batch_i['masked_token_identity']
        masked_token_indices = batch_i['masked_token_indices']
        sequences = batch_i['sequences']
        attentions = batch_i['attentions']          # 12 tensors of shape [batch_size, n_heads, seq_len, seq_len]
        hidden_states = batch_i['hidden_states']    # 13 tensors of shape [batch_size, seq_len, hidden_size]

        for i in range(len(sequences)):
            masked_token_idx = masked_token_indices[i]
            row = {
                'step': checkpoint,
                'masked_token_identity': masked_token_identity,
                'sequence': sequences[i],
                '1st_layer_attns': attentions[0][i, :, masked_token_idx, :],
                '6th_layer_attns': attentions[5][i, :, masked_token_idx, :],
                'last_layer_attns': attentions[-1][i, :, masked_token_idx, :],
                '1st_layer_hs': hidden_states[0][i, masked_token_idx, :],
                '6th_layer_hs': hidden_states[6][i, masked_token_idx, :],
                'last_layer_hs': hidden_states[-1][i, masked_token_idx, :]
            }
            data.append(row)

    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)



def normalize_embeds(unpadded_encoded_tokens):
    """
    Function that normalizes sentence representations using the standard scaler.

    Args:
    - unpadded_encoded_tokens: a list of batches of unpadded encoded tokens
    """
    # Concatenate all the batches into a single tensor
    all_tokens = torch.cat(unpadded_encoded_tokens, dim=0)
   
    # Normalize the tokens using the standard scaler
    scaler = StandardScaler()
    normalized_tokens = scaler.fit_transform(all_tokens.cpu().numpy())
    # print(scaler.mean_)
    
    # Convert the normalized tokens back to a PyTorch tensor
    normalized_tokens = torch.from_numpy(normalized_tokens).to(DEVICE)
    
    # Split the normalized tokens back into batches
    start = 0
    normalized_batches = []
    for batch in unpadded_encoded_tokens:
        end = start + len(batch)
        normalized_batches.append(normalized_tokens[start:end])
        start = end

    return normalized_batches


def extract_sentence_embeddings(unpadded_encoded_tokens, all_sentences_token_number):
    sentence_embeddings = []
    sent_count = 0
    for i, (batch_tok_embeddings, batch_nb_tokens) in enumerate(zip(unpadded_encoded_tokens, all_sentences_token_number)):
        start = 0  # Reset start index for each batch
        for num_tokens in batch_nb_tokens:
            sent_count += 1
            end = start + num_tokens
            sent_emb = batch_tok_embeddings[start:end]
            sentence_embeddings.append(sent_emb)
            start = end

    return sentence_embeddings


def extract_sentence_attention_heads(attention_heads, all_sentences_token_number):
    sentence_attention_heads = []
    
    for batch_attention, batch_nb_tokens in zip(attention_heads, all_sentences_token_number):
        # batch_attention shape: (batch_size, num_heads, seq_length, seq_length)
        batch_size, num_heads, seq_length, _ = batch_attention.shape
        
        start = 0
        for i, num_tokens in enumerate(batch_nb_tokens):
            end = start + num_tokens
            
            # Extract attention for this sentence
            sent_attention = batch_attention[i, :, start:end, start:end]
            
            sentence_attention_heads.append(sent_attention)
            
            start = 0  # Reset start for the next sentence in the batch

    return sentence_attention_heads


def get_nb_tokens_per_sent(all_sentences, batch_size, num_batches,tokenizer=bert_tokenizer):
    all_sentences_batches = [all_sentences[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    all_sentences_tokenized = [[tokenized for row in batch for tokenized in [tokenizer.tokenize(sent) for sent in row]] for batch in all_sentences_batches]
    all_sentences_token_number = [[len(tokenized_sentence) for tokenized_sentence in batch] for batch in all_sentences_tokenized]
    return all_sentences_token_number


def retrieve_sentence_embeddings(paragraphs, batch_size, num_batches, layer_id, all_sentences_token_number, normalize, tokenizer=bert_tokenizer, model= bert_model):
    paragraphs_batches = [paragraphs[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    embeds = encode_paragraphs(paragraphs_batches, layer_id, tokenizer, model)
    if normalize:
        embeds = normalize_embeds(embeds)
    sentence_embeddings = extract_sentence_embeddings(embeds, all_sentences_token_number)
    return sentence_embeddings  



def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()



def retrieve_encoded_df(data_df, batch_size, layer_id, normalize, print_ignored, tokenizer= bert_tokenizer, model= bert_model):
    paragraphs = [' '.join(row) for row in zip(data_df['prior'], data_df['sentence'], data_df['after'])]
    all_sentences = [row for row in zip(data_df['prior'], data_df['sentence'], data_df['after'])]
    num_batches = (len(paragraphs) + batch_size - 1) // batch_size
    all_sentences_token_number = get_nb_tokens_per_sent(all_sentences, batch_size, num_batches, tokenizer)
    sentence_embeddings = retrieve_sentence_embeddings(paragraphs, batch_size, num_batches, layer_id, all_sentences_token_number, normalize, tokenizer, model)

    # Extract embeddings from sentence_embeddings list
    prior_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[::3]]
    sentence_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[1::3]]
    after_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[2::3]]

    data_df['prior_embeddings'] = prior_embeddings_list
    data_df['sentence_embeddings'] = sentence_embeddings_list
    data_df['after_embeddings'] = after_embeddings_list


    data_df = add_mwe_embeddings_to_df(data_df, print_ignored, tokenizer)
    data_df = drop_na_rows(data_df)

    return data_df

    

def retrieve_encoded_df_single_sentences(data_df, batch_size, layer_id, normalize, print_ignored, add_attentions=False,tokenizer= bert_tokenizer, model= bert_model):
    
    all_sentences = [row for row in data_df['sentence']]
    num_batches = (len(all_sentences) + batch_size - 1) // batch_size
    all_sentences_batches = [all_sentences[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    all_sentences_tokenized = [[tokenizer.tokenize(sent) for sent in batch] for batch in all_sentences_batches]
    all_sentences_token_number = [[len(tokenized_sentence) for tokenized_sentence in batch] for batch in all_sentences_tokenized]
    embeds, attention_heads = encode_paragraphs(all_sentences_batches, layer_id, tokenizer, model)
    if normalize:
        embeds = normalize_embeds(embeds)

    sentence_embeddings = extract_sentence_embeddings(embeds, all_sentences_token_number)
    
    # Extract embeddings from sentence_embeddings list
    sentence_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings]
    assert len(sentence_embeddings_list) == len(data_df), "Mismatch between number of embeddings and DataFrame rows"
    data_df['sentence_embeddings'] = sentence_embeddings_list
 
    if add_attentions:
        sentence_attentions = extract_sentence_attention_heads(attention_heads, all_sentences_token_number)
        assert len(sentence_attentions) == len(data_df), f"Mismatch between number of attention data entries ({len(sentence_attentions)}) and DataFrame rows ({len(data_df)})"
        for head in range(1, 13):
            head_data = []
            for sent_attention in sentence_attentions:
                # Convert the attention values for this head to a list
                head_attention = sent_attention[head-1].detach().cpu().numpy().tolist()
                head_data.append(head_attention)
            data_df[f'attention_head_{head}'] = head_data
        data_df['attention_shape'] = [sent_attention.shape for sent_attention in sentence_attentions]

    data_df = add_mwe_embeddings_to_df(data_df, print_ignored, tokenizer)
    data_df = drop_na_rows(data_df)

    return data_df



###------------------------------------- MWE embeddings -------------------------------------###

def create_literal_idiomatic_pairs(data_df, unique_occurrences=False):
    # Create groups of the same id (i.e. same MWE)
    groups = data_df.groupby('id')
    selected_pairs = []
    pair_info = {}
    
    # Select pairs of the same MWE
    # in which the first element's fine-grained meaning is the same as the literal meaning 
    # and the second is the same as one of the idiomatic meanings 
    for _, group in groups:
        literal_indices = []
        idiomatic_indices = []
     
        for idx in group.index:
            if group.loc[idx, 'fine_grained'].strip() == group.loc[idx, 'literal_meaning'].strip():
                literal_indices.append(idx)
            else:
                # TODO: only take _1 idiomatic meaning
                idiomatic_indices.append(idx)
        
        pair_info[group['mwe'].iloc[0]] = (len(literal_indices), len(idiomatic_indices))

        # Shuffle indices within each subgroup
        random.shuffle(literal_indices)
        random.shuffle(idiomatic_indices)
        
        if unique_occurrences: 
            # TODO: Create unique pairs where each mwe appears only once
            num_pairs = min(len(literal_indices), len(idiomatic_indices))
            for i in range(num_pairs):
                selected_pairs.append((literal_indices[i], idiomatic_indices[i]))

        else: 
            # Create unique pairs by pairing each literal mwe with each idiomatic mwe
            for lit_idx in literal_indices:
                for idi_idx in idiomatic_indices:
                    pair = (lit_idx, idi_idx)
                    if pair not in selected_pairs:
                        selected_pairs.append(pair)                      
                    else:
                        continue
                                  
    return selected_pairs, pair_info



def get_pair_info_df(pair_info):
    
    pair_df = pd.DataFrame(pair_info)
    pair_df.index.name = 'mwe'
    pair_df.index = pair_df.index.map({0: 'literal', 1: 'idiomatic'})
    pair_df = pair_df.T
    
    pair_df['difference'] = pair_df.apply(lambda row: compute_difference(row, 'literal', 'idiomatic'), axis=1)
    pair_df['most_represented'] = pair_df.apply(get_most_represented, axis=1)
    return pair_df




def compute_difference(row, col1, col2):
    return max(row[col1], row[col2]) - min(row[col1], row[col2])



def get_mean_positions(mwe_starts):
    mwe_mean_starts = {}
    for mwe, positions in mwe_starts.items():
        mean_lit = np.mean(positions['lit'])
        mean_idiom = np.mean(positions['idiom'])
        mwe_mean_starts[mwe] = mean_lit, mean_idiom

    return mwe_mean_starts

    

def get_most_represented(row):
    if row['literal'] > row['idiomatic'] :
        return 'literal'
    elif row['idiomatic'] > row['literal']:
        return 'idiomatic'
    else: 
        return 'equal'
    


def extract_mwe_embeddings(sentence_embedding, tokenized_sent, mwe, tokenizer):
  
    tokenized_sent_norm = normalize_tokens(tokenized_sent)
    # tokenized_mwe_norm = normalize_tokens(tokenizer.tokenize(mwe_string))

    # Find the sublist in the normalized tokenized sentence
    mwe_start = find_sublist(mwe.split(), tokenized_sent_norm)

    if mwe_start is not None:
        mwe_end = mwe_start + len(mwe.split())
        
        # Check original tokens match (optional, for debugging)
        if tokenized_sent_norm[mwe_start:mwe_end] != mwe.split():
            print("Mismatch in tokenized MWE extraction:")
            print("Expected:", mwe.split())
            print("Found   :", tokenized_sent_norm[mwe_start:mwe_end])

        return sentence_embedding[mwe_start:mwe_end], mwe_start

    return None



def add_mwe_embeddings_to_df(data_df, print_ignored=False, tokenizer= bert_tokenizer):
    # Initialize new columns for embeddings
    data_df['literal_embedding'] = None
    data_df['idiomatic_embedding'] = None
    data_df['mwe_start'] = None

    count_ignored = 0
    for index, row in data_df.iterrows():
        res = extract_mwe_embeddings(row['sentence_embeddings'], 
                                           row['sentence_tokens'], 
                                        #    row['tokenized_mwe'])
                                             row['mwe'], 
                                             tokenizer)
        # print(embedding)
        if res is not None:
            embedding, mwe_start = res
            # Add literal and idiomatic embeddings to DataFrame
            if row['fine_grained'] == row['literal_meaning']:
                data_df.at[index, 'literal_embedding'] = embedding
            elif row['fine_grained'] in [row['_1'], row['_2'], row['_3']]:
                data_df.at[index, 'idiomatic_embedding'] = embedding
            data_df.at[index, 'mwe_start'] = mwe_start
        else:
            # If MWE embedding extraction failed
            if print_ignored:
                print("Ignored index:", index)
            count_ignored += 1

    if print_ignored: print(f"[MWE embedding extraction] There are {count_ignored} ignored examples.")
    return data_df



def get_group_start_positions(mwe_df):
    group_starts = {'lit': [], 'idiom': []}

    # Select rows where literal_embedding is not None
    lit_rows = mwe_df[mwe_df['literal_embedding'].notnull()]
    group_starts['lit'] = lit_rows['mwe_start'].tolist()

    # Select rows where idiomatic_embedding is not None
    idiom_rows = mwe_df[mwe_df['idiomatic_embedding'].notnull()]
    group_starts['idiom'] = idiom_rows['mwe_start'].tolist()
    return group_starts




def retrieve_mwe_embeddings(mwe_set, df,tokenizer= bert_tokenizer):
    # Initialize dictionary to store MWE embeddings
    mwe_embeddings = {}

    # Iterate over each MWE in the set
    for mwe in mwe_set:
        # Select rows where the MWE's fine_grained is non compositional                             Note: currently only using a single meaning per mwe to not mix their semantics
        mwe_occurrences = df[(df['mwe'] == mwe) & (df['fine_grained'] == df['_1'])]

        # Get the row ids corresponding to the selected occurrences
        occurrence_row_ids = mwe_occurrences.index.tolist()
        # TODO:  why certain mwe have no occurrences... there are MWE that don't have an idiomatic meaning
        
        # Initialize list to store embeddings for the current mwe
        mwe_occurrence_embeddings = []
        
        for index, row in mwe_occurrences.iterrows():      
            # Extract MWE embeddings
            embeddings, _ = extract_mwe_embeddings(row['sentence_embeddings'], row['sentence_tokens'], row['mwe'], tokenizer)
        
            # Append MWE embeddings to the list
            if embeddings is not None:
                mwe_occurrence_embeddings.append(embeddings)
        
        # Add the list of embeddings and row IDs to the dictionary for the current MWE
        mwe_embeddings[mwe] = {'meaning': mwe_occurrences['_1'], 'row_ids': occurrence_row_ids, 'embeddings': mwe_occurrence_embeddings} 
   
    return mwe_embeddings



def average_mwe_embedding(mwe_embeddings, second_only, combination_type=None):
    
    averaged_embeddings = []
    occurrences = []
    meanings = []

    for mwe, description in mwe_embeddings.items():
        # print(mwe)
        meaning, _, embeddings = description.values()       
        # print([len(e) for e in embeddings])
        # print(len(embeddings)) #(nb_occ, 2, 768)
        if second_only:
            # Initialize an empty list to store the second arrays (i.e non_comp_1)
            embeds = []      
            for sublist in embeddings:
                # Access the second array in the sublist and append it to the second_arrays list
                embeds.append(sublist[1])
        else:
            if combination_type == "concatenation":
                # Concatenate both MWE vectors 
                embeds = [np.concatenate((emb[0], emb[1])) for emb in embeddings]
            elif combination_type == "average":
                # Average both vectors
                embeds = [np.mean(emb, axis=0) for emb in embeddings]
        # print([len(e) for e in embeds]) # (nb_occ, 768)
        

        # Average all combined embeddings for the current mwe
        avg_embedding = np.mean(embeds, axis=0)
                
        # Append the averaged embedding and count of occurrences
        averaged_embeddings.append(avg_embedding)
        occurrences.append(len(embeddings))
        meanings.append(meaning.iloc[0] if not meaning.empty else None)
    
    return averaged_embeddings, occurrences, meanings



def drop_na_rows(data_df, print_ignored=False):
    rows_to_drop = data_df[(data_df['literal_embedding'].isna()) & (data_df['idiomatic_embedding'].isna())].index
    if print_ignored: print(f"Dropping {len(rows_to_drop)} row(s).")
    data_df = data_df.drop(index=rows_to_drop)
    data_df = data_df.reset_index(drop=True)
    return data_df


###------------------------------------- Noun embeddings -------------------------------------###

def isolate_sentences(df):
    all_sentences = {}
    # Iterate over each row in df
    for _, row in df.iterrows(): 
        # Keep track of the mwe and all sentences
        key = row['mwe']
        # Convert NaN values to empty strings
        prior = str(row['prior']) if not pd.isnull(row['prior']) else ''
        sentence = str(row['sentence']) if not pd.isnull(row['sentence']) else ''
        after = str(row['after']) if not pd.isnull(row['after']) else ''
        # sentences = ' '.join([prior, sentence, after])
        
        if key not in all_sentences:
            all_sentences[key] = []
        for s in prior, sentence, after:
            all_sentences[key].append(s)
    return all_sentences



def extract_nouns(text, to_exclude, empty_nouns, nlp, nb_tokens, n, tokenizer= bert_tokenizer):
    # Extract nouns that are different from the mwes
    nouns = [token.lemma_ for mwe, sentences in text.items() for sent in sentences for token in nlp(sent) if token.pos_ == "NOUN" and token.lemma_ != mwe]
    
    # Only keep nouns tokenized with n token(s)
    selected = [noun for noun in nouns if len(tokenizer.tokenize(noun)) == nb_tokens]

    # Sort the list for reproductibility 
    selected = sorted(list(set(selected)), key=lambda x: x.lower())
    
    # Shuffle the list with a seed
    random.Random(42).shuffle(selected)

    # Clean list
    selected = [noun for noun in selected if noun not in to_exclude and noun not in empty_nouns]

    ignored_nouns = list(set([noun for noun in nouns if len(tokenizer.tokenize(noun)) != nb_tokens]))
    print(f"Skipped nouns are (15/{len((ignored_nouns))}): {ignored_nouns[:15]}")
    
    return (print(f"Only {len(selected)}/{n} nouns available.") or selected) if n > len(selected) else list(selected)[:n]



def find_noun_positions(tokens, nouns, nb_tokens, tokenizer= bert_tokenizer):
    positions = []
    tokenized_nouns = [tokenizer.tokenize(n) for n in nouns]

    for context_type, tokens_list in tokens.items():
        for i in range(len(tokens_list) - nb_tokens + 1):
            token_ngram = tokens_list[i:i+nb_tokens]
            for n, noun in zip(tokenized_nouns, nouns):
                if token_ngram == n: #Compare n-grams depending on nb_tokens
                    positions.append([noun, i, context_type])
                
    return positions



def extract_noun_embeddings(noun_list, data_df, cols, nb_tokens):
    noun_embeddings = {noun: [] for noun in noun_list}
    skipped = 0

    for col_name in cols:
        for row_id, row in data_df.iterrows():
            if col_name == 'prior_noun_positions' and row_id == 159 or row_id == 181: # dealing with the same problematic sentence twice
                continue
            for group in row[col_name]:
                noun, position, context_type = group                                
                noun = noun.strip()  
                
                # Compare the id of the noun compared to the number of available embeddings
                if position >= len(row[f'{context_type}_embeddings']):
                    print(noun)
                    print(position, len(row[f'{context_type}_tokens']), len(row[f'{context_type}_embeddings']))
                    print(noun, position, context_type)
                    print(data_df.at[row_id, f'{context_type}_tokens'])
                    skipped +=1
                    continue  # Skip this cell if position is out of bounds

                # Extract the embeddings based on the nb_tokens value
                embeddings = row[f'{context_type}_embeddings'][position:position + nb_tokens]
                noun_embeddings[noun].append(embeddings) 

    print(f"There are {skipped} skipped embeddings")           
    return noun_embeddings



def get_combined_embeddings(embeddings: dict):
    # Initialize lists to store averaged embeddings and counts
    combined_embeddings = []
    occurrences = []

    # Filter out empty embeddings
    filtered_embeddings = {k: v for k, v in embeddings.items() if v}

    for _, embeds in filtered_embeddings.items():
        # Compute the average embedding
        combined_embed  = np.concatenate(embeds, axis=0)
       
        combined_embeddings.append(combined_embed )
        occurrences.append(len(embeds))
    
    return combined_embeddings, occurrences



def get_sim_scores(selected_pairs, data_df):
    # Dictionary to store the computed similarities, distances, and correlations
    results = {'C_0 vs NC_0': {}, 'C_1 vs NC_1': {}, 'C_0 vs C_1': {}, 'NC_0 vs NC_1': {}}

    # Counter for ignored pairs
    ignored_pairs_count = 0

    for literal_index, idiomatic_index in selected_pairs:
        # Get the MWE embeddings for each pair
        mwe = data_df.loc[literal_index, 'mwe']
        literal_embedding = data_df.loc[literal_index, 'literal_embedding']
        idiomatic_embedding = data_df.loc[idiomatic_index, 'idiomatic_embedding']
        
        # Check if either embedding is None, if so, skip this pair
        # The reason being a lemmatization issue
        if literal_embedding is None or idiomatic_embedding is None:       
            ignored_pairs_count += 1
            continue
            
        # Compute metrics
        cosine_sims = compute_similarity(np.asarray(literal_embedding[0]), np.asarray(literal_embedding[1]), 
                                        np.asarray(idiomatic_embedding[0]), np.asarray(idiomatic_embedding[1]))

        # Update the results dictionary
        for key, cosine_sim in zip(results.keys(), cosine_sims):
            if mwe not in results[key]:
                results[key][mwe] = []
            results[key][mwe].append(cosine_sim)
    
    return results



def compute_centroids(mwe_df):      
    centroid_lit = np.mean(np.array([emb for emb in mwe_df['combined_lit'] if emb is not None]), axis=0)
    centroid_idiom = np.mean(np.array([emb for emb in mwe_df['combined_idiom'] if emb is not None]), axis=0)
    return centroid_lit, centroid_idiom


def export_to_csv(results, file_path):
    with closing(open(file_path, 'w', newline='')) as f:
        fieldnames = ['mwe'] + list(results.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mwe, cosine_sims in results['C_0 vs NC_0'].items():
            row = {'mwe': mwe}
            for key, sim_list in results.items():
                row[key] = sim_list[mwe][0]
            writer.writerow(row)



def get_noun_centroid_df(noun_embeddings):
    noun_embeddings_filtered = {k: v for k, v in noun_embeddings.items() if v}


    centroids = []
    occurrences = []
    # Concatenate subword embeddings for each noun
    for noun, embeds in noun_embeddings_filtered.items():
        noun_combined_embeds = []
        for occurrence in embeds:
            combined_embed = np.concatenate(occurrence, axis=0)
            noun_combined_embeds.append(combined_embed)
        
        # Compute centroids for all occurrences of that noun
        centroid =  np.mean(np.array(noun_combined_embeds), axis=0)
        centroids.append(centroid)
        occurrences.append(len(embeds))

    noun_df = pd.DataFrame({'noun': list(noun_embeddings_filtered.keys()),
                            'centroid': centroids,
                            'nb_occ': occurrences})
    return noun_df

###------------------------------------- Layer processing -------------------------------------###

def create_layer_df(data_df, layer_id, batch_size, normalize, print_ignored, tokenizer= bert_tokenizer, model= bert_model):
    # Concatenate sentences in each row to form a paragraph
    paragraphs = [' '.join(row) for row in zip(data_df['prior'], data_df['sentence'], data_df['after'])]
    all_sentences = [row for row in zip(data_df['prior'], data_df['sentence'], data_df['after'])]
    num_batches = (len(paragraphs) + batch_size - 1) // batch_size
    all_sentences_token_number = get_nb_tokens_per_sent(all_sentences, batch_size, num_batches, tokenizer)
    sentence_embeddings = retrieve_sentence_embeddings(paragraphs, batch_size, num_batches, layer_id, all_sentences_token_number, normalize, tokenizer, model)

    # Extract embeddings from sentence_embeddings list
    prior_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[::3]]
    sentence_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[1::3]]
    after_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings[2::3]]

    data_df['prior_embeddings'] = prior_embeddings_list
    data_df['sentence_embeddings'] = sentence_embeddings_list
    data_df['after_embeddings'] = after_embeddings_list


    data_df = add_mwe_embeddings_to_df(data_df, print_ignored, tokenizer)
    
    return data_df


def get_init_last_avg_sim_df(meaning_type, init_df, last_df):
    res_df = pd.DataFrame(columns=['mwe', f'{meaning_type}_meaning', f'init_{meaning_type}_embedding', f'last_{meaning_type}_embedding', f'{meaning_type}_concat_sim'])

    for i, row in init_df.iterrows():
        init_embedding = row[f'{meaning_type}_embedding']
        # print("init_embedding: ", init_embedding)
        last_embedding = last_df.loc[i, f'{meaning_type}_embedding']
        
        if np.all(pd.notnull(init_embedding)) and np.all(pd.notnull(last_embedding)):
            avg_init_embedding = np.mean(init_embedding, axis=0)
            # print("avg_init_embedding: ", avg_init_embedding)
            avg_last_embedding = np.mean(last_embedding, axis=0)
            concat_sim = 1 - cosine(avg_init_embedding, avg_last_embedding)
            
            new_row = {
                'mwe': row['mwe'],
                f'{meaning_type}_meaning': row[f'{meaning_type}_meaning'],
                f'init_{meaning_type}_embedding': avg_init_embedding,
                f'last_{meaning_type}_embedding': avg_last_embedding,
                f'{meaning_type}_concat_sim': concat_sim
            }
            
            res_df = pd.concat([res_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return res_df



def get_init_last_embedwise_sim_df(meaning_type, init_df, last_df):
    res_df = pd.DataFrame(columns=['mwe', f'{meaning_type}_meaning', f'init_{meaning_type}_embedding_0', f'init_{meaning_type}_embedding_1', f'last_{meaning_type}_embedding_0', f'last_{meaning_type}_embedding_1', f'{meaning_type}_sim_0', f'{meaning_type}_sim_1'])

    for i, row in init_df.iterrows():
        init_embedding = row[f'{meaning_type}_embedding']
        last_embedding = last_df.loc[i, f'{meaning_type}_embedding']
        
        if np.all(pd.notnull(init_embedding)) and np.all(pd.notnull(last_embedding)):
            sim_0 = 1 - cosine(init_embedding[0], last_embedding[0])
            sim_1 = 1 - cosine(init_embedding[1], last_embedding[1])
            
            new_row = {
                'mwe': row['mwe'],
                f'{meaning_type}_meaning': row[f'{meaning_type}_meaning'],
                f'init_{meaning_type}_embedding_0': init_embedding[0],
                f'init_{meaning_type}_embedding_1': init_embedding[1],
                f'last_{meaning_type}_embedding_0': last_embedding[0],
                f'last_{meaning_type}_embedding_1': last_embedding[1],
                f'{meaning_type}_sim_0': sim_0,
                f'{meaning_type}_sim_1': sim_1
            }
            
            res_df = pd.concat([res_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return res_df



def combine_embeddings(emb1, emb2, combination_type):
    if combination_type == "concatenation":
        combined_embeds = np.concatenate((emb1, emb2), axis=0)
    elif combination_type == "average":
        combined_embeds = np.mean([emb1, emb2], axis=0)
    
    return combined_embeds




def get_comp_noncomp_sim_scores(selected_pairs, data_df, combination_type):   
    results = {}
    ignored_pairs_count = 0

    for literal_index, idiomatic_index in selected_pairs:
        # Get the MWE embeddings for each pair
        mwe = data_df.loc[literal_index, 'mwe']
        literal_embedding = data_df.loc[literal_index, 'literal_embedding']
        idiomatic_embedding = data_df.loc[idiomatic_index, 'idiomatic_embedding']
        
        # Check if either embedding is None, if so, skip this pair
        # The reason being a lemmatization issue
        if literal_embedding is None or idiomatic_embedding is None:       
            ignored_pairs_count += 1
            continue
            
        # Compute similarity
        lit_embs = combine_embeddings(np.asarray(literal_embedding[0]), np.asarray(literal_embedding[1]), combination_type)
        idiom_embs = combine_embeddings(np.asarray(idiomatic_embedding[0]), np.asarray(idiomatic_embedding[1]), combination_type)
        cosine_sims = cosine_similarity(lit_embs.reshape(1, -1), idiom_embs.reshape(1, -1))[0] #TODO: unsure about this line

        for cosine_sim in cosine_sims:
            if mwe not in results:
                results[mwe] = []
            results[mwe].append(cosine_sim)
    
    return results



def convert_dict_to_df(results):
    rows = []
    for mwe, cosine_sims in results.items():
        for cosine_sim in cosine_sims:
            rows.append({'mwe': mwe, 'combined_comp vs combined_noncomp': cosine_sim})

    comp_noncom_df = pd.DataFrame(rows)
    return comp_noncom_df





#----------------------------------------- Paraphrases ----------------------------------------------#
def populate_mwe_meanings(df):
    """
    Function to populate a dictionary where the key is the MWE and the value is a list of its possible fine-grained meanings.
    """
    mwe_meanings = {}

    # Iterate over each row of the dataframe
    for index, row in df.iterrows():
        mwe = row['mwe']
        fine_grained = row['fine_grained']

        # If the MWE is not yet in the dictionary, initialize an empty list
        if mwe not in mwe_meanings:
            mwe_meanings[mwe] = []

        # Append the fine-grained meaning if it's not already in the list
        if fine_grained not in mwe_meanings[mwe]:
            mwe_meanings[mwe].append(fine_grained)

    return mwe_meanings


def replace_mwe(row):
    """ 
    function to replace the MWE with its fine-grained meaning
    """
   
    # Split the MWE into two words
    mwe_words = row['mwe'].split()
    
    if len(mwe_words) == 2:
        # Create a regex pattern to match the MWE (case-insensitive)
        pattern = r'\b{}\s+{}\b'.format(mwe_words[0], mwe_words[1])
        
        # Replace the MWE with the fine-grained meaning
        new_sentence = re.sub(pattern, row['fine_grained'], row['sentence'], flags=re.IGNORECASE)
        
        return new_sentence
    else:
        # If MWE is not two words, return the original sentence
        return row['sentence']
    

def replace_mwe_with_alt_meaning(row, mwe_meanings):
    """ 
    Function to replace the MWE with the alternate fine-grained meaning
    """
    mwe = row['mwe']
    
    if mwe in mwe_meanings:
        meanings = mwe_meanings[mwe]
        
        # Identify the alternate meaning (the one that is NOT equal to 'fine_grained')
        alternate_meaning = [meaning for meaning in meanings if meaning != row['fine_grained']][0]
        
        # Create a regex pattern to match the MWE (case-insensitive)
        mwe_words = mwe.split()
        if len(mwe_words) == 2:
            pattern = r'\b{}\s+{}\b'.format(mwe_words[0], mwe_words[1])
            
            # Replace the MWE in the sentence with the alternate fine-grained meaning
            new_sentence = re.sub(pattern, alternate_meaning, row['sentence'], flags=re.IGNORECASE)
            return new_sentence
        else:
            # If MWE is not two words, return the original sentence
            return row['sentence']
    else:
        # If the MWE doesn't have alternate meanings, return the original sentence
        return row['sentence']
    


def encode_df(mwe_df, batch_size, layer_id, tokenizer= bert_tokenizer, model= bert_model):
    all_sentences = [row for row in mwe_df['sentence']]
    num_batches = (len(all_sentences) + batch_size - 1) // batch_size
    all_sentences_batches = [all_sentences[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    all_sentences_tokenized = [[tokenizer.tokenize(sent) for sent in batch] for batch in all_sentences_batches]
    all_sentences_token_number = [[len(tokenized_sentence) for tokenized_sentence in batch] for batch in all_sentences_tokenized]
    embeds = encode_paragraphs(all_sentences_batches, layer_id, tokenizer, model)
    sentence_embeddings = extract_sentence_embeddings(embeds, all_sentences_token_number)
    sentence_embeddings_list = [embedding.detach().cpu().numpy() for embedding in sentence_embeddings]
    assert len(sentence_embeddings_list) == len(mwe_df), "Mismatch between number of embeddings and DataFrame rows"
    mwe_df['sentence_embeddings'] = sentence_embeddings_list

    mwe_df['compound_embeddings'] = None
    for index, row in mwe_df.iterrows():
        mwe_start = None
        tokenized_sent= row['sentence_tokens']
        tokenized_mwe = row["mwe"].split()
        for i in range(len(tokenized_sent) - len(tokenized_mwe) +1):  
            if tokenized_sent[i: i+len(tokenized_mwe)] == tokenized_mwe: 
                mwe_start = i 
                break
                
        if mwe_start is not None:
            mwe_end = mwe_start + len(tokenized_mwe) - 1
            # Extract the embeddings corresponding to the MWE
            mwe_embeddings = row["sentence_embeddings"][mwe_start : mwe_end + 1]
            mwe_df.at[index, 'compound_embeddings'] = mwe_embeddings

    return mwe_df



def add_C_NC_tag(df):
    # Create the new column 'C/NC' with default value 'X'
    df['C/NC'] = 'X'

    # Apply the conditions
    df.loc[df['fine_grained'] == df['literal_meaning'], 'C/NC'] = 'C'
    df.loc[(df['fine_grained'] == df['_1']) | (df['fine_grained'] == df['_2']), 'C/NC'] = 'NC'

    # Reorder columns 
    cols = list(df.columns)
    cols.insert(2, cols.pop(cols.index('C/NC')))
    df = df[cols]
    return df



def process_mwe_scores(mwe_scores_df):
    """
    Process MWE (Multi-Word Expression) scores from a DataFrame.

    Parameters:
    ----------
    mwe_scores_df : pandas.DataFrame
        DataFrame containing MWE scores with a 'mwe' column and 13 'layer_X' columns.

    Returns:
    -------
    dict
        A dictionary with the original 'mwe' values and processed scores for each layer:
        - 0 for equality ('=')
        - -1 for NC (1)
        - 1 for C (0)
        - Other values remain unchanged.

    """
    
    data = {'mwe': mwe_scores_df['mwe'].tolist()}
    for i in range(13):
        data[f'layer_{i}'] = [0 if x == '=' else -1 if x == 1 else 1 if x == 0 else x 
                              for x in mwe_scores_df[f'layer_{i}']]
    return data
