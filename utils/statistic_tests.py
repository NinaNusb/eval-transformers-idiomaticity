from scipy.stats import entropy, ttest_ind, kstest, shapiro, chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd


def compute_similarity(lit_0, lit_1, idiom_0, idiom_1):    
    # cosine_similarity(u, v) = (u · v) / (||u|| * ||v||)
    cosine_sims = [
        cosine_similarity(lit_0.reshape(1, -1), idiom_0.reshape(1, -1))[0][0],
        cosine_similarity(lit_1.reshape(1, -1), idiom_1.reshape(1, -1))[0][0],
        cosine_similarity(lit_0.reshape(1, -1), lit_1.reshape(1, -1))[0][0],
        cosine_similarity(idiom_0.reshape(1, -1), idiom_1.reshape(1, -1))[0][0]
    ]      
    
    return cosine_sims


def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]


def average_embeddings(embeddings):
    if not isinstance(embeddings, (list, np.ndarray)) or len(embeddings) == 0:
        raise ValueError("Empty or invalid embeddings")
    return np.mean(np.array(embeddings), axis=0)

def get_sentence_and_compound_sim(mwe_df, paraphrase_df):
    sentence_similarities = []
    compound_similarities = []
    for idx, ((mwe_idx, mwe_row), (para_idx, para_row)) in enumerate(zip(mwe_df.iterrows(), paraphrase_df.iterrows()), 1):
        try:
            mwe_sent_emb = average_embeddings(mwe_row['sentence_embeddings'])
            para_sent_emb = average_embeddings(para_row['sentence_embeddings'])
            sent_sim = compute_cosine_similarity(mwe_sent_emb, para_sent_emb)
            sentence_similarities.append(sent_sim)
            
            mwe_comp_emb = average_embeddings(mwe_row['compound_embeddings'])
            para_comp_emb = average_embeddings(para_row['compound_embeddings'])
            comp_sim = compute_cosine_similarity(mwe_comp_emb, para_comp_emb)
            compound_similarities.append(comp_sim)
        except Exception as e:
            print(f"Error at index {idx} (mwe_df index: {mwe_idx}, paraphrase_df index: {para_idx})")
            print(f"Error message: {str(e)}")
            print("MWE row:")
            print(mwe_row)
            print("\nParaphrase row:")
            print(para_row)
            raise  # Re-raise the exception after printing debug info

    return sentence_similarities, compound_similarities


def compute_centroid_cos_sim(centroid_lit, centroid_idiom):  
    return cosine_similarity([centroid_lit], [centroid_idiom])[0][0]


def compute_jensen_shannon_divergence(p, q):
    """ 
    Compute the Jensen-Shannon divergence between two probability distributions.
    Jensen-Shannon divergence is a symmetric version of the Kullback-Leibler divergence, which measures the similarity between two probability distributions.
    It can be used to compare the similarity between two histograms, with a value of 0 indicating the histograms are identical, and higher values indicating more dissimilarity.

    JSD(P || Q) = (1/2) * D(P || M) + (1/2) * D(Q || M) Where:

        P and Q are the two probability distributions (histograms) being compared
        M = (1/2) * (P + Q) is the mixture distribution


    Args:
        p (numpy.ndarray): The first probability distribution.
        q (numpy.ndarray): The second probability distribution.
        
    Returns:
        float: The Jensen-Shannon divergence between the two distributions.
    
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def compute_pair_JSD(results, nb_rows, nb_col, figsize=(12,8)):
    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(nb_rows, nb_col, figsize=figsize)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Extract the histograms from the subplots
    histograms = [np.array([patch.get_height() for patch in axs[i].patches]) for i in range(nb_rows * nb_col)]

    # Normalize the histograms to obtain probability distributions
    histograms = [hist / np.sum(hist) for hist in histograms]

    # Get the keys from the results DataFrame
    keys = results.columns[1:]

    # Compute the JSD for each pair of histograms
    for i in range(len(histograms)):
        for j in range(i+1, len(histograms)):
            jsd_value = compute_jensen_shannon_divergence(histograms[i], histograms[j])
            hist1_key = keys[i]
            hist2_key = keys[j]
            print(f"JSD between  {hist1_key:<25} and  {hist2_key:<30}: {jsd_value:>8.2f}")



def compute_ttest(results):
    # Get the keys from the results DataFrame
    keys = results.columns[1:]

    # Perform the t-test for each pair of keys
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            
            # Get the observed values for the two keys
            observed1 = results[key1].values
            observed2 = results[key2].values
            
            # Compute the t-test statistic and p-value
            t_stat, p_value = ttest_ind(observed1, observed2)
            
            print(f"T-test statistic between {key1} and {key2}: \n{t_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print()


def compute_kstest(results):
    for key in results.columns[1:]:
        # Extract the cosine similarities for the current key
        cosine_scores = results[key].values
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = kstest(cosine_scores, 'norm')
        
        print(f"Results for {key}:")
        print(f"Kolmogorov-Smirnov test statistic: {ks_statistic:.4f}, p-value: {ks_p_value:.4f}")
        print()



def compute_shapiro(results):
    for key in results.columns[1:]:
        # Extract the cosine similarities for the current key
        cosine_scores = results[key].values
                
        # Perform Shapiro-Wilk test
        shapiro_statistic, shapiro_p_value = shapiro(cosine_scores)
        
        print(f"Results for {key}:")
        print(f"Shapiro-Wilk test statistic: {shapiro_statistic}, p-value: {shapiro_p_value}")
        print()



def compute_chi2(results):
    # Get the keys from the results DataFrame
    keys = results.columns[1:]

    # Perform the chi-square test for each pair of keys
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            
            # Get the observed values for the two keys
            observed = np.column_stack((results[key1].values, results[key2].values))
            
            # Compute the chi-squared statistic and p-value
            chi2_stat, p_value, _, _ = chi2_contingency(observed)
            
            print(f"Chi-square statistic between {key1} and {key2}: \n{chi2_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print()




def compute_cohens_d(results):
    # Get the keys from the results DataFrame
    keys = results.columns[1:]

    # Compute Cohen's d for each pair of groups
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            
            # Get the observed values for the two groups
            group1_values = results[key1].values
            group2_values = results[key2].values
            
            # Calculate the mean and standard deviation for each group
            group1_mean = np.mean(group1_values)
            group1_std = np.std(group1_values)
            group2_mean = np.mean(group2_values)
            group2_std = np.std(group2_values)
            
            # Compute the pooled standard deviation
            pooled_std = np.sqrt(((len(group1_values)-1)*(group1_std**2) + (len(group2_values)-1)*(group2_std**2)) / (len(group1_values) + len(group2_values) - 2))
            
            # Compute Cohen's d
            cohen_d = (group1_mean - group2_mean) / pooled_std
            
            print(f"Cohen's d between {key1} and {key2}: {cohen_d:.4f}")




def compute_sim_pair_same_position(data_df, mwe, start, layer_id, group_type, retrieve_encoded_df, combine_embeddings):
    """ 
    group_type: "C" ou "NC"
    """

    mwe_df = data_df[data_df['mwe'] == mwe].copy()
    mwe_df = retrieve_encoded_df(mwe_df, batch_size=32, layer_id=layer_id, normalize=True, print_ignored=False)
    mwe_df["combined_lit"] = mwe_df.apply(lambda row: combine_embeddings(row['literal_embedding'][0], row['literal_embedding'][1], "concatenation") if row['literal_embedding'] is not None else None, axis=1)
    mwe_df["combined_idiom"] = mwe_df.apply(lambda row: combine_embeddings(row['idiomatic_embedding'][0], row['idiomatic_embedding'][1], "concatenation") if row['idiomatic_embedding'] is not None else None, axis=1)

    # Pair mwe with the same position
    if group_type == "C":
        col = "combined_lit"
    elif group_type == "NC":
        col= "combined_idiom"
    mwe_filtered = mwe_df[mwe_df[col].notnull()]
    duplicate_starts = mwe_filtered.groupby('mwe_start')['mwe_start'].transform('count') > 1
    result_df = mwe_filtered[duplicate_starts & mwe_filtered[col].notnull()]

    similarity_data = []
    filtered_df = result_df[result_df['mwe_start'] == start]

    combined_embeddings = filtered_df[col].apply(np.array)
    embeddings = combined_embeddings.tolist()
    row_ids = filtered_df.index.tolist()

    # Get all possible pairs of embeddings and row IDs
    pairs = list(itertools.combinations(zip(embeddings, row_ids), 2))

    # Compute cosine similarity for each pair
    for pair in pairs:
        (emb1, row_id1), (emb2, row_id2) = pair
        cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
        similarity_data.append({'Row1': row_id1, 
                                'Row2': row_id2, 
                                'Nb tokens sent1': len(result_df.at[row_id1, "sentence_tokens"]), 
                                'Nb tokens sent2': len(result_df.at[row_id2, "sentence_tokens"]),
                                'Nb tokens diff': max(len(result_df.at[row_id1, "sentence_tokens"]), len(result_df.at[row_id2, "sentence_tokens"])) - min(len(result_df.at[row_id1, "sentence_tokens"]), len(result_df.at[row_id2, "sentence_tokens"])),
                                'Cosine Similarity': cosine_sim}
                                )

    similarity_df = pd.DataFrame(similarity_data)
    return result_df, similarity_df