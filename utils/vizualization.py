import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
import seaborn as sns
from sklearn.metrics import accuracy_score, v_measure_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from math import atan2

from data_processing import *




def create_result_df(results, weighted):
    results_df = pd.DataFrame.from_dict({
                'Pair': results.keys(),
                'Example': ['big/big', 'fish/fish', 'big_1/fish_1', 'big_2/fish_2'],
                'Cosine similarity': [np.average([np.mean(results[key][mwe]) for mwe in results[key].keys()], weights=[len(results[key][mwe]) for mwe in results[key].keys()]) 
                                      if weighted else np.average([np.mean(results[key][mwe]) for mwe in results[key].keys()]) for key in results.keys()]
                }, orient='columns')
    return results_df
    


def get_all_cosine_scores(results): 
    return [score for row in results.iloc[:, 1:].values for score in row] 


def plot_subplot_pair_hist(ax, data, nb_bins, distributional_curve, std_dev, normalized, add_gridlines, title):
    # Plot the histogram
    ax.hist(data, bins=nb_bins, edgecolor='black', density=normalized) # Set density=True for normalized histograms

    mu, std = np.mean(data), np.std(data)

    if distributional_curve:
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)

    if std_dev:
        # 1 std
        ax.axvline(mu - std, color='r', linestyle='--', label='Mean - 1 std dev')
        ax.axvline(mu + std, color='r', linestyle='--', label='Mean + 1 std dev')

        # 2 std
        ax.axvline(mu - 2*std, color='g', linestyle='--', label='Mean - 2 std dev')
        ax.axvline(mu + 2*std, color='g', linestyle='--', label='Mean + 2 std dev')

        ax.legend()

    if add_gridlines:
        ax.grid(True, linestyle='--')

    ax.set_title(title)
    ax.set_xlabel(f'Cosine "similarity" score')
    ax.set_ylabel('Count')



def plot_pair_histograms(results, nb_rows, nb_col, nb_bins, std_dev, distributional_curve, normalized, figsize=(12,8), add_gridlines=False):
    fig, axs = plt.subplots(nb_rows, nb_col, figsize=figsize)
    axs = axs.flatten() # Flatten the axs array for easier indexing

    # Initialize variables to store min and max values for x and y axes
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

    for i, col in enumerate(results.columns[1:]): # (except the 'mwe' column)
        cosine_scores = results[col].values
        plot_subplot_pair_hist(axs[i], cosine_scores, nb_bins, std_dev, distributional_curve, normalized, add_gridlines, title=f'Histogram of Cosine Scores: {results.columns[i+1]}')

        # Update min and max values for x and y axes
        min_x = min(min_x, np.min(cosine_scores))
        max_x = max(max_x, np.max(cosine_scores))
        min_y = min(min_y, 0)  # Assuming frequency cannot be negative
        max_y = max(max_y, axs[i].get_ylim()[1])

    # Set the same limits for x and y axes for all subplots
    for ax in axs:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    plt.show()


    
def create_aggregated_plot(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over the columns (except the 'mwe' column)
    for col in results.columns[1:]:
        # Extract the cosine similarities for the current column
        cosine_scores = results[col].values
        
        # Plot the normalized histogram
        ax.hist(cosine_scores, bins=20, edgecolor='black', alpha=0.5, label=f'{col}', density=True)

    # Calculate overall mean and standard deviation
    all_cosine_scores = get_all_cosine_scores(results)
    overall_mean = np.mean(all_cosine_scores)
    overall_std_dev = np.std(all_cosine_scores)

    # Add vertical lines for one standard deviation
    std_dev_line1 = overall_mean - overall_std_dev
    std_dev_line2 = overall_mean + overall_std_dev
    ax.axvline(std_dev_line1, color='r', linestyle='--', label=f'Mean - 1 std dev ({std_dev_line1:.2f})')
    ax.axvline(std_dev_line2, color='r', linestyle='--', label=f'Mean + 1 std dev ({std_dev_line2:.2f})')

    # Add vertical lines for two standard deviations
    std_dev_line3 = overall_mean - 2*overall_std_dev
    std_dev_line4 = overall_mean + 2*overall_std_dev
    ax.axvline(std_dev_line3, color='g', linestyle='--', label=f'Mean - 2 std dev ({std_dev_line3:.2f})')
    ax.axvline(std_dev_line4, color='g', linestyle='--', label=f'Mean + 2 std dev ({std_dev_line4:.2f})')

    ax.set_title('Aggregated Histogram of Cosine Scores')
    ax.set_xlabel('Cosine similarity score')
    ax.set_ylabel('Normalized Frequency')
    ax.legend()

    plt.show()



def plot_pair_bars(results):
    all_cosine_scores = np.concatenate([results[col].values for col in results.columns[1:]])

    # Calculate the mean and standard deviation for each group
    unique_keys = results.columns[1:]
    means = [np.mean(results[key].values) for key in unique_keys]
    stds = [np.std(results[key].values) for key in unique_keys]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(unique_keys))
    plt.bar(x, means, yerr=stds, capsize=10, color=['#CCCCCC', '#AAAAAA', '#888888', '#666666'])
    plt.xticks(x, unique_keys, ha='right')
    plt.title('Bar Plot of Cosine Scores')
    plt.xlabel('Compared vectors')
    plt.ylabel(f'Cosine "similarity" score')

    plt.axhline(y=np.mean(all_cosine_scores), color='lightgrey', linestyle='--', label='Average')
    plt.legend()
    plt.show()




def plot_whisker(results, add_gridlines=True):
    # Extract the cosine similarity values for each column (except the 'mwe' column)
    whisker_data = [results[col].values for col in results.columns[1:]]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(whisker_data)
    ax.set_xticklabels(results.columns[1:], ha='right')
    ax.set_ylabel('Cosine Similarities')

    if add_gridlines:
        ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()



#-------------------------------------------- Noun embeddings --------------------------------------------#

def print_k_nearest_neigh(target_mwes, k, mwe_df, noun_df):
    for mwe in target_mwes:
        # Extract the embedding of the target MWE
        embedding = mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].values[0]
        embedding = embedding.reshape(1, -1)  # Reshape to match sklearn's input format

        # Compute cosine similarity between the target MWE and all noun embeddings
        noun_embeddings = np.stack(noun_df['avg_embedding'].to_numpy())
        cosine_similarities = cosine_similarity(embedding, noun_embeddings)

        # Find indices of k nearest neighbors
        nearest_indices = cosine_similarities.argsort()[0][-k:][::-1]

        print(f"Top {k} nearest neighbors to '{mwe}':")
        print("Noun\t| Similarity score")
        print("-" * 40)
        for index in nearest_indices:
            print(f"{noun_df.iloc[index]['noun']}\t| {cosine_similarities[0][index]:.3f}")
        print()


def plot_k_nearest_neigh(target_mwes, k, mwe_df, noun_df, figsize=(8, 6), print_nouns=True, print_zoom=False, only_neigh=False, print_distances=False):
    for mwe in target_mwes:
        # Extract the embedding of the target MWE
        target_embedding = mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].values[0]
        target_embedding = target_embedding.reshape(1, -1)

        # Compute cosine similarity between the target MWE and all noun embeddings
        noun_embeddings = np.stack(noun_df['avg_embedding'].to_numpy())
        cosine_similarities = cosine_similarity(target_embedding, noun_embeddings)

        # Find indices of k nearest neighbors
        nearest_indices = cosine_similarities.argsort()[0][-k:][::-1]
        
        if print_zoom:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)
        else: 
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the target MWE and its k nearest neighbors
        if not only_neigh:
            ax.scatter(noun_embeddings[:, 0], noun_embeddings[:, 1], alpha=0.5)
        else:
            pass 
        
        ax.scatter(noun_embeddings[nearest_indices, 0], noun_embeddings[nearest_indices, 1], c='r', marker='x')
        ax.scatter(target_embedding[0, 0], target_embedding[0, 1], c='orange', marker='o')
        ax.set_title(f"Top {k} nearest neighbors to '{mwe}'")
        if only_neigh:
            ax.legend(['Nearest Neighbors', 'Target MWE'])
        else:
            ax.legend(['Nouns', 'Nearest Neighbors', 'Target MWE'])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        if print_zoom:
            # Plot the zoomed view
            ax2.scatter(noun_embeddings[:, 0], noun_embeddings[:, 1], alpha=0.5)
            ax2.scatter(noun_embeddings[nearest_indices, 0], noun_embeddings[nearest_indices, 1], c='r', marker='x')
            ax2.scatter(target_embedding[0, 0], target_embedding[0, 1], c='orange', marker='o')
            ax2.set_title(f"Zoomed view of '{mwe}' and its nearest neighbors")

            # Determine the zoom area
            x_min = min(noun_embeddings[nearest_indices, 0]) - 0.1
            x_max = max(noun_embeddings[nearest_indices, 0]) + 0.1
            y_min = min(noun_embeddings[nearest_indices, 1]) - 0.1
            y_max = max(noun_embeddings[nearest_indices, 1]) + 0.1
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)            

        if print_nouns:
            ax = ax2 if print_zoom else ax
            # Add labels for the nouns
            for j, index in enumerate(nearest_indices):
                ax.text(noun_embeddings[index, 0], noun_embeddings[index, 1], noun_df.iloc[index]['noun'], fontsize=10)


        if print_distances:
            for j, index in enumerate(nearest_indices):
                # Calculate the cosine similarity between the target MWE and each nearest neighbor
                cosine_score = cosine_similarities[0, index]
                
                # Calculate the mid-point coordinates of the line
                mid_x = (target_embedding[0, 0] + noun_embeddings[index, 0]) / 2
                mid_y = (target_embedding[0, 1] + noun_embeddings[index, 1]) / 2
                
                # Add lines to symbolize the distance
                ax.plot([target_embedding[0, 0], noun_embeddings[index, 0]],
                        [target_embedding[0, 1], noun_embeddings[index, 1]],
                        c='b', linestyle='--', linewidth=0.5)
                
                # Add labels for the distances on the lines
                ax.text(mid_x, mid_y, f"{cosine_score:.2f}", fontsize=8, ha='center', va='bottom')

        plt.show()


def plot_eucli_k_nearest_neigh(target_mwes, k, mwe_df, noun_df, figsize=(8, 6), print_nouns=True, only_neigh=False, print_distances=False):
    for mwe in target_mwes:
        # Extract the embedding of the target MWE
        target_embedding = mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].values[0]
        target_embedding = target_embedding.reshape(1, -1)

        # Compute Euclidean distances between the target MWE and all noun embeddings
        noun_embeddings = np.stack(noun_df['avg_embedding'].to_numpy())
        euclidean_dist = euclidean_distances(target_embedding, noun_embeddings)

        # Find indices of k nearest neighbors
        nearest_indices = euclidean_dist.argsort()[0][:k]

        # Compute t-SNE projection of the embeddings
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(noun_embeddings)

        # Plot the target MWE and its k nearest neighbors
        fig, ax = plt.subplots(figsize=figsize)
        if not only_neigh:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        else: 
            pass
        ax.scatter(embeddings_2d[nearest_indices, 0], embeddings_2d[nearest_indices, 1], c='r', marker='x')
        ax.scatter(embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 0],
                   embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 1],
                   c='orange', marker='o')
        ax.set_title(f"Top {k} nearest neighbors to '{mwe}'")
        if only_neigh:
            ax.legend(['Nearest Neighbors', 'Target MWE'])
        else:
            ax.legend(['Nouns', 'Nearest Neighbors', 'Target MWE'])
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')

        if print_nouns:
            # Add labels for the nouns
            for j, index in enumerate(nearest_indices):
                ax.text(embeddings_2d[index, 0], embeddings_2d[index, 1], noun_df.iloc[index]['noun'], fontsize=10)

        if print_distances:
            for j, index in enumerate(nearest_indices):
                # Calculate the Euclidean distance between the target MWE and each nearest neighbor
                euclidean_score = euclidean_dist[0, index]

                # Calculate the mid-point coordinates of the line
                mid_x = (embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 0] +
                         embeddings_2d[index, 0]) / 2
                mid_y = (embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 1] +
                         embeddings_2d[index, 1]) / 2

                # Add lines to symbolize the distance
                ax.plot([embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 0],
                         embeddings_2d[index, 0]],
                        [embeddings_2d[mwe_df.loc[mwe_df['mwe'] == mwe, 'avg_embedding'].index[0], 1],
                         embeddings_2d[index, 1]],
                        c='b', linestyle='--', linewidth=0.5)

                # Add labels for the distances on the lines
                ax.text(mid_x, mid_y, f"{euclidean_score:.2f}", fontsize=8, ha='center', va='bottom')

        plt.show()



#-------------------------------------------- Layers --------------------------------------------#

def plot_init_last_histogram(res_df, meaning_type):   
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 6))
    res_df[f'{meaning_type}_concat_sim'].hist(bins=20, ax=ax, edgecolor='black', color='lightblue', density=True)

    ax.set_title(f'First and last layer {meaning_type.capitalize()} Averaged Embedding Similarity')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Normalized Frequency')

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(f'../results/plots/{meaning_type}_avg_sim_histogram.png', dpi=300)
    plt.show()



def plot_init_last_comparison(literal_res_df, idiomatic_res_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))

    literal_res_df['literal_concat_sim'].hist(bins=20, ax=ax, edgecolor='black', color='lightblue', alpha=0.5, label='Literal', density=True)
    idiomatic_res_df['idiomatic_concat_sim'].hist(bins=20, ax=ax, edgecolor='black', color='orange', alpha=0.5, label='Idiomatic', density=True)

    ax.set_title('Averaged Embedding Similarity')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Normalized Frequency')
    ax.legend()

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    plt.savefig('../results/plots/avg_sim_histogram.png', dpi=300)
    plt.show()



def plot_init_last_sim_contrast(literal_res_df, idiomatic_res_df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(0, literal_res_df['literal_sim_0'].mean(), color='lightblue', alpha=0.5, label='literal_sim_0')
    ax.bar(1, literal_res_df['literal_sim_1'].mean(), color='lightblue', alpha=0.5, label='literal_sim_1')
    ax.bar(2, idiomatic_res_df['idiomatic_sim_0'].mean(), color='orange', alpha=0.5, label='idiomatic_sim_0')
    ax.bar(3, idiomatic_res_df['idiomatic_sim_1'].mean(), color='orange', alpha=0.5, label='idiomatic_sim_1')

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['literal_sim_0', 'literal_sim_1', 'idiomatic_sim_0', 'idiomatic_sim_1'])

    ax.set_title('Contrast between Literal and Idiomatic Similarities')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Mean Similarity')
    ax.legend()

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    plt.savefig('../results/plots/sim_contrast.png', dpi=300)
    plt.show()



def plot_init_last_whisker(literal_res_df, idiomatic_res_df, add_gridlines=True):
    # Extract the cosine similarity values for each column (except the 'mwe' and 'meaning' columns)
    whisker_data = [
        literal_res_df['literal_sim_0'],
        literal_res_df['literal_sim_1'],
        idiomatic_res_df['idiomatic_sim_0'],
        idiomatic_res_df['idiomatic_sim_1']
    ]

    # Generate whisker plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(whisker_data, labels=['literal_sim_0', 'literal_sim_1', 'idiomatic_sim_0', 'idiomatic_sim_1'], showmeans=True)
    ax.set_title('Contrast between Literal and Idiomatic Similarities')
    ax.set_xlabel('Similarity Metric')
    ax.set_ylabel('Cosine Similarity')

    if add_gridlines:
        ax.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()



def plot_layer_pair_histograms(pair_name, nb_rows, nb_col, nb_bins, std_dev, distributional_curve, normalized, figsize=(15, 10), add_gridlines=False, plot_title=None):
    fig, axs = plt.subplots(nb_rows, nb_col, figsize=figsize)
    axs = axs.flatten() # Flatten the axs array for easier indexing

    # Initialize variables to store min and max values for x and y axes
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
   
    for i, layer_id in enumerate(range(1, 13)):
        results = pd.read_csv(f'../results/similarities/{layer_id}_pair_similarities.csv')
        cosine_scores = results[pair_name].values
        
        plot_subplot_pair_hist(axs[i], cosine_scores, nb_bins, distributional_curve, std_dev, add_gridlines, normalized, title= f'Layer {layer_id}')

        # Update min and max values for x and y axes
        min_x = min(min_x, np.min(cosine_scores))
        max_x = max(max_x, np.max(cosine_scores))
        min_y = min(min_y, 0)  # Assuming frequency cannot be negative
        max_y = max(max_y, axs[i].get_ylim()[1])

    # Set the same limits for x and y axes for all subplots
    for ax in axs:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    
    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    plt.suptitle(plot_title)
    plt.savefig(f'../results/plots/{plot_title}.png')
    plt.show()

    

#-------------------------------------------- Lexical units --------------------------------------------#
def plot_mwe_2d(mwe_df, x, y, title, title_size=20, label_size=16, dot_size=70, xlabel="1st eigenvector", ylabel="2nd eigenvector", legend_slot=None): # dimension
    sns.set_style("darkgrid")
    
    # Differentiate between compositional and non-compositional occurrences
    comp_mask = ~mwe_df['literal_embedding'].isna()
    noncomp_mask = ~mwe_df['idiomatic_embedding'].isna()
    
    plt.scatter(x[comp_mask], y[comp_mask],
                c='blue', label='Compositional', s=dot_size)
    plt.scatter(x[noncomp_mask], y[noncomp_mask],
                c='orange', label='Non-compositional', s=dot_size)
    
    plt.title(title, fontsize=title_size, y=1.03)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    
    if legend_slot is None:
        plt.legend()
    else:
        plt.subplot(4, 4, legend_slot )
        plt.axis('off')
        plt.legend()
        plt.subplot(4, 4, legend_slot +1)



def vizualize_mwe(mwe, data_df, layer_id, batch_size, normalize, print_ignored, combination_type, plot_title, tokenizer=bert_tokenizer, model=bert_model, title_size=20, label_size=16, dot_size=70, legend_slot=None):
    mwe_df = data_df[data_df['mwe'] == mwe].copy()    
    mwe_df = retrieve_encoded_df(mwe_df, batch_size, layer_id, normalize, print_ignored, tokenizer, model)
    mwe_df["combined_embed"] = mwe_df.apply(lambda row: 
                                       combine_embeddings(row['literal_embedding'][0], row['literal_embedding'][1], combination_type) if row['literal_embedding'] is not None else
                                       combine_embeddings(row['idiomatic_embedding'][0], row['idiomatic_embedding'][1], combination_type) if row['idiomatic_embedding'] is not None else
                                       None, axis=1)
    
    mwe_df = mwe_df.dropna(subset=['combined_embed'])
    X = np.array(mwe_df['combined_embed'].tolist())
    pca = PCA(n_components=2)
    points = pca.fit_transform(X)
    plot_mwe_2d(mwe_df, x=points[:, 0], y=points[:, 1], title=plot_title, title_size=title_size, label_size=label_size, dot_size=dot_size, legend_slot=legend_slot)




def vizualize_one_mwe_embedding(mwe, data_df, layer_id, batch_size, normalize, print_ignored, embed_id, plot_title,tokenizer=bert_tokenizer, model=bert_model, title_size=20, label_size=16, dot_size=70, legend_slot=None):
    mwe_df = data_df[data_df['mwe'] == mwe].copy()    
    mwe_df = retrieve_encoded_df(mwe_df, batch_size, layer_id, normalize, print_ignored, tokenizer, model)
    mwe_df['target_embed'] = mwe_df.apply(lambda row: row['literal_embedding'][embed_id] if row['literal_embedding'] is not None else row['idiomatic_embedding'][embed_id], axis=1)

    X = np.array(mwe_df['target_embed'].tolist())
    pca = PCA(n_components=2)
    points = pca.fit_transform(X)
    plot_mwe_2d(mwe_df, x=points[:, 0], y=points[:, 1], title=plot_title, title_size=title_size, label_size=label_size, dot_size=dot_size, legend_slot=legend_slot)




def vizualize_both_mwe_embedding(mwe, data_df, layer_id, batch_size, normalize, print_ignored, plot_title,tokenizer=bert_tokenizer, model=bert_model, title_size=10, label_size=16, dot_size=70, legend_slot=None):
    mwe_df = data_df[data_df['mwe'] == mwe].copy()
    mwe_df = retrieve_encoded_df(mwe_df, batch_size, layer_id, normalize, print_ignored, tokenizer, model)

    for embed_id in [0, 1]:
        mwe_df['target_embed'] = mwe_df.apply(lambda row: row['literal_embedding'][embed_id] if row['literal_embedding'] is not None else row['idiomatic_embedding'][embed_id], axis=1)

        X = np.array(mwe_df['target_embed'].tolist())
        pca = PCA(n_components=2)
        points = pca.fit_transform(X)
        plot_mwe_2d(mwe_df, x=points[:, 0], y=points[:, 1], title=plot_title, title_size=title_size, label_size=label_size, dot_size=dot_size, legend_slot=legend_slot)



def plot_mwe_similarity(mwe_name, target_noun, layer_res):
    """
        Function to plot the similarity evolution for a given MWE
    """
    layers = list(range(13))
    literal_similarities = []
    idiomatic_similarities = []

    for layer_id in layers:
        layer_dict = layer_res[layer_id]
        if mwe_name in layer_dict:
            mwe_similarities = layer_dict[mwe_name][target_noun]
            literal_similarities.append(list(mwe_similarities)[0])
            idiomatic_similarities.append(list(mwe_similarities)[1])
        else:
            literal_similarities.append(None)
            idiomatic_similarities.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, literal_similarities, label='Literal Similarity', marker='o')
    ax.plot(layers, idiomatic_similarities, label='Idiomatic Similarity', marker='o')
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Similarity Score')
    ax.set_title(f'Similarity Evolution for "{mwe_name}" and "{target_noun}"')
    ax.legend()
    plt.show()


def plot_mwe_similarity(mwe_name, target_nouns, layer_res):
    """
    Function to plot the similarity evolution for a given MWE and target nouns
    """
    num_nouns = len(target_nouns)
    num_rows = math.ceil(num_nouns / 4)  # Calculate the number of rows needed
    num_cols = 4  # Set the number of columns to 4

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))

    for i, target_noun in enumerate(target_nouns):
        row = i // num_cols  # Calculate the row index
        col = i % num_cols  # Calculate the column index
        ax = axes[row, col]  # Get the corresponding subplot

        layers = list(range(13))
        literal_similarities = []
        idiomatic_similarities = []

        for layer_id in layers:
            layer_dict = layer_res[layer_id]
            if mwe_name in layer_dict and target_noun in layer_dict[mwe_name]:
                mwe_similarities = layer_dict[mwe_name][target_noun]
                literal_similarities.append(list(mwe_similarities)[0])
                idiomatic_similarities.append(list(mwe_similarities)[1])
            else:
                literal_similarities.append(None)
                idiomatic_similarities.append(None)

        ax.plot(layers, literal_similarities, label='Literal Similarity', marker='o')
        ax.plot(layers, idiomatic_similarities, label='Idiomatic Similarity', marker='o')
        ax.set_xlabel('Layer ID')
        ax.set_ylabel('Similarity Score')
        ax.set_title(f'"{mwe_name}" and "{target_noun}"')
        ax.legend()

    fig.suptitle(f'Similarity Evolution for "{mwe_name}"', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"../results/plots/Similarity Evolution for '{mwe_name}' with random nouns.png")
    plt.show()


def plot_decision_boundary_all_layers(mwe, data_df, tokenizer=bert_tokenizer, model=bert_model):
    n_layers = model.config.num_hidden_layers + 1  # including the embedding layer
    nrows = 4
    ncols = math.ceil(n_layers / nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()

    layer_accuracies = []  # To store accuracy per layer for this MWE

    for layer_id, ax in zip(range(n_layers), axs):
        print(f"[{mwe}] Layer {layer_id}")

        data_df = retrieve_encoded_df(data_df, batch_size=32, layer_id=layer_id, normalize=True, print_ignored=False, tokenizer=tokenizer, model=model)
        
        mwe_df = data_df[data_df['mwe'] == mwe].copy()
        filtered_df = mwe_df[mwe_df['literal_embedding'].notna()]
        X_C = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['literal_embedding']])
        y_C = [1] * len(X_C)

        filtered_df = mwe_df[mwe_df['idiomatic_embedding'].notna()]
        X_NC = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['idiomatic_embedding']])
        y_NC = [0] * len(X_NC)

        if len(X_C) == 0 or len(X_NC) == 0:
            print(f"[Warning] No embeddings found for MWE '{mwe}' (either compositional or non-compositional).")
            layer_accuracies.append(None)
            continue
        
        X = np.vstack((X_C, X_NC))
        y = np.hstack((y_C, y_NC))

        if X.shape[1] == 0:
            print(f"[Warning] Embeddings found for '{mwe}', but they are empty.")
            layer_accuracies.append(None)
            continue

        pca = PCA(n_components=2)
        points = pca.fit_transform(X)

        perceptron = Perceptron()
        perceptron.fit(points, y)
        y_pred = perceptron.predict(points)
        accuracy = accuracy_score(y, y_pred)
        acc_percent = round(accuracy * 100)
        layer_accuracies.append(acc_percent)

        print(f"Training accuracy: {acc_percent}% => {'linearly separable' if acc_percent == 100 else 'not linearly separable'}")

        ax.scatter(points[:len(X_C), 0], points[:len(X_C), 1], color='blue', label='Compositional')
        ax.scatter(points[len(X_C):, 0], points[len(X_C):, 1], color='orange', label='Non-compositional')

        if acc_percent == 100:
            x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
            y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['orange', 'blue']))

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(f"Layer {layer_id}")
        ax.legend()

    for i in range(len(axs)):
        if i >= n_layers:
            axs[i].set_visible(False)

    fig.suptitle(f"2D Embedding Visualization for '{mwe}'")
    plt.tight_layout()
    
    os.makedirs("../results/plots", exist_ok=True)
    filename = f"../results/plots/{model.config.name_or_path}-embedding-vis-{mwe}.pdf"
    plt.savefig(filename)
    plt.show()

    return layer_accuracies

    

def generate_latex_table(mwe_accuracies: dict):
    # Automatically determine number of layers from longest list
    max_layers = max(len(v) for v in mwe_accuracies.values())
    n_layers = max_layers - 1  # exclude embedding layer 0

    latex = []

    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\setlength{\tabcolsep}{2pt}")
    latex.append(fr"\begin{{tabular}}{{l*{{{n_layers}}}{{c}}}}")
    latex.append(r"\toprule")
    latex.append(f"    & \multicolumn{{{n_layers}}}{{c}}{{Couche}} \\")
    latex.append(fr"    \cline{{2-{n_layers + 1}}}")
    latex.append("EP & " + " & ".join(str(i) for i in range(n_layers)) + r" \\")
    latex.append(r"\midrule")

    perc_100 = [0] * n_layers
    perc_90 = [0] * n_layers
    total = 0

    for mwe, accs in mwe_accuracies.items():
        if len(accs) < max_layers:
            accs += [None] * (max_layers - len(accs))
        accs = accs[1:]  # skip embedding layer
        total += 1

        # Highlighting
        reached_100_first_layers = any(a == 100 for a in accs[:3])
        never_reached_100 = all((a is None or a < 100) for a in accs)

        mwe_str = mwe.replace("_", " ")
        if reached_100_first_layers:
            mwe_str = f"\\textbf{{{mwe_str}}}"
        elif never_reached_100:
            mwe_str = f"\\textcolor{{lightgray}}{{{mwe_str}}}"

        formatted_vals = []
        for i, val in enumerate(accs):
            if val is None:
                formatted_vals.append("--")
                continue
            if val == 100:
                perc_100[i] += 1
            if val >= 90:
                perc_90[i] += 1
            val_str = f"\\textbf{{{val}}}" if val == 100 else str(val)
            formatted_vals.append(val_str)

        row = mwe_str + " & " + " & ".join(formatted_vals) + r" \\"
        latex.append(row)

    # Footer
    latex.append(r"\midrule")
    perc_100_str = " & ".join(str(round(100 * x / total)) for x in perc_100)
    perc_90_str = " & ".join(str(round(100 * x / total)) for x in perc_90)
    latex.append(r"\multicolumn{1}{r}{\% de 100\%} & " + perc_100_str + r" \\")
    latex.append(r"\multicolumn{1}{r}{\% de $\geq$90\%} & " + perc_90_str + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Séparabilité linéaire (\%) au travers des couches du modèle.}")
    latex.append(r"\label{tab:detailed_linear_separability}")
    latex.append(r"\end{table}")

    return "\n".join(latex)




def analyze_linear_separability(mwe_list, data_df, tokenizer=bert_tokenizer, model=bert_model):
    results = {}

    for mwe in mwe_list:
        results[mwe] = []
        
        for layer_id in range(13):
            data_df = retrieve_encoded_df(data_df, tokenizer, batch_size=32, layer_id=layer_id, normalize=True, print_ignored=False, model=model)
            mwe_df = data_df[data_df['mwe'] == mwe].copy()
            
            # Retrieve compositional embeddings (X_C)
            filtered_df = mwe_df[mwe_df['literal_embedding'].notna()]
            X_C = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['literal_embedding']])
            y_C = [1] * len(X_C)  # compositional == 1

            # Retrieve non-compositional embeddings (X_NC)
            filtered_df = mwe_df[mwe_df['idiomatic_embedding'].notna()]
            X_NC = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['idiomatic_embedding']])
            y_NC = [0] * len(X_NC)  # non-compositional == 0

            X = np.vstack((X_C, X_NC))
            y = np.hstack((y_C, y_NC))

            pca = PCA(n_components=2)
            points = pca.fit_transform(X)

            # Train a sklearn perceptron on the transformed data
            perceptron = Perceptron()
            perceptron.fit(points, y)

            # Calculate accuracy on the transformed data
            y_pred = perceptron.predict(points)
            accuracy = accuracy_score(y, y_pred)

            results[mwe].append(accuracy * 100)

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results, index=[f"Layer {i}" for i in range(13)])
    
    return result_df.T 



def plot_pca_all_layers(mwe, data_df):
    plt.figure(figsize=(15, 10))
    
    # Create a buffer to store the x and y limits
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf

    for layer_id in range(0, 13):
        plt.subplot(4, 4, layer_id+1)
        x_lim, y_lim = vizualize_mwe(mwe, data_df, layer_id=layer_id, batch_size=32, normalize=True, print_ignored=False, combination_type="concatenation", plot_title=f"Layer {layer_id}", title_size=10, label_size=8, dot_size=35)
        
        # Update the buffer with the current limits
        x_min = min(x_min, x_lim[0])
        x_max = max(x_max, x_lim[1])
        y_min = min(y_min, y_lim[0])
        y_max = max(y_max, y_lim[1])

    # Set the same x and y limits for all subplots
    for ax in plt.gcf().get_axes():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Add the legend to the 16th subplot
    plt.subplot(4, 4, 16)
    plt.axis('off')
    handles, labels = plt.gcf().get_axes()[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc='center', frameon=False)

    plt.suptitle(f"Combined '{mwe}' visualization with PCA", fontsize=20, y=0.98)
    plt.savefig(f"../results/plots/Combined '{mwe}' visualization with PCA.png")
    plt.tight_layout()
    plt.show()



def visualize_sentence_attentions(data_df, row_index, tokenizer=bert_tokenizer):
    # Get the sentence and its tokens
    sentence = data_df.loc[row_index, 'sentence']
    tokens = data_df.loc[row_index, 'sentence_tokens']
    
    # Get attention matrices for all heads
    attention_matrices = [data_df.loc[row_index, f'attention_head_{i}'] for i in range(1, 13)]  # Changed to range from 1 to 13
    
    # Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    
    if data_df.loc[row_index, 'literal_embedding'] is not None:
        meaning = 'C'
    elif data_df.loc[row_index, 'idiomatic_embedding'] is not None:
        meaning ='NC'

    fig.suptitle(f"Attention Heads for MWE '{data_df.loc[row_index,'mwe']}', sentence {row_index}, {meaning} meaning", fontsize=16)

    for i, (ax, attention_matrix) in enumerate(zip(axes.flatten(), attention_matrices), 1):  # Start enumeration from 1
        # Create a DataFrame for the attention matrix
        df_attention = pd.DataFrame(attention_matrix, index=tokens, columns=tokens)
        
        # Plot heatmap
        sns.heatmap(df_attention, ax=ax, cmap="crest", cbar=False)
        
        # Set title and labels
        ax.set_title(f"Attention Head {i}")  
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    return fig


def visualize_all_sentences(mwe_df, output_dir, tokenizer=bert_tokenizer):
    for index, row in mwe_df.iterrows():
        if row['literal_embedding'] is not None:
            print(f"MWE '{row['mwe']}', sentence {index}, C meaning")
        elif row['idiomatic_embedding'] is not None:
            print(f"MWE '{row['mwe']}', sentence {index}, NC meaning")

        fig = visualize_sentence_attentions(mwe_df, index, tokenizer)
        
        # Create a filename for the plot
        mwe = row['mwe'].replace(' ', '_')
        filename = f"{mwe}_sentence_{index}.png"
        
        # Save the figure
        fig.savefig(f"{output_dir}/{filename}")
        plt.close(fig)  # Close the figure to free up memory
        
        
#------------------------------------ Paraphrases ---------------------------------------#

def create_similarity_heatmap(df, similarity_type):
    # Extract relevant columns
    sim_columns = [col for col in df.columns if col.startswith(f'{similarity_type}_similarity_')]
    plot_data = df[['sentence_number'] + sim_columns].set_index('sentence_number')

  
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                xticklabels=range(13), yticklabels=plot_data.index)

   
    ax.set_title(f'{similarity_type.capitalize()} Similarity Across BERT Layers\n'
                 f'MWE: {", ".join(df["mwe"].unique())}', 
                 fontsize=16, wrap=True)
    ax.set_xlabel('BERT Layer', fontsize=12)
    ax.set_ylabel('Sentence Number', fontsize=12)
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Similarity Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{similarity_type}_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()



def v_measure_cluster_analysis(mwe, data_df):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))
    axs = axs.flatten()

    v_measure_scores = []
    all_points = []
    distances = []

    for layer_id, ax in zip(range(13), axs):
        data_df = retrieve_encoded_df(data_df, batch_size=32, layer_id=layer_id, normalize=True, print_ignored=False)
        mwe_df = data_df[data_df['mwe'] == mwe].copy()

        filtered_df = mwe_df[mwe_df['literal_embedding'].notna()]
        X_C = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['literal_embedding']])

        filtered_df = mwe_df[mwe_df['idiomatic_embedding'].notna()]
        X_NC = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['idiomatic_embedding']])

        X = np.vstack((X_C, X_NC))
        all_points.append(X)

        pca = PCA(n_components=2)
        points = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = kmeans.fit_predict(points)

        true_labels = np.concatenate((np.zeros(len(X_C)), np.ones(len(X_NC))))
        v_measure_avg = v_measure_score(true_labels, labels)
        v_measure_scores.append(v_measure_avg)

        ax.scatter(points[:len(X_C), 0], points[:len(X_C), 1], color='blue', label='Compositional')
        ax.scatter(points[len(X_C):, 0], points[len(X_C):, 1], color='orange', label='Non-compositional')

        # Draw a line between the cluster centers
        center_0, center_1 = kmeans.cluster_centers_
        ax.plot([center_0[0], center_1[0]], [center_0[1], center_1[1]], 'k-')

        # Calculate and display the distance between the centers
        distance = euclidean(center_0, center_1)
        distances.append(distance)

        # Determine the midpoint of the line
        midpoint_x = (center_0[0] + center_1[0]) / 2
        midpoint_y = (center_0[1] + center_1[1]) / 2

        # Calculate the angle of the line
        angle = atan2(center_1[1] - center_0[1], center_1[0] - center_0[0])
        
        # Adjust text position above the line
        offset = 0.5  # You can adjust this offset value
        text_x = midpoint_x - offset * np.sin(angle)
        text_y = midpoint_y + offset * np.cos(angle)

        # Display the distance text above the line
        ax.text(text_x, text_y, f"{distance:.2f}", fontsize=15, ha='center', va='bottom')

        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
        ax.set_xlabel('1st eigenvector')
        ax.set_ylabel('2nd eigenvector')
        ax.set_title(f"Layer {layer_id}")
        ax.legend()

    # Get the overall minimum and maximum values for x and y
    all_points = np.vstack(all_points)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])

    # Add a buffer to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    buffer = 4.5  # Adjust this buffer value as needed
    x_min -= buffer * x_range
    x_max += buffer * x_range
    y_min -= buffer * y_range
    y_max += buffer * y_range

    # Set the same x and y limits for all subplots
    for ax in axs[:13]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide the frames of subplots 14, 15, and 16
    for ax in axs[13:]:
        ax.set_visible(False)

    fig.suptitle(f"V-measure Clustering of '{mwe}' Embeddings Across Layers")
    plt.tight_layout()
    plt.savefig(f"../results/plots/V-measure Clustering of '{mwe}' Embeddings Across Layers.png")
    plt.show()

    return v_measure_scores, distances




def silhouette_score_cluster_analysis(mwe, data_df):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))
    axs = axs.flatten()

    silhouette_scores = []

    for layer_id, ax in zip(range(13), axs):
        data_df = retrieve_encoded_df(data_df, batch_size=32, layer_id=layer_id, normalize=True, print_ignored=False)
        mwe_df = data_df[data_df['mwe'] == mwe].copy()

        filtered_df = mwe_df[mwe_df['literal_embedding'].notna()]
        X_C = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['literal_embedding']])

        filtered_df = mwe_df[mwe_df['idiomatic_embedding'].notna()]
        X_NC = np.array([combine_embeddings(emb1, emb2, "concatenation") for emb1, emb2 in filtered_df['idiomatic_embedding']])

        X = np.vstack((X_C, X_NC))

        pca = PCA(n_components=2)
        points = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = kmeans.fit_predict(points)

        silhouette_avg = silhouette_score(points, labels)
        silhouette_scores.append(silhouette_avg)

        ax.scatter(points[:len(X_C), 0], points[:len(X_C), 1], color='blue', label='Compositional')
        ax.scatter(points[len(X_C):, 0], points[len(X_C):, 1], color='orange', label='Non-compositional')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
        ax.set_xlabel('1st eigenvector')
        ax.set_ylabel('2nd eigenvector')
        ax.set_title(f"Layer {layer_id}")
        ax.legend()

    # Hide the frames of subplots 14, 15, and 16
    for ax in axs[13:]:
        ax.set_visible(False)

    fig.suptitle(f"K-Means (silhouette) Clustering of '{mwe}' Embeddings Across Layers")
    plt.tight_layout()
    plt.savefig(f"../results/plots/K-Means (silhouette) Clustering of '{mwe}' Embeddings Across Layers.png")
    plt.show()

    return silhouette_scores
