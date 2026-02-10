import os 
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from src.utils.dimensionality_reduction import tsne_method, umap_method, pca_method

def plot_distribution(data_val, data_test, threshold, output_dir, score_type="log_likelihood_scores", anomaly_key="", epoch=-1):

    # Create figure with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    if anomaly_key == "":
        # Normal objects (not embeddings)
        title = "Distribution of Log-Likelihood Score normal and augmented Object Data \n" + anomaly_key
    else:
        # Embeddings 
        title = "Embeddings (epoch="+str(epoch)+"): Distribution of Log-Likelihood Score normal and augmented data \n" + anomaly_key
    fig.suptitle(title, fontsize=10)

    # Define consistent color palette
    palette = sns.color_palette("tab10", 4)
    unique_labels = ["normal", "anomaly"]
    unique_data_keys = list(data_val.keys())
    color_mapping = {label: color for label, color in zip(unique_labels, palette)}

    # Function to plot KDE with filled + outline
    def plot_kde(data, ax, title, anomaly_key, threshold=None):
        labels = (["normal"]  * len(data['normal']) + 
                  ["anomaly"] * len(data[anomaly_key]))        
        scores = (data['normal'] + 
                  data[anomaly_key] )

        df = pd.DataFrame({'scores': scores, 'labels': labels})

        # Plot filled KDE
        for label in unique_labels:
            subset = df[df["labels"] == label]
            sns.kdeplot(subset["scores"], fill=True, color=color_mapping[label], alpha=0.3, bw_adjust=0.25, ax=ax)

        # Overlay KDE outlines
        for label in unique_labels:
            subset = df[df["labels"] == label]
            sns.kdeplot(subset["scores"], color=color_mapping[label], linewidth=2, bw_adjust=0.25, ax=ax)

        # Add mean lines
        for label, data_key in zip(unique_labels, unique_data_keys):
            mean_value = np.mean(data[data_key])
            ax.axvline(x=mean_value, color=color_mapping[label], linestyle="--", linewidth=2, label=f"Mean {label} score")
        
        if threshold != None:
            ax.axvline(x=threshold, color='black', linestyle="-", linewidth=2, label=f"Threshold {str(round(threshold,2))}")
        ax.set_title(title)
        ax.set_xlabel("Anomaly Scores")
        ax.set_ylabel("Density")
        ax.legend()

    # Plot for both datasets
    plot_kde(data=data_val,  ax=axes[0], anomaly_key=anomaly_key, title=f"Distribution ({score_type} - {'Val-split'})",  threshold=threshold)
    plot_kde(data=data_test, ax=axes[1], anomaly_key=anomaly_key, title=f"Distribution ({score_type} - {'Test-split'})", threshold=threshold)
  
    # Save figure
    plot_name = 'plot_'+score_type+'_scores_epoch' +str(epoch)+'.png'
    output_dir = os.path.join(output_dir, anomaly_key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, plot_name)
    plt.savefig(filename)
    plt.close()


def visualize_anomaly_score_distribution(res_ad, output_dir, params, epoch=-1, anomaly_key=''):

    if 'GMM' in params:
        scores_val =   {'normal':    res_ad['GMM_log_likelihood_score']['log_likelihood_scores_val_normal'],
                        anomaly_key: res_ad['GMM_log_likelihood_score']['log_likelihood_scores_val_aug'],}
        scores_test =  {'normal':    res_ad['GMM_log_likelihood_score']['log_likelihood_scores_test_normal'],
                        anomaly_key: res_ad['GMM_log_likelihood_score']['log_likelihood_scores_test_aug'],}  
    
        plot_distribution(data_val=scores_val, data_test=scores_test, threshold=res_ad['GMM_log_likelihood_score']['threshold'], 
                          output_dir=output_dir, score_type='GMM_log_likelihood', epoch=epoch, anomaly_key=anomaly_key)


    if 'COPOD' in params:    
        scores_val =   {'normal':    res_ad['COPOD']['anomaly_proba_val_normal'],
                        anomaly_key: res_ad['COPOD']['anomaly_proba_val_anomaly'],}
        scores_test  = {'normal':    res_ad['COPOD']['anomaly_proba_test_normal'],
                        anomaly_key: res_ad['COPOD']['anomaly_proba_test_anomaly'],}  
        plot_distribution(data_val=scores_val, data_test=scores_test, threshold=res_ad['COPOD']['threshold'], 
                          output_dir=output_dir, score_type='COPOD scores', epoch=epoch, anomaly_key=anomaly_key)


def visualize_embeddings(data, output_dir, params, anomaly_key='', epoch=-1):
    # Reduce the embeddings into a 2-Dim space
    output_dir = os.path.join(output_dir, "plots_dim_reduction")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_normal_val     = data['X_val_normal']
    X_augmented_val  = data['X_val_augmented']
    X_normal_test    = data['X_test_normal']
    X_augmented_test = data['X_test_augmented']

    X_val  = np.concatenate((X_normal_val,  X_augmented_val),  axis=0)
    X_test = np.concatenate((X_normal_test, X_augmented_test), axis=0)
    y_val  = np.array([0]*len(X_normal_val)  + [1]*len(X_augmented_val))
    y_test = np.array([0]*len(X_normal_test) + [1]*len(X_augmented_test))
    

    for dim_red_method in params:
        ### Reduction into 2-dim space        
        if dim_red_method == "umap":
            X_val_red  = umap_method(data=X_val)
            X_test_red = umap_method(data=X_test)

        elif dim_red_method == "tsne":
            X_val_red  = tsne_method(data=X_val)
            X_test_red = tsne_method(data=X_test)
            
        elif dim_red_method == "pca":
            X_val_red  = pca_method(data=X_val)
            X_test_red = pca_method(data=X_test)
        else:
            assert False, "Undefined dim_red_method: " + str(dim_red_method)

        ### Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        
        if epoch == -1:
            # Normal objects (not embeddings)
            title = dim_red_method + '-reduced normal and augmented Object Data\n' + anomaly_key
        else:
            # Embeddings 
            title = dim_red_method + '-reduced embeddings (epoch='+str(epoch)+') \n' + anomaly_key
        fig.suptitle(title, fontsize=10)
        color_mapping = {"normal": "#1f77b4", "anomaly": "#d62728"}  # Blue for normal, Red for anomalies

        # Function to create scatter plot
        def plot_scatter(ax, X, y, title):
            # Convert y to NumPy array to ensure indexing
            y = np.array(y)
            
            # Plot normal data
            normal_mask = y == 0 #"normal"
            sns.scatterplot(x=X[normal_mask, 0], y=X[normal_mask, 1], color=color_mapping["normal"], label="Normal",
                            ax=ax, edgecolor="black", alpha=0.6, s=50)

            # Plot anomaly data (larger & more visible)
            anomaly_mask = y == 1 # "anomaly"
            sns.scatterplot(x=X[anomaly_mask, 0], y=X[anomaly_mask, 1], color=color_mapping["anomaly"], label="Anomaly",
                            ax=ax, edgecolor="black", alpha=0.6, s=50, marker="X")

            # Titles and labels
            ax.set_title(title)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend(loc="upper right", fontsize=10)

        # Create scatter plots
        plot_scatter(axes[0], X_val_red, y_val, "Val-Split")
        plot_scatter(axes[1], X_test_red, y_test, "Test-Split")
 

        # Save figure
        plot_name = 'plot_'+dim_red_method+'_embeddings_epoch'+str(epoch)+'.png'
        filename = os.path.join(output_dir, plot_name)
        plt.savefig(filename)
        plt.close()


    