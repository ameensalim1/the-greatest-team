import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def genresImportantAttributes(data, important_features):
    data = data.assign(genre=data['genre'].str.split(', ')).explode('genre').reset_index(drop=True)

    # Plotting important features for each genre
    genres = data['genre'].unique()
    num_genres = len(genres)
    num_cols = 3  # Number of columns for subplot grid
    num_rows = (num_genres + num_cols - 1) // num_cols  # Calculate rows based on number of genres

    # Set up subplots
    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    for i, genre in enumerate(genres):
        genre_data = data[data['genre'] == genre]
        genre_data[important_features].mean().plot(kind='bar', ax=axes[i], title=f'{genre} - Important Features')
        axes[i].set_ylabel('Average Value')
        axes[i].set_xlabel('Features')

    # Hide any empty subplots (if genres don't fill the grid)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def plotFeatureDistributions(data, features):
    # Load data
    data = data.assign(genre=data['genre'].str.split(', ')).explode('genre').reset_index(drop=True)
    
    # Process each genre and each feature
    genres = data['genre'].unique()
    num_rows = len(genres)
    num_cols = len(features)
    
    # Set up subplots in a grid layout for all genres and features
    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
    fig.suptitle('Outlier Detection for Each Genre and Feature', fontsize=16, y=1.02)
    
    for row, genre in enumerate(genres):
        genre_data = data[data['genre'] == genre]
        
        for col, feature in enumerate(features):
            ax = axes[row, col]  # Select the correct subplot
            
            # Calculate mean and standard deviation
            feature_values = genre_data[feature].dropna()
            mean = feature_values.mean()
            std_dev = feature_values.std()
            
            # Define outliers as values more than 2 standard deviations from the mean
            outliers = genre_data[(genre_data[feature] < mean - 2 * std_dev) |
                                  (genre_data[feature] > mean + 2 * std_dev)]
            
            # Plot the feature values with outliers highlighted
            ax.scatter(genre_data.index, genre_data[feature], label=feature, color='blue', alpha=0.7)
            ax.scatter(outliers.index, outliers[feature], label='Outliers', color='red', edgecolor='black')
            ax.axhline(mean, color='green', linestyle='--', label='Mean')
            ax.axhline(mean + 2 * std_dev, color='orange', linestyle='--', label='+2 Std Dev')
            ax.axhline(mean - 2 * std_dev, color='orange', linestyle='--', label='-2 Std Dev')
            
            # Set subplot titles and labels
            ax.set_title(f'{genre} - {feature}')
            ax.set_ylabel(feature if col == 0 else "")  # Label y-axis only on the first column
            ax.set_xlabel("Index" if row == num_rows - 1 else "")  # Label x-axis only on the last row
            if row == 0 and col == num_cols - 1:  # Add legend only once
                ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust layout to fit title
    plt.show()