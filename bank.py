import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Plot enhanced relational plots between multiple columns with better styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer Data Analysis: Relational Plots', fontsize=18, y=1.02, fontweight='bold')
    
    # Plot 1: Credit Limit vs Total Transaction Amount
    sns.scatterplot(
        data=df, x='Credit_Limit', y='Total_Trans_Amt', 
        ax=axes[0, 0], color='royalblue', alpha=0.7, edgecolor='w', s=80
    )
    axes[0, 0].set_title('Credit Limit vs Transaction Amount', fontsize=14, pad=10)
    axes[0, 0].set_xlabel('Credit Limit ($)', fontsize=12)
    axes[0, 0].set_ylabel('Total Transaction Amount ($)', fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: Customer Age vs Credit Limit
    sns.scatterplot(
        data=df, x='Customer_Age', y='Credit_Limit', 
        ax=axes[0, 1], color='forestgreen', alpha=0.7, edgecolor='w', s=80
    )
    axes[0, 1].set_title('Customer Age vs Credit Limit', fontsize=14, pad=10)
    axes[0, 1].set_xlabel('Customer Age', fontsize=12)
    axes[0, 1].set_ylabel('Credit Limit ($)', fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    
    # Plot 3: Card Category vs Credit Limit
    sns.boxplot(
        data=df, x='Card_Category', y='Credit_Limit', ax=axes[1, 0], 
        hue='Card_Category', palette='viridis', width=0.6, 
        linewidth=1.5, fliersize=4
    )
    axes[1, 0].set_title('Card Category vs Credit Limit', fontsize=14, pad=10)
    axes[1, 0].set_xlabel('Card Category', fontsize=12)
    axes[1, 0].set_ylabel('Credit Limit ($)', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)
    if axes[1, 0].get_legend() is not None:
        axes[1, 0].get_legend().remove()
    
    # Plot 4: Total Transactions vs Total Amount (Regression)
    sns.regplot(
        data=df, x='Total_Trans_Ct', y='Total_Trans_Amt', 
        ax=axes[1, 1], color='crimson', scatter_kws={'alpha':0.4, 's':60},
        line_kws={'color': 'darkred', 'lw': 2}
    )
    axes[1, 1].set_title('Transaction Count vs Amount (with Regression)', fontsize=14, pad=10)
    axes[1, 1].set_xlabel('Transaction Count', fontsize=12)
    axes[1, 1].set_ylabel('Transaction Amount ($)', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def plot_categorical_plot(df):
    """Plot an enhanced categorical plot for the 'Card_Category' column."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    # Get dynamic palette based on unique card categories
    card_types = df['Card_Category'].unique()
    palette = sns.color_palette("husl", n_colors=len(card_types))
    
    ax = sns.countplot(
        data=df, 
        x='Card_Category', 
        palette=palette,
        hue='Card_Category',
        edgecolor='black',
        linewidth=1,
        saturation=0.9,
        dodge=False,
        order=['Blue', 'Silver', 'Gold', 'Platinum']
    )
    
    # Only remove legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', 
            va='center', 
            xytext=(0, 5), 
            textcoords='offset points',
            fontsize=12
        )
    
    plt.title('Distribution of Card Categories', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Card Category', fontsize=14, labelpad=10)
    plt.ylabel('Number of Customers', fontsize=14, labelpad=10)
    sns.despine(left=True, bottom=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def plot_statistical_plot(df):
    """Plot an enhanced histogram of transaction amount distribution with stylish formatting"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    
    # Create histogram with custom styling (without kde_kws)
    hist = sns.histplot(
        data=df, 
        x='Total_Trans_Amt', 
        color='#4B8BBE',  # Custom blue color for bars
        bins=40,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.85,
        kde=True  # Keep KDE but don't style it here
    )
    
    # Manually style the KDE line after creation
    for line in ax.lines:
        if 'kde' in str(line).lower():
            line.set(color='#306998', linewidth=2.5)
    
    # Calculate statistics
    median = df['Total_Trans_Amt'].median()
    q1 = df['Total_Trans_Amt'].quantile(0.25)
    q3 = df['Total_Trans_Amt'].quantile(0.75)
    
    # Add statistical lines with custom styling
    ax.axvline(median, color='#FFD43B', linestyle='--', linewidth=2.5, 
               label=f'Median: ${median:,.0f}')
    ax.axvline(q1, color='#646464', linestyle=':', linewidth=1.8, 
               label=f'Q1: ${q1:,.0f}')
    ax.axvline(q3, color='#646464', linestyle=':', linewidth=1.8, 
               label=f'Q3: ${q3:,.0f}')
    
    # Customize grid and spines
    ax.grid(True, linestyle='--', alpha=0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Formatting
    plt.title('Transaction Amount Distribution\nWith Key Statistics', 
             fontsize=18, pad=20, fontweight='bold', color='#333333')
    plt.xlabel('Transaction Amount ($)', fontsize=14, labelpad=12, color='#333333')
    plt.ylabel('Frequency', fontsize=14, labelpad=12, color='#333333')
    
    # Customize legend
    legend = plt.legend(loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('#F0F0F0')
    legend.get_frame().set_edgecolor('#CCCCCC')
    
    # Improve x-axis formatting
    ax.xaxis.set_major_formatter('${x:,.0f}')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add subtle background color
    ax.set_facecolor('#F9F9F9')
    fig.set_facecolor('#F9F9F9')
    
    # Add watermark-like signature
    plt.text(0.95, 0.02, 'Data Analysis', 
             transform=ax.transAxes, fontsize=12, 
             color='gray', alpha=0.5, ha='right', va='bottom')
    
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight', facecolor='#F9F9F9')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculate statistical moments for a given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the dataset: clean, handle missing values, and remove outliers."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Handle missing values
    numerical_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                     'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                     'Total_Trans_Amt', 'Total_Trans_Ct']
    
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 
                       'Income_Category', 'Card_Category']
    
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove outliers using IQR for numerical columns
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    for col in ['Credit_Limit', 'Total_Trans_Amt', 'Total_Revolving_Bal']:
        if col in df.columns:
            df = remove_outliers(df, col)
    
    return df


def writing(moments, col):
    """Print statistical moments for a given column."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    skewness = 'not skewed' if abs(moments[2]) < 0.5 else 'right skewed' if moments[2] > 0 else 'left skewed'
    kurtosis = 'platykurtic' if moments[3] < 0 else 'leptokurtic' if moments[3] > 0 else 'mesokurtic'
    print(f'The data was {skewness} and {kurtosis}.')
    return


def perform_clustering(df, col1, col2):
    """Perform clustering on the dataset using KMeans."""
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[[col1, col2]])
    df_scaled = pd.DataFrame(df_scaled, columns=[col1, col2])
    
    def plot_elbow_method():
        inertia = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), inertia, marker='o', linestyle='--', color='blue')
        plt.title('Elbow Method for Optimal K', fontsize=16)
        plt.xlabel('Number of Clusters', fontsize=14)
        plt.ylabel('Inertia', fontsize=14)
        plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_elbow_method()
    
    optimal_k = 5  # Can be adjusted based on elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    centres = kmeans.cluster_centers_
    
    score = silhouette_score(df_scaled, labels)
    print(f'Silhouette Score for k={optimal_k}: {score:.2f}')
    
    # Inverse transform centers for original scale visualization
    centres_original = scaler.inverse_transform(centres)
    return labels, df_scaled, centres_original[:, 0], centres_original[:, 1]


def plot_clustered_data(labels, data, xkmeans, ykmeans, df, col1, col2):
    """Plot the clustered data in original scale."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[col1], df[col2], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Cluster Centers')
    plt.title(f'Customer Segmentation: {col1} vs {col2}', fontsize=16)
    plt.xlabel(col1, fontsize=14)
    plt.ylabel(col2, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression fitting with outlier removal."""
    # Prepare data
    x = df[[col1]].values
    y = df[col2].values
    
    # Remove outliers using IQR
    def remove_outliers(x_data, y_data):
        q1_x, q3_x = np.percentile(x_data, [25, 75])
        iqr_x = q3_x - q1_x
        x_mask = (x_data >= (q1_x - 1.5 * iqr_x)) & (x_data <= (q3_x + 1.5 * iqr_x))
        
        q1_y, q3_y = np.percentile(y_data, [25, 75])
        iqr_y = q3_y - q1_y
        y_mask = (y_data >= (q1_y - 1.5 * iqr_y)) & (y_data <= (q3_y + 1.5 * iqr_y))
        
        return x_data[x_mask & y_mask], y_data[x_mask & y_mask]
    
    x_clean, y_clean = remove_outliers(x.flatten(), y)
    x_clean = x_clean.reshape(-1, 1)
    
    # Fit model
    model = LinearRegression()
    model.fit(x_clean, y_clean)
    y_pred = model.predict(x_clean)
    
    # Print diagnostics
    print(f"\n=== Regression Diagnostics ===")
    print(f"Original data points: {len(x)}")
    print(f"After outlier removal: {len(x_clean)}")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: ${model.intercept_:.2f}")
    
    return x_clean, y_clean, y_pred, model.coef_[0], model.intercept_


def plot_fitted_data(x, y, y_pred, slope, intercept, col1, col2):
    """Plot the fitted data with actual vs predicted."""
    plt.figure(figsize=(12, 7))
    
    # Plot actual data points
    plt.scatter(x, y, c='blue', label='Actual Data', alpha=0.6)
    
    # Plot regression line (sorted for smooth line)
    sorted_idx = x.flatten().argsort()
    plt.plot(x[sorted_idx], y_pred[sorted_idx], 
             c='red', linewidth=2, label=f'Regression (Slope: {slope:.4f})')
    
    # Add regression equation
    plt.annotate(
        f'y = {slope:.2f}x + {intercept:.2f}',
        xy=(0.05, 0.9),
        xycoords='axes fraction',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=12
    )
    
    # Formatting
    plt.title(f'Linear Relationship: {col1} vs {col2}', fontsize=16, pad=20)
    plt.xlabel(f'{col1} ($)', fontsize=14)
    plt.ylabel(f'{col2} ($)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    df = pd.read_csv("BankChurners.csv")
    df = preprocessing(df)
    
    print("First Five rows in a dataset:\n")
    print(df.head())
    print("\nInformation of data:\n")
    print(df.info())
    print("\nSummary statistics of a data:\n")
    print(df.describe(include='all'))
    print("\nCorrelation of numerical data:\n")
    print(df.select_dtypes(include=['number']).corr())
    
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    # Analyze Credit Limit distribution
    moments = statistical_analysis(df, 'Credit_Limit')
    writing(moments, 'Credit_Limit')
    
    # Cluster customers based on spending behavior
    clustering_results = perform_clustering(df, 'Total_Trans_Amt', 'Total_Trans_Ct')
    plot_clustered_data(clustering_results[0], clustering_results[1], 
                       clustering_results[2], clustering_results[3],
                       df, 'Total_Trans_Amt', 'Total_Trans_Ct')
    
    # Analyze relationship between credit limit and transaction amount
    x_clean, y_clean, y_pred, slope, intercept = perform_fitting(df, 'Credit_Limit', 'Total_Trans_Amt')
    plot_fitted_data(x_clean, y_clean, y_pred, slope, intercept, 
                    'Credit Limit', 'Total Transaction Amount')


if __name__ == '__main__':
    main()