import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as ss

def plot_relational_plot(df):
    """
    This function plots a relational plot for relevant
    columns of the DataFrame.
    """
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        x=df["actual_distance_to_destination"], 
        y=df["actual_time"], 
        hue=df["route_type"], 
        palette="coolwarm", 
        size=df["actual_time"], 
        sizes=(20, 200),  # Adjust point sizes
        alpha=0.7,  # Transparency
        edgecolor="black"
    )
    plt.title("Distance vs. Time by Route Type", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Distance to Destination (km)", fontsize=12)
    plt.ylabel("Actual Time (minutes)", fontsize=12)
    plt.legend(title="Route Type", loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig('relational_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    This function plots a statistical plot for
    the distribution of actual travel time.
    """
    plt.figure(figsize=(12, 7))
    sns.histplot(df["actual_time"], bins=30, kde=True, color="dodgerblue", edgecolor="black", alpha=0.8)
    plt.axvline(df["actual_time"].mean(), color="red", linestyle="dashed", linewidth=2, label="Mean Travel Time")
    plt.title("Distribution of Actual Travel Time", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Travel Time (minutes)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.savefig('statistical_plot.png')
    plt.show()


def plot_boxplot(df):
    """
    This function creates a boxplot to show distribution of actual time by route type.
    """
    plt.figure(figsize=(12, 7))
    sns.boxplot(x="route_type", y="actual_time", data=df, hue='route_type')
    plt.title("Distribution of Actual Time Across Route Types", fontsize=14, fontweight="bold")
    plt.xlabel("Route Type", fontsize=12)
    plt.ylabel("Actual Time (minutes)", fontsize=12)
    plt.savefig('boxplot.png')
    plt.show()


def categorical_plot(data, col):
    """
    This function creates a count plot for a given categorical column.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(
        x=col, 
        data=data, 
        palette="Set1", 
        hue=col,  # Preventing the deprecation warning
        legend=False  # Disables the legend for a cleaner look
    )
    plt.title(f"Distribution of {col}", fontsize=14, fontweight="bold")
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.savefig(f'{col}_count_plot.png')
    plt.show()


def statistical_analysis(df, col):
    """
    This function calculates the mean, standard deviation,
    skewness, and excess kurtosis for a given column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = ss.kurtosis(df[col])

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    This function handles preprocessing steps such as filling missing values.
    Fills numeric columns with the median and object columns with the mode.
    """
    def replace_null(data, cols):
        for col in cols:
            if data[col].dtype in ['int64', 'float64']:  # Numeric columns
                median = data[col].median()
                data[col] = data[col].fillna(median)
            else:  # Non-numeric columns (Object type)
                mode = data[col].mode()[0]
                data[col] = data[col].fillna(mode)

    # List of columns to replace missing values
    cols = [
        'source_name', 'destination_name'  # Update column names as per your dataset
    ]
    replace_null(df, cols)

    return df


def writing(moments, col):
    """
    This function writes the statistical results for a specific column.
    """
    print(f'For the attribute {col}:')
    print(
        f'Mean = {moments[0]:.2f}, '
        f'Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, and '
        f'Excess Kurtosis = {moments[3]:.2f}.'
    )

    skewness = 'right skewed' if moments[2] > 0 else 'left skewed'

    if moments[3] > 0:
        kurtosis_type = 'leptokurtic'
    elif moments[3] < 0:
        kurtosis_type = 'platykurtic'
    else:
        kurtosis_type = 'mesokurtic'

    print(f'The data is {skewness} and {kurtosis_type}.')
    return


def get_correlation_matrix(df):
    """
    This function calculates the correlation matrix for
    numerical columns of the DataFrame and returns it.
    """
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    if numerical_df.shape[1] > 1:
        corr_matrix = numerical_df.corr()
        print("Correlation Matrix of Numerical Columns:\n")
        print(corr_matrix)
    else:
        print("Not enough numerical columns for correlation.")


def main():
    """
    Main function that orchestrates all tasks: reading data, preprocessing,
    plotting, statistical analysis, and calculating correlation matrix.
    """
    # Load the data
    df = pd.read_csv('data.csv')

    print("First few rows of the data:\n")
    print(df.head())

    # Preprocess the data
    df = preprocessing(df)

    print("\nData Description:\n")
    print(df.describe())

    # Generate and save the plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_boxplot(df)

    # Example categorical plot
    categorical_column = 'route_type'  # Adjust this column based on your dataset
    categorical_plot(df, categorical_column)

    # Statistical analysis for 'actual_time' (you can change the column name)
    col = 'actual_time'
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # Correlation matrix
    get_correlation_matrix(df)


if __name__ == '__main__':
    main()
