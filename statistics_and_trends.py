import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Plots mileage vs. price as a scatter plot.
    """
    # selects the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    # plots the scatterplot using seaborn
    sns.scatterplot(x=df['Mileage'], y=df['Price'], color='red')
    # formatting the x and y labels
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Price')
    # title of the plot
    ax.set_title('Price vs Mileage')
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Plot the categorical distribution of car prices.
    """
    # sets the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    sns.histplot(df['Price'], bins=50, kde=True)
    # formatting
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    # sets the title of the plot
    ax.set_title('Distribution of Car Prices')
    plt.savefig('statistical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    Plot a categorical plot showing the distribution of car prices
    by fuel type.
    """
    # selects the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    # plots the boxplot with seaborn
    sns.boxplot(x=df['Fuel_Type'], y=df['Price'], hue=df['Fuel_Type'], 
                palette='Set2', legend=False)
    # formatting the x and y label
    ax.set_xlabel('Fuel_Type')
    ax.set_ylabel('Price')
    # Sets the plot title
    ax.set_title('Distribution of Price by Fuel Type')
    plt.savefig('categorical_plot.png')
    plt.show()


def statistical_analysis(df, col: str):
    """
    Compute statistical moments: mean, standard deviation, skewness,
    and excess kurtosis.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the data by checking for missing values,
    describing data, and checking correlation.
    """
    # basic summary statistics
    df.head()
    df.describe()
    print('\n Basic Summary of the data\n', df.head())
    print('\n Data Description:\n', df.describe())
    # Select only numeric columns before calculating correlation
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.corr()
    # Compute correlation only on numeric data
    print('\nCorrelation:\n', numeric_df.corr())
    return df


def writing(moments, col):
    """
    Print statistical moments analysis.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    print('The data was not skewed and platykurtic.')
    return


def main():
    """
    function to process,analyze data set and plots charts
    """
    # import and read data
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Price'
    # calls functions
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
