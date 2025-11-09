import numpy as np
import matplotlib.pyplot as plt

def basic_statistics(df):
    # Returnerar medelvärde, median och min och max för numeriska kolumner
    
    stats = {
        'mean': df.mean(numeric_only=True),
        'median': df.median(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True)
    }
    return stats

def simulate_disease_probability(df, n=1000, seed=42):
    # Simulerar andelen sjukdomar baserat på verklig sannolikhet

    np.random.seed(seed)
    p = df['disease'].mean()
    simulations = np.random.binomial(1, p, n)
    return simulations.mean(), p

def plot_histograms(df):
    # Ett histogram för nummeriska variabler 

    df[['age','height','weight','systolic_bp','cholesterol']].hist(figsize=(12,8))
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, column):
    # En boxplot för vald kolumn.

    plt.boxplot(x=df[column])
    plt.title(f"Boxplot: {column}")
    plt.show()

def plot_smoker_bar(df):
    # Stapeldiagram över rökare och icke rökare

    df['smoker'].value_counts().plot(kind='bar')
    plt.title("Andel rökare vs icke-rökare")
    plt.show()