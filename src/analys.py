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