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

def confidence_interval_bp(df):
    #95% konfidensintervall för medelvärdet av systolic_bp (normalapprox).

    x = df['systolic_bp'].dropna().to_numpy()
    mean = x.mean()
    std = x.std(ddof=1)
    n = x.size
    margin = 1.96 * (std / np.sqrt(n))
    return (mean - margin, mean + margin)

def hypothesis_test_smokers_higher_bp(df, n_perm=5000, seed=42):

    np.random.seed(seed)
    
    # Hämta grupperna
    sm = df[df['smoker'] == 'Yes']['systolic_bp'].dropna().to_numpy()
    ns = df[df['smoker'] == 'No']['systolic_bp'].dropna().to_numpy()
    
    # Observerad skillnad smokers och no smokers
    diff_obs = sm.mean() - ns.mean()

    # Kombinera grupperna
    all_values = np.concatenate([sm, ns])
    diff_perm = []

    # Skapa slumpfördelningar
    for _ in range(n_perm):
        np.random.shuffle(all_values)
        sm_fake = all_values[:len(sm)]
        ns_fake = all_values[len(sm):]
        diff_perm.append(sm_fake.mean() - ns_fake.mean())

    diff_perm = np.array(diff_perm)

    # p-värde hur ofta slumpen gav större skillnad än observerat
    p_value = (diff_perm >= diff_obs).mean()

    return diff_obs, p_value

def plot_histograms(df):
    # Ett histogram för nummeriska variabler 

    df[['age','height','weight','systolic_bp','cholesterol']].hist(figsize=(12,8))
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, column):
    # En boxplot för vald kolumn.

    plt.boxplot(x=df[column])
    plt.title(f'Boxplot: {column}')
    plt.show()

def plot_smoker_bar(df):
    # Stapeldiagram över rökare och icke rökare

    df['smoker'].value_counts().plot(kind='bar')
    plt.title("Andel rökare vs icke-rökare")
    plt.show()