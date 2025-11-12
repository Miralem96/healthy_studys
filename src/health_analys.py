import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class HealthAnalyzer:
    """
    En klass som analyserar hälsodata.
    Den innehåller metoder för att köra en enkel analys pipeline
    med linjär regression, residualanalys och korrelationsmatris

    """
    def __init__(self, df: pd.DataFrame):
        """
        Initierar analysobjektet.
        Parametrar df (DataFrame) Hälsodata som ska analyseras

        """
        self.df = df.copy()
        self.model = None
        self.features = None

    def preprocess(self):
        """
        Förbereder datan: binär kodar och tar bort saknade värden

        """
        if 'smoker' in self.df:
            self.df['smoker_bin'] = (self.df['smoker'] == 'Yes').astype(int)
        if 'sex' in self.df:
            self.df['sex_bin'] = (self.df['sex'] == 'M').astype(int)

        cols = [c for c in ['age','height','weight','systolic_bp','cholesterol'] if c in self.df]
        self.df = self.df.dropna(subset=cols).reset_index(drop=True)
        return self.df
    
    def regress_1d(self, feature: str):
        """
        Enkel linjär regression mellan en variabel och blodtryck.
        Visar scatterplot och linjär modell

        """
        X = self.df[[feature]].to_numpy()
        y = self.df['systolic_bp'].to_numpy()

        m = LinearRegression().fit(X, y)
        y_hat = m.predict(X)

        plt.figure(figsize=(6,4))
        plt.scatter(X, y, alpha=0.4, label='Data')
        plt.plot(X, y_hat, label='Linjär modell', linewidth=2)
        plt.title(f'systolic_bp ~ {feature}')
        plt.xlabel(feature); plt.ylabel('systolic_bp')
        plt.grid(alpha=0.2); plt.legend(); plt.tight_layout(); plt.show()

        print(f'koef({feature}) = {m.coef_[0]:.3f}  |  intercept = {m.intercept_:.3f}  |  R² = {m.score(X,y):.3f}')

    
    def regress_multi(self, features):
        """
        Multipel regression för flera variabler
        returnerar koef, intercept och r2

        """
        X = self.df[features].to_numpy()
        y = self.df['systolic_bp'].to_numpy()

        self.model = LinearRegression().fit(X, y)
        self.features = features

        return {
            'features': features,
            'coef': self.model.coef_,
            'intercept': self.model.intercept_,
            'R2': self.model.score(X, y),
        }

    
    def plot_residuals(self):
        """
        Ritar residualdiagram (y - y_hat)
        visar hur bra modellen passar datan

        """
        assert self.model is not None and self.features is not None, 'Kör regress_multi() först.'
        X = self.df[self.features].to_numpy()
        y = self.df['systolic_bp'].to_numpy()
        y_hat = self.model.predict(X)
        resid = y - y_hat

        plt.figure(figsize=(6,4))
        plt.scatter(y_hat, resid, alpha=0.4)
        plt.axhline(0, color='black', linewidth=1)
        plt.title('Residualer vs. förutsagt värde')
        plt.xlabel('Förutsagt blodtryck'); plt.ylabel('Residual')
        plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

    
    def corr_heatmap(self, columns):
        C = self.df[columns].corr()
        plt.figure(figsize=(6,5))
        plt.imshow(C, cmap='Blues', vmin=-1, vmax=1)
        plt.colorbar(fraction=0.05, pad=0.05)
        plt.title('Korrelationsmatris')
        plt.xticks(range(len(columns)), columns, rotation=45, ha='right')
        plt.yticks(range(len(columns)), columns)
        plt.tight_layout(); plt.show()

    
    def run_pipeline(self):
        """
        Kör hela analysflödet steg för steg
        
        """
        self.preprocess()
        self.regress_1d('age')
        summary = self.regress_multi(['age','weight'])
        self.plot_residuals()
        cols = [c for c in ['age','height','weight','systolic_bp','cholesterol'] if c in self.df]
        if len(cols) >= 2:
            self.corr_heatmap(cols)
        return summary