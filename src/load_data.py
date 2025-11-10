import pandas as pd

# Laddar ner data från csv filen
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)