import numpy as np
import pandas as pd

#Generare date cu ajutorul unui seed
def genereaza_date_ortodontie(num_pacienti, seed_val=42):
    np.random.seed(seed_val)
    varsta = np.random.randint(8, 51, size=num_pacienti)
    gen = np.random.choice(['Masculin', 'Feminin'], size=num_pacienti, p=[0.48, 0.52])
    inghesuire = np.random.exponential(scale=2.0, size=num_pacienti).round(1)
    inghesuire = np.clip(inghesuire, 0.0, 10.0)
    tip_muscatura = np.random.choice(['Normala', 'Overbite', 'Underbite'], size=num_pacienti, p=[0.55, 0.30, 0.15])
    dificultate_masticatie = np.random.randint(1, 11, size=num_pacienti)
    durere_mandibula = np.random.choice([True, False], size=num_pacienti, p=[0.25, 0.75])
    
    scor_orto = (inghesuire * 1.2) + (dificultate_masticatie * 0.3)
    scor_orto += np.array([2.5 if m == 'Underbite' else 1.5 if m == 'Overbite' else -1.5 for m in tip_muscatura])
    scor_orto += np.array([1.2 if d else 0.0 for d in durere_mandibula])
    scor_orto += np.array([0.8 if v < 18 else 0.0 for v in varsta])
    
    probabilitate = 1 / (1 + np.exp(-scor_orto))
    recomandare_aparat = (np.random.rand(num_pacienti) < probabilitate).astype(int)
    
    df = pd.DataFrame({
        'Varsta': varsta, 'Gen': gen, 'Inghesuire_Dinti_mm': inghesuire,
        'Tip_Muscatura': tip_muscatura, 'Dificultate_Masticatie': dificultate_masticatie,
        'Durere_Mandibula': durere_mandibula, 'Recomandare_Aparat': recomandare_aparat
    })
    return df

def main():
    print("[Pasul 1] Generare dataset stomatologic (Aparate Dentare)...")
    df_brut = genereaza_date_ortodontie(900)
    for _ in range(15):
        df_brut.iloc[np.random.randint(0, 900), 2] = np.nan
    df_amestecat = df_brut.sample(frac=1, random_state=42).reset_index(drop=True)
    train_set = df_amestecat.iloc[:600].copy()
    test_set = df_amestecat.iloc[600:900].copy()
    train_set.insert(0, 'id', range(1, len(train_set) + 1))
    test_set.insert(0, 'id', range(1, len(test_set) + 1))
    train_set.to_csv('train.csv', index=False)
    test_set.to_csv('test.csv', index=False)
    print("-> Fișierele 'train.csv' și 'test.csv' au fost generate cu succes!")

if __name__ == "__main__":
    main()
