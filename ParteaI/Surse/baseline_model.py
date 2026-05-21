import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def incarca_si_preproceseaza(cale_csv, is_train=True, mediana_inghesuire=None):
    df = pd.read_csv(cale_csv)
    
    # 1. Eliminăm coloana 'id' (nu are valoare predictivă)
    df = df.drop(columns=['id'])
    
    # 2. Tratarea valorilor lipsă prin imputare cu mediana de pe Train
    if is_train:
        mediana_inghesuire = df['Inghesuire_Dinti_mm'].median()
    df['Inghesuire_Dinti_mm'] = df['Inghesuire_Dinti_mm'].fillna(mediana_inghesuire)
    
    # 3. One-Hot Encoding complet (identic cu app.py pentru consistența interfeței)
    df = pd.get_dummies(df, columns=['Gen', 'Tip_Muscatura', 'Durere_Mandibula'])
    
    return df, mediana_inghesuire

def main():
    print("[Pasul 3] Încărcare date și preprocesare pentru Machine Learning...")
    
    # Preprocesăm subseturile antrenare/testare
    df_train, mediana = incarca_si_preproceseaza('train.csv', is_train=True)
    df_test, _ = incarca_si_preproceseaza('test.csv', is_train=False, mediana_inghesuire=mediana)
    
    # Separăm X (caracteristici) de y (variabila țintă) ÎNAINTE de aliniere
    X_train = df_train.drop(columns=['Recomandare_Aparat'])
    y_train = df_train['Recomandare_Aparat'].astype(int)
    
    X_test = df_test.drop(columns=['Recomandare_Aparat'])
    y_test = df_test['Recomandare_Aparat'].astype(int)
    
    # Aliniem doar caracteristicile X în caz de categorii lipsă în test
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    print("[Pasul 3] Antrenare model de bază (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicții pe setul de test
    y_pred = model.predict(X_test)
    
    # Calculăm metricile oficiale solicitate în barem
    acuratete = accuracy_score(y_test, y_pred)
    precizie = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*15 + " EVALUARE MODEL PE SUBSETUL DE TEST " + "="*15)
    print(f"Acuratețe (Accuracy): {acuratete:.4f} ({acuratete*100:.2f}%)")
    print(f"Precizie (Precision): {precizie:.4f}")
    print(f"Sensibilitate (Recall): {recall:.4f}")
    print(f"Scor F1 (F1-Score):   {f1:.4f}")
    print("="*68)
    
    # Generăm și salvăm Matricea de Confuzie grafică
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Nu are nevoie', 'Are nevoie'], 
                yticklabels=['Nu are nevoie', 'Are nevoie'])
    plt.title('Matricea de Confuzie - Model Ortodontic')
    plt.ylabel('Realitate (Adevăr)')
    plt.xlabel('Predicție Model')
    plt.tight_layout()
    plt.savefig('matrice_confuzie.png')
    plt.close()
    print("-> Matricea de confuzie a fost salvată ca 'matrice_confuzie.png'!")

if __name__ == "__main__":
    main()