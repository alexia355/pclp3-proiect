import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ruleaza_eda_subset(cale_csv, nume_subset):
    print(f"\n" + "="*20 + f" ANALIZĂ EDA ORTODONTIE: {nume_subset} " + "="*20)
    df = pd.read_csv(cale_csv)
    
    # Ne asigurăm că directorul dedicat pentru grafice există
    director_iesire = "Grafice_EDA"
    os.makedirs(director_iesire, exist_ok=True)
    
    # a) Analiza valorilor lipsă (Cerința 6.a)
    print("\n[a] Gestiunea Valorilor Lipsă:")
    nr_lipsa = df.isna().sum()
    procent_lipsa = (df.isna().sum() / len(df)) * 100
    tabel_lipsa = pd.DataFrame({'Total Lipsă (NaN)': nr_lipsa, 'Procent (%)': procent_lipsa})
    print(tabel_lipsa[tabel_lipsa['Total Lipsă (NaN)'] > 0])
    
    # b) Statistici descriptive (Cerința 6.b)
    print("\n[b] Statistici Descriptive:")
    print(df.describe(include='all'))
    
    coloane_numerice = ['Varsta', 'Inghesuire_Dinti_mm', 'Dificultate_Masticatie']
    coloane_categorice = ['Gen', 'Tip_Muscatura', 'Durere_Mandibula']
    
    # c) Analiza distribuției (Cerința 6.c)
    for col in coloane_numerice:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, color='purple', edgecolor='black')
        plt.title(f'Distributia variabilei: {col} ({nume_subset})')
        plt.xlabel(col)
        plt.ylabel('Numar Pacienti')
        plt.tight_layout()
        plt.savefig(os.path.join(director_iesire, f"{nume_subset}_distributie_{col}.png"))
        plt.close()
        
    for col in coloane_categorice:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue=col, palette='Pastel1', edgecolor='black', legend=False)
        plt.title(f'Distributia categoriilor: {col} ({nume_subset})')
        plt.xlabel(col)
        plt.ylabel('Numar Pacienti')
        plt.tight_layout()
        plt.savefig(os.path.join(director_iesire, f"{nume_subset}_countplot_{col}.png"))
        plt.close()

    # d) Detectarea outlierilor prin regula IQR (Cerința 6.d)
    print("\n[d] Analiza Outlierilor (Regula IQR):")
    for col in coloane_numerice:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limita_inf = Q1 - 1.5 * IQR
        limita_sup = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < limita_inf) | (df[col] > limita_sup)]
        print(f" - Coloana '{col}': s-au detectat {len(outliers)} valori atipice.")
        
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col], color='lightblue')
        plt.title(f'Boxplot pentru {col} ({nume_subset})')
        plt.tight_layout()
        plt.savefig(os.path.join(director_iesire, f"{nume_subset}_boxplot_{col}.png"))
        plt.close()

    # e) Heatmap Corelații (Cerința 6.e)
    plt.figure(figsize=(7, 5))
    df_num = df[coloane_numerice + ['Recomandare_Aparat']]
    sns.heatmap(df_num.corr(), annot=True, fmt='.2f', cmap='mako', center=0, linewidths=0.5)
    plt.title(f'Heatmap-ul Corelatiilor ({nume_subset})')
    plt.tight_layout()
    plt.savefig(os.path.join(director_iesire, f"{nume_subset}_heatmap_corelatii.png"))
    plt.close()

    # f) Analiza relațiilor cu variabila țintă - Violin Plots (Cerința 6.f)
    for col in coloane_numerice:
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=df, x='Recomandare_Aparat', y=col, hue='Recomandare_Aparat', palette='pastel', legend=False)
        plt.title(f'Relatia dintre {col} si Recomandarea de Aparat ({nume_subset})')
        plt.xlabel('Recomandare Aparat (0=Nu, 1=Da)')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(director_iesire, f"{nume_subset}_relatie_tinta_{col}.png"))
        plt.close()

def main():
    ruleaza_eda_subset('train.csv', 'Train')
    ruleaza_eda_subset('test.csv', 'Test')
    print("\n[Pasul 2] Analiza exploratorie completa! Graficele au fost salvate curat in folderul 'Grafice_EDA'.")

if __name__ == "__main__":
    main()