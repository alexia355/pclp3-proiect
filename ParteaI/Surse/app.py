import gradio as gr
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from baseline_model import incarca_si_preproceseaza

print("[Bonus] Inițializare server Gradio conform specificațiilor din barem...")

# Pregătire model pentru interfață
df_train, mediana = incarca_si_preproceseaza('train.csv', is_train=True)
X_train = df_train.drop(columns=['Recomandare_Aparat'])
y_train = df_train['Recomandare_Aparat']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

structura_coloane = X_train.columns.tolist()

# Funcția de predicție avansată (returnează verdictul și dicționarul de probabilități)
def prezice_necesitate_aparat(varsta, gen, inghesuire, tip_muscatura, dificultate, durere):
    date_pacient = pd.DataFrame([{
        'Varsta': int(varsta),
        'Gen': gen,
        'Inghesuire_Dinti_mm': float(inghesuire),
        'Tip_Muscatura': tip_muscatura,
        'Dificultate_Masticatie': int(dificultate),
        'Durere_Mandibula': True if durere == "Da" else False
    }])
    
    date_procesate = pd.get_dummies(date_pacient, columns=['Gen', 'Tip_Muscatura', 'Durere_Mandibula'])
    
    for col in structura_coloane:
        if col not in date_procesate.columns:
            date_procesate[col] = 0
            
    date_procesate = date_procesate[structura_coloane]
    
    # Calculare probabilitățile pentru ambele clase
    probabilitati = model.predict_proba(date_procesate)[0]
    prob_nu = float(probabilitati[0])
    prob_da = float(probabilitati[1])
    
    predictie = model.predict(date_procesate)[0]
    
    if predictie == 1:
        verdict = f"RECOMANDARE: APARAT DENTAR INDICAT\nModelul are o certitudine de {prob_da*100:.2f}% în acest diagnostic."
    else:
        verdict = f"RECOMANDARE: TRATAMENT ORTODONTIC NEINDICAT\nPacientul nu necesită aparat dentar în acest moment."
        
    # Returnare verdict text și un dicționar pentru componenta de tip Label din Gradio
    return verdict, {"Nu are nevoie": prob_nu, "Are nevoie de aparat": prob_da}

# Calea către matricea de confuzie salvată
cale_matrice = "matrice_confuzie.png" if os.path.exists("matrice_confuzie.png") else None

# Construirea interfeței grafice vizuale
with gr.Blocks() as demo:
    gr.Markdown("# 🦷 Sistem Inteligent de Diagnostic Ortodontic (Proiect PCLP3)")
    gr.Markdown("Aplicație clinică interactivă bazată pe modelul de Regresie Logistică (Acuratețe: 92.00%).")
    
    with gr.Tab("Diagnostic în Direct"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Introducere Date Clinic")
                varsta = gr.Slider(minimum=8, maximum=50, value=20, step=1, label="Vârstă (ani)")
                gen = gr.Radio(choices=["Masculin", "Feminin"], value="Feminin", label="Gen")
                inghesuire = gr.Slider(minimum=0.0, maximum=10.0, value=2.5, step=0.1, label="Înghesuire Dinți (mm)")
                tip_muscatura = gr.Dropdown(choices=["Normala", "Overbite", "Underbite"], value="Normala", label="Tipul Mușcăturii")
                dificultate = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Dificultate Masticatie (1-10)")
                durere = gr.Radio(choices=["Da", "Nu"], value="Nu", label="Prezență Durere Mandibulară")
                
                buton_analizeaza = gr.Button("Calculează Diagnostic", variant="primary")
                
            with gr.Column():
                gr.Markdown("Rezultat Evaluare")
                iesire_text = gr.Textbox(label="Verdict Final", interactive=False, lines=2)
                iesire_procente = gr.Label(label="Probabilitățile Fiecărei Clase")

    with gr.Tab("Performanță Model (Vizualizări)"):
        gr.Markdown("Matricea de Confuzie obținută pe setul de Testare")
        if cale_matrice:
            gr.Image(value=cale_matrice, label="Matrice de Confuzie", interactive=False)
        else:
            gr.Markdown("Fișierul 'matrice_confuzie.png' nu a fost găsit. Rulează mai întâi scriptul baseline_model.py!")

    buton_analizeaza.click(
        fn=prezice_necesitate_aparat, 
        inputs=[varsta, gen, inghesuire, tip_muscatura, dificultate, durere], 
        outputs=[iesire_text, iesire_procente]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="purple"))