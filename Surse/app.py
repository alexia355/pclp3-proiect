import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from baseline_model import incarca_si_preproceseaza

print("[Bonus] Inițializare server Gradio și antrenare model în background...")

# 1. Pregătim modelul pentru interfață
df_train, mediana = incarca_si_preproceseaza('train.csv', is_train=True)
X_train = df_train.drop(columns=['Recomandare_Aparat'])
y_train = df_train['Recomandare_Aparat']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

structura_coloane = X_train.columns.tolist()

# 2. Funcția de predicție
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
    
    probabilitate = model.predict_proba(date_procesate)[0][1]
    predictie = model.predict(date_procesate)[0]
    
    if predictie == 1:
        return f"🔴 RECOMANDARE: APARAT DENTAR INDICAT\n(Probabilitate calculată de model: {probabilitate*100:.2f}%)"
    else:
        return f"🟢 RECOMANDARE: NU ESTE NECESAR APARAT DENTAR\n(Probabilitate de tratament: {probabilitate*100:.2f}%)"

# 3. Interfața grafică
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown("# 🦷 Sistem Inteligent de Diagnostic Ortodontic")
    gr.Markdown("Introduceți caracteristicile clinice ale pacientului pentru a evalua necesitatea unui aparat dentar.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Date Pacient")
            varsta = gr.Slider(minimum=8, maximum=50, value=20, step=1, label="Vârstă (ani)")
            gen = gr.Radio(choices=["Masculin", "Feminin"], value="Feminin", label="Gen")
            inghesuire = gr.Slider(minimum=0.0, maximum=10.0, value=2.5, step=0.1, label="Înghesuire Dinți (mm)")
            tip_muscatura = gr.Dropdown(choices=["Normala", "Overbite", "Underbite"], value="Normala", label="Tipul Mușcăturii")
            dificultate = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Dificultate Masticatie (1-10)")
            durere = gr.Radio(choices=["Da", "Nu"], value="Nu", label="Prezență Durere Mandibulară")
            buton_analizeaza = gr.Button("🔮 Calculează Diagnostic", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🩺 Rezultat Evaluare")
            iesire_text = gr.Textbox(label="Verdict Model Inteligent", interactive=False, lines=4)

    buton_analizeaza.click(
        fn=prezice_necesitate_aparat, 
        inputs=[varsta, gen, inghesuire, tip_muscatura, dificultate, durere], 
        outputs=iesire_text
    )

if __name__ == "__main__":
    demo.launch()