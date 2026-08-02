import os, urllib.request, json, io
from datetime import datetime
import streamlit as st
import numpy as np, tensorflow as tf, cv2
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# ==========================================================================
MODEL_URL = "https://huggingface.co/akert123/densenet-thorax/resolve/main/densenet121_14maladies_v3.keras"
MODEL_PATH = "densenet121_14maladies_v3.keras"
THRESH_URL = "https://huggingface.co/akert123/densenet-thorax/resolve/main/thresholds.json"
THRESH_PATH = "thresholds.json"
IMG_SIZE = 320

st.set_page_config(page_title="Diagnostic Thorax IA - Hopital Cheikh Zaid",
                   page_icon="🫁", layout="wide")

PATHOLOGIES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass',
               'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema',
               'Emphysema','Fibrosis','Pleural_Thickening','Hernia']

FR = {'Atelectasis':'Atelectasie','Cardiomegaly':'Cardiomegalie',
      'Effusion':'Epanchement pleural','Infiltration':'Infiltrat','Mass':'Masse',
      'Nodule':'Nodule','Pneumonia':'Pneumonie','Pneumothorax':'Pneumothorax',
      'Consolidation':'Consolidation','Edema':'Oedeme','Emphysema':'Emphyseme',
      'Fibrosis':'Fibrose','Pleural_Thickening':'Epaississement pleural','Hernia':'Hernie'}

# --------------------------------------------------------------- STYLE
st.markdown("""
<style>
    .stApp {background-color:#f4f7fb;}
    .bandeau {background:linear-gradient(90deg,#0b3d66 0%,#1a6fb0 100%);
        padding:1.2rem 1.6rem; border-radius:12px; color:white; margin-bottom:1.4rem;}
    .bandeau h1 {margin:0; font-size:1.5rem;}
    .bandeau p {margin:0; opacity:0.9; font-size:0.9rem;}
    .carte {background:white; border-radius:10px; padding:0.9rem 1.2rem;
        box-shadow:0 1px 5px rgba(0,0,0,0.08); margin-bottom:0.7rem;}
    .pos {color:#c0392b; font-weight:700; font-size:1.05rem;}
    .kpi {background:white; border-radius:10px; padding:1rem; text-align:center;
        box-shadow:0 1px 5px rgba(0,0,0,0.08);}
    .kpi .v {font-size:1.6rem; font-weight:800; color:#0b3d66;}
    .kpi .l {font-size:0.8rem; color:#7f8c8d;}
    .avert {background:#fff4e5; border-left:4px solid #ff9800; padding:0.8rem 1rem;
        border-radius:6px; font-size:0.85rem; color:#7a4b00; margin-top:1.2rem;}
    .stProgress > div > div > div {background-color:#1a6fb0;}
    section[data-testid="stSidebar"] {background-color:#0b3d66;}
    section[data-testid="stSidebar"] * {color:#eaf2fa;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------- EN-TETE
c1, c2 = st.columns([1, 6])
with c1:
    if os.path.exists("logo_hopital.png"):
        st.image("logo_hopital.png", width=110)
    else:
        st.markdown("<div style='width:110px;height:110px;border:2px dashed #b0bec5;"
                    "border-radius:12px;display:flex;align-items:center;justify-content:center;"
                    "color:#90a4ae;font-size:0.72rem;text-align:center;'>Logo<br>Hopital<br>"
                    "Cheikh Zaid</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='bandeau'><h1>🫁 Aide au diagnostic - Radiographies thoraciques</h1>"
                "<p>Detection automatique de 14 pathologies · DenseNet121 · Hopital Cheikh Zaid</p></div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown("### 🏥 Hopital Cheikh Zaid")
    st.markdown("**Outil d'assistance radiologique**")
    st.markdown("---")
    st.markdown("**Modele :** DenseNet121\n\n**Dataset :** NIH ChestX-ray14\n\n"
                "**AUC moyen :** 0.77\n\n**14 pathologies** (multi-label)")
    st.markdown("---")
    voir_tout = st.checkbox("Afficher les 14 probabilites", value=False)
    st.caption("Projet de stage - IA imagerie medicale")

# --------------------------------------------------------------- MODELE
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Telechargement du modele (une seule fois)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dense_base = m.layers[0]
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = dense_base(inp)
    x = feat
    for layer in m.layers[1:]:
        x = layer(x)
    return m, tf.keras.Model(inp, [feat, x])

@st.cache_resource
def charger_seuils():
    try:
        if not os.path.exists(THRESH_PATH):
            urllib.request.urlretrieve(THRESH_URL, THRESH_PATH)
        with open(THRESH_PATH) as f:
            return json.load(f)
    except Exception:
        return {p: 0.5 for p in PATHOLOGIES}

model, grad_model = charger_modele()
SEUILS = charger_seuils()

# --------------------------------------------------------------- FONCTIONS
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preparer(pil_img):
    g = np.array(pil_img.convert("L"))
    g = _clahe.apply(g.astype("uint8"))
    g = cv2.resize(g, (IMG_SIZE, IMG_SIZE))
    rgb = np.stack([g]*3, axis=-1)
    return np.expand_dims(preprocess_input(rgb.astype("float32")), 0), rgb

def est_radio(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype("float32")
    r, gg, b = arr[...,0], arr[...,1], arr[...,2]
    ecart = (np.abs(r-gg).mean() + np.abs(gg-b).mean())/2
    h, w = arr.shape[:2]
    return ecart < 15 and max(h,w)/min(h,w) < 1.6 and arr.mean(-1).std() > 25

def gradcam(arr, idx):
    arr = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(arr, training=False)
        tape.watch(conv_out)
        loss = pred[:, idx]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(0,1,2))
    heat = tf.reduce_sum(conv_out[0]*w, axis=-1).numpy()
    heat = np.maximum(heat, 0)
    return cv2.resize(heat/(heat.max()+1e-8), (IMG_SIZE, IMG_SIZE))

def rapport_texte(positifs, ordre):
    d = datetime.now().strftime("%d/%m/%Y a %H:%M")
    t  = "="*56 + "\n"
    t += "   HOPITAL CHEIKH ZAID - RAPPORT D'AIDE AU DIAGNOSTIC\n"
    t += "="*56 + "\n\n"
    t += f"Date d'analyse : {d}\n"
    t += "Modele        : DenseNet121 (NIH ChestX-ray14, AUC 0.77)\n\n"
    t += "-"*56 + "\nPATHOLOGIES SIGNALEES\n" + "-"*56 + "\n"
    if positifs:
        for n, p in positifs:
            t += f"  • {FR[n]:<28} {p*100:5.1f}% de confiance\n"
    else:
        t += "  Aucune anomalie detectee au-dessus du seuil.\n"
    t += "\n" + "-"*56 + "\nDETAIL DES 14 PATHOLOGIES\n" + "-"*56 + "\n"
    for n, p in ordre:
        t += f"  {FR[n]:<28} {p*100:5.1f}%\n"
    t += "\n" + "="*56 + "\n"
    t += "AVERTISSEMENT : Outil d'aide au pre-signalement a but\n"
    t += "pedagogique. Ne remplace pas l'avis d'un radiologue.\n"
    t += "="*56 + "\n"
    return t

# --------------------------------------------------------------- INTERFACE
fichier = st.file_uploader("Deposez une radiographie thoracique frontale (PA/AP)",
                           type=['png','jpg','jpeg'])

if fichier is not None:
    pil = Image.open(fichier)
    if not est_radio(pil):
        st.warning("⚠️ Cette image ne ressemble pas a une radiographie thoracique standard. "
                   "Le resultat serait peu fiable.")

    x, rgb = preparer(pil)
    probs = model.predict(x, verbose=0)[0]
    ordre = sorted(zip(PATHOLOGIES, probs), key=lambda t: -t[1])
    positifs = [(n,p) for n,p in ordre if p >= SEUILS.get(n, 0.5)]

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'><div class='v'>{len(positifs)}</div>"
                f"<div class='l'>Pathologie(s) signalee(s)</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='v'>{FR[ordre[0][0]]}</div>"
                f"<div class='l'>Suspicion principale</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='v'>{ordre[0][1]*100:.0f}%</div>"
                f"<div class='l'>Confiance max</div></div>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radiographie")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("Zone d'attention (Grad-CAM)")
        try:
            top = int(np.argmax(probs))
            heat = gradcam(x, top)
            heat_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET),
                                  cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(rgb, 0.6, heat_c, 0.4, 0)
            st.image(overlay, use_container_width=True,
                     caption=f"Zone influencant : {FR[PATHOLOGIES[top]]}")
        except Exception as e:
            st.image(rgb, use_container_width=True)
            st.warning(f"Grad-CAM indisponible : {e}")

    st.subheader("Resultat")
    if positifs:
        st.markdown("**Pathologies signalees :**")
        for n, p in positifs:
            st.markdown(f"<div class='carte'><span class='pos'>{FR[n]}</span> "
                        f"- {p*100:.0f}% de confiance</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie detectee au-dessus du seuil de confiance.")

    st.markdown("**Top 5 probabilites :**")
    for n, p in ordre[:5]:
        st.write(f"{FR[n]} - {p*100:.1f}%")
        st.progress(float(p))

    if voir_tout:
        st.markdown("**Les 14 pathologies :**")
        for n, p in ordre:
            st.write(f"{FR[n]} - {p*100:.1f}%")

    # ---------- RAPPORT + TELECHARGEMENTS ----------
    st.markdown("---")
    st.subheader("📄 Rapport de diagnostic")
    txt = rapport_texte(positifs, ordre)
    st.code(txt, language=None)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Telecharger le rapport (.txt)", txt,
                       file_name=f"rapport_thorax_{datetime.now():%Y%m%d_%H%M}.txt",
                       mime="text/plain", use_container_width=True)

    # PDF simple
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", size=10)
        for ligne in txt.split("\n"):
            pdf.cell(0, 5, ligne.encode('latin-1','replace').decode('latin-1'), ln=1)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        d2.download_button("⬇️ Telecharger le rapport (.pdf)", pdf_bytes,
                           file_name=f"rapport_thorax_{datetime.now():%Y%m%d_%H%M}.pdf",
                           mime="application/pdf", use_container_width=True)
    except Exception:
        d2.info("PDF indisponible (ajoutez 'fpdf2' au requirements.txt)")

st.markdown("<div class='avert'>⚠️ <b>Avertissement medical.</b> Cet outil est une aide au "
            "pre-signalement a but pedagogique. Il n'est pas un dispositif medical valide et ne "
            "remplace en aucun cas l'interpretation d'un radiologue. Aucune decision clinique ne "
            "doit reposer sur ce resultat.</div>", unsafe_allow_html=True)
