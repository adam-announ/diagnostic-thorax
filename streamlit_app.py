import os, urllib.request, json
from datetime import datetime
import streamlit as st
import numpy as np, tensorflow as tf, cv2
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

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

# ===================== CHARTE CHEIKH ZAID (rouge corail) =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    .stApp {background-color:#ffffff; font-family:'Poppins',sans-serif;}
    #MainMenu, footer {visibility:hidden;}

    .nav {display:flex; align-items:center; justify-content:space-between;
        padding:1rem 0.5rem; border-bottom:1px solid #f0e6e3; margin-bottom:2rem;}
    .nav .marque {font-weight:800; color:#e8503a; font-size:1.15rem; letter-spacing:0.5px;}
    .nav .marque span {color:#1a1a2e; font-weight:700;}
    .badge {display:inline-block; background:#fdeeeb; color:#e8503a; font-weight:600;
        padding:0.4rem 1rem; border-radius:30px; font-size:0.85rem; margin-bottom:1.2rem;}
    .hero-titre {font-size:3rem; font-weight:800; line-height:1.1; color:#1a1a2e; margin:0;}
    .hero-titre .rouge {color:#e8503a; border-bottom:4px solid #f4a698;}
    .hero-sous {color:#6b7280; font-size:1.05rem; margin:1.2rem 0 0 0; max-width:520px;}

    .kpi {background:#fff; border:1px solid #f0e6e3; border-radius:16px; padding:1.2rem;
        text-align:center; box-shadow:0 4px 20px rgba(232,80,58,0.06);}
    .kpi .v {font-size:2rem; font-weight:800; color:#e8503a;}
    .kpi .l {font-size:0.82rem; color:#6b7280; font-weight:500;}

    .carte {background:#fff; border:1px solid #f0e6e3; border-left:5px solid #e8503a;
        border-radius:12px; padding:1rem 1.3rem; margin-bottom:0.7rem;
        box-shadow:0 2px 12px rgba(232,80,58,0.05);}
    .carte .nom {color:#1a1a2e; font-weight:700; font-size:1.1rem;}
    .carte .pct {color:#e8503a; font-weight:800;}

    .stButton>button, .stDownloadButton>button {
        background:#e8503a !important; color:white !important; border:none !important;
        border-radius:30px !important; padding:0.6rem 1.8rem !important;
        font-weight:600 !important; font-family:'Poppins',sans-serif !important;}
    .stButton>button:hover, .stDownloadButton>button:hover {background:#d13f2a !important;}

    .stProgress > div > div > div {background:linear-gradient(90deg,#e8503a,#f4a698) !important;}
    [data-testid="stFileUploader"] {background:#fdf6f4; border:2px dashed #f4a698;
        border-radius:16px; padding:1rem;}

    .avert {background:#fdf6f4; border-left:4px solid #e8503a; padding:1rem 1.3rem;
        border-radius:10px; font-size:0.85rem; color:#8a3a2a; margin-top:2rem;}
    .section-titre {font-size:1.6rem; font-weight:800; color:#1a1a2e; margin:1.5rem 0 0.8rem 0;}
    .section-titre::before {content:''; display:inline-block; width:6px; height:24px;
        background:#e8503a; border-radius:3px; margin-right:12px; vertical-align:middle;}
</style>
""", unsafe_allow_html=True)

# ===================== NAV =====================
n1, n2 = st.columns([1, 3])
with n1:
    if os.path.exists("logo_hopital.png"):
        st.image("logo_hopital.png", width=200)
    else:
        st.markdown("<div class='nav'><div class='marque'>◢ CHEIKH ZAID "
                    "<span>· IA Radiologie</span></div></div>", unsafe_allow_html=True)

# ===================== HERO =====================
st.markdown("<div class='badge'>● Hopital Universitaire International Cheikh Zaid</div>",
            unsafe_allow_html=True)
st.markdown("<h1 class='hero-titre'>Diagnostic <span class='rouge'>Radiologique</span><br>"
            "assiste par IA</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sous'>Detection automatique de 14 pathologies thoraciques a partir "
            "d'une radiographie. Analyse en quelques secondes, avec carte d'attention et rapport "
            "telechargeable.</p>", unsafe_allow_html=True)
st.write("")

s1, s2, s3 = st.columns(3)
s1.markdown("<div class='kpi'><div class='v'>14</div><div class='l'>Pathologies detectees</div></div>", unsafe_allow_html=True)
s2.markdown("<div class='kpi'><div class='v'>0.77</div><div class='l'>AUC moyen du modele</div></div>", unsafe_allow_html=True)
s3.markdown("<div class='kpi'><div class='v'>~3s</div><div class='l'>Temps d'analyse</div></div>", unsafe_allow_html=True)
st.write("")

# ===================== MODELE =====================
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Telechargement du modele (une seule fois)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dense_base = m.layers[0]
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = dense_base(inp); x = feat
    for layer in m.layers[1:]:
        x = layer(x)
    return m, tf.keras.Model(inp, [feat, x])

@st.cache_resource
def charger_seuils():
    try:
        if not os.path.exists(THRESH_PATH):
            urllib.request.urlretrieve(THRESH_URL, THRESH_PATH)
        with open(THRESH_PATH) as f: return json.load(f)
    except Exception:
        return {p: 0.5 for p in PATHOLOGIES}

model, grad_model = charger_modele()
SEUILS = charger_seuils()

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def preparer(pil_img):
    g = _clahe.apply(np.array(pil_img.convert("L")).astype("uint8"))
    g = cv2.resize(g, (IMG_SIZE, IMG_SIZE))
    rgb = np.stack([g]*3, axis=-1)
    return np.expand_dims(preprocess_input(rgb.astype("float32")), 0), rgb

def est_radio(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype("float32")
    ecart = (np.abs(arr[...,0]-arr[...,1]).mean()+np.abs(arr[...,1]-arr[...,2]).mean())/2
    h,w = arr.shape[:2]
    return ecart<15 and max(h,w)/min(h,w)<1.6 and arr.mean(-1).std()>25

def gradcam(arr, idx):
    arr = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(arr, training=False)
        tape.watch(conv_out); loss = pred[:, idx]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(0,1,2))
    heat = np.maximum(tf.reduce_sum(conv_out[0]*w, axis=-1).numpy(), 0)
    return cv2.resize(heat/(heat.max()+1e-8), (IMG_SIZE, IMG_SIZE))

def rapport_texte(positifs, ordre):
    d = datetime.now().strftime("%d/%m/%Y a %H:%M")
    t = "="*56+"\n   HOPITAL CHEIKH ZAID - RAPPORT D'AIDE AU DIAGNOSTIC\n"+"="*56+"\n\n"
    t += f"Date d'analyse : {d}\nModele        : DenseNet121 (AUC 0.77)\n\n"
    t += "-"*56+"\nPATHOLOGIES SIGNALEES\n"+"-"*56+"\n"
    if positifs:
        for n,p in positifs: t += f"  - {FR[n]:<28} {p*100:5.1f}%\n"
    else: t += "  Aucune anomalie au-dessus du seuil.\n"
    t += "\n"+"-"*56+"\nDETAIL DES 14 PATHOLOGIES\n"+"-"*56+"\n"
    for n,p in ordre: t += f"  {FR[n]:<28} {p*100:5.1f}%\n"
    t += "\n"+"="*56+"\nAVERTISSEMENT : aide au pre-signalement pedagogique.\n"
    t += "Ne remplace pas l'avis d'un radiologue.\n"+"="*56+"\n"
    return t

# ===================== UPLOAD + RESULTAT =====================
st.markdown("<div class='section-titre'>Deposer une radiographie</div>", unsafe_allow_html=True)
fichier = st.file_uploader("Radiographie thoracique frontale (PA/AP) - PNG ou JPG",
                           type=['png','jpg','jpeg'])

if fichier is not None:
    pil = Image.open(fichier)
    if not est_radio(pil):
        st.warning("⚠️ Image possiblement non-radiographique. Resultat peu fiable.")
    x, rgb = preparer(pil)
    probs = model.predict(x, verbose=0)[0]
    ordre = sorted(zip(PATHOLOGIES, probs), key=lambda t:-t[1])
    positifs = [(n,p) for n,p in ordre if p >= SEUILS.get(n,0.5)]

    r1,r2,r3 = st.columns(3)
    r1.markdown(f"<div class='kpi'><div class='v'>{len(positifs)}</div><div class='l'>Pathologie(s) signalee(s)</div></div>", unsafe_allow_html=True)
    r2.markdown(f"<div class='kpi'><div class='v' style='font-size:1.3rem'>{FR[ordre[0][0]]}</div><div class='l'>Suspicion principale</div></div>", unsafe_allow_html=True)
    r3.markdown(f"<div class='kpi'><div class='v'>{ordre[0][1]*100:.0f}%</div><div class='l'>Confiance max</div></div>", unsafe_allow_html=True)
    st.write("")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-titre'>Radiographie</div>", unsafe_allow_html=True)
        st.image(rgb, use_container_width=True)
    with c2:
        st.markdown("<div class='section-titre'>Zone d'attention</div>", unsafe_allow_html=True)
        try:
            top = int(np.argmax(probs))
            heat = gradcam(x, top)
            heat_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            st.image(cv2.addWeighted(rgb,0.6,heat_c,0.4,0), use_container_width=True,
                     caption=f"Zone : {FR[PATHOLOGIES[top]]}")
        except Exception as e:
            st.image(rgb, use_container_width=True); st.warning(f"Grad-CAM: {e}")

    st.markdown("<div class='section-titre'>Resultat</div>", unsafe_allow_html=True)
    if positifs:
        for n,p in positifs:
            st.markdown(f"<div class='carte'><span class='nom'>{FR[n]}</span> "
                        f"<span class='pct'>{p*100:.0f}%</span></div>", unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie detectee au-dessus du seuil.")

    st.markdown("**Top 5 probabilites**")
    for n,p in ordre[:5]:
        st.write(f"{FR[n]} - {p*100:.1f}%"); st.progress(float(p))

    st.markdown("<div class='section-titre'>📄 Rapport de diagnostic</div>", unsafe_allow_html=True)
    txt = rapport_texte(positifs, ordre)
    st.code(txt, language=None)
    d1,d2 = st.columns(2)
    d1.download_button("⬇️ Rapport (.txt)", txt,
        file_name=f"rapport_thorax_{datetime.now():%Y%m%d_%H%M}.txt",
        mime="text/plain", use_container_width=True)
    try:
        from fpdf import FPDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Courier", size=10)
        for l in txt.split("\n"):
            pdf.cell(0,5,l.encode('latin-1','replace').decode('latin-1'), ln=1)
        d2.download_button("⬇️ Rapport (.pdf)", pdf.output(dest='S').encode('latin-1'),
            file_name=f"rapport_thorax_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf", use_container_width=True)
    except Exception:
        d2.info("PDF: ajoutez 'fpdf2' au requirements.txt")

st.markdown("<div class='avert'>⚠️ <b>Avertissement medical.</b> Outil d'aide au pre-signalement "
            "a but pedagogique. Ne remplace pas l'interpretation d'un radiologue. Aucune decision "
            "clinique ne doit reposer sur ce resultat.</div>", unsafe_allow_html=True)
