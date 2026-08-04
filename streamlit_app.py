import os, urllib.request, json, base64
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

st.set_page_config(page_title="Cheikh Zaid - Diagnostic Thorax IA",
                   page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

PATHOLOGIES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass',
               'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema',
               'Emphysema','Fibrosis','Pleural_Thickening','Hernia']
FR = {'Atelectasis':'Atelectasie','Cardiomegaly':'Cardiomegalie',
      'Effusion':'Epanchement pleural','Infiltration':'Infiltrat','Mass':'Masse',
      'Nodule':'Nodule','Pneumonia':'Pneumonie','Pneumothorax':'Pneumothorax',
      'Consolidation':'Consolidation','Edema':'Oedeme','Emphysema':'Emphyseme',
      'Fibrosis':'Fibrose','Pleural_Thickening':'Epaississement pleural','Hernia':'Hernie'}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    .stApp {background:#ffffff; font-family:'Poppins',sans-serif;}
    #MainMenu, footer, header {visibility:hidden;}
    .block-container {padding-top:1rem; max-width:1200px;}

    /* HEADER */
    .hdr {display:flex; align-items:center; justify-content:space-between;
        padding:1rem 1.5rem; background:#fff; border-radius:0 0 16px 16px;
        box-shadow:0 2px 20px rgba(0,0,0,0.04); margin-bottom:0;}
    .hdr-logo {display:flex; align-items:center; gap:12px;}
    .hdr-logo .tri {color:#e8503a; font-size:1.8rem;}
    .hdr-logo .txt {line-height:1.1;}
    .hdr-logo .t1 {font-size:0.7rem; color:#6b7280; letter-spacing:1px; font-weight:600;}
    .hdr-logo .t2 {font-size:1.05rem; color:#1a1a2e; font-weight:800;}
    .hdr-urg {background:#fdeeeb; border-radius:12px; padding:0.5rem 1.2rem; text-align:right;}
    .hdr-urg .u1 {font-size:0.62rem; color:#e8503a; font-weight:700; letter-spacing:1px;}
    .hdr-urg .u2 {font-size:0.95rem; color:#1a1a2e; font-weight:800;}

    /* HERO */
    .hero {background:linear-gradient(135deg,#fdf6f4 0%,#ffffff 60%);
        border-radius:24px; padding:3rem 2.5rem; margin:1.5rem 0;
        border:1px solid #f5e9e6;}
    .hero .badge {display:inline-block; background:#fff; border:1px solid #f4c9bf;
        color:#e8503a; font-weight:600; padding:0.45rem 1.1rem;
        border-radius:30px; font-size:0.82rem; margin-bottom:1.5rem;}
    .hero h1 {font-size:2.8rem; font-weight:800; line-height:1.15; color:#1a1a2e; margin:0;}
    .hero h1 .r {color:#e8503a;}
    .hero p {color:#6b7280; font-size:1.05rem; margin:1.3rem 0 0 0; max-width:600px; line-height:1.6;}

    .kpi {background:#fff; border:1px solid #f0e6e3; border-radius:18px; padding:1.4rem 1rem;
        text-align:center; box-shadow:0 6px 24px rgba(232,80,58,0.07); height:100%;}
    .kpi .v {font-size:2.1rem; font-weight:800; color:#e8503a; line-height:1;}
    .kpi .l {font-size:0.8rem; color:#6b7280; font-weight:500; margin-top:0.4rem;}

    .sect {font-size:1.5rem; font-weight:800; color:#1a1a2e; margin:2rem 0 1rem 0;
        display:flex; align-items:center; gap:12px;}
    .sect::before {content:''; width:6px; height:26px; background:#e8503a; border-radius:3px;}

    .carte {background:#fff; border:1px solid #f0e6e3; border-left:5px solid #e8503a;
        border-radius:14px; padding:1.1rem 1.4rem; margin-bottom:0.8rem;
        box-shadow:0 3px 14px rgba(232,80,58,0.05);
        display:flex; justify-content:space-between; align-items:center;}
    .carte .nom {color:#1a1a2e; font-weight:700; font-size:1.1rem;}
    .carte .pct {color:#e8503a; font-weight:800; font-size:1.2rem;}

    .stButton>button, .stDownloadButton>button {background:#e8503a !important; color:#fff !important;
        border:none !important; border-radius:30px !important; padding:0.65rem 2rem !important;
        font-weight:600 !important; font-family:'Poppins',sans-serif !important;
        box-shadow:0 4px 14px rgba(232,80,58,0.3) !important;}
    .stButton>button:hover, .stDownloadButton>button:hover {background:#d13f2a !important;}
    .stProgress > div > div > div {background:linear-gradient(90deg,#e8503a,#f4a698) !important;}
    [data-testid="stFileUploader"] {background:#fdf6f4; border:2px dashed #f4a698;
        border-radius:18px; padding:1.2rem;}
    .avert {background:#fdf6f4; border-left:4px solid #e8503a; padding:1.1rem 1.4rem;
        border-radius:12px; font-size:0.85rem; color:#8a3a2a; margin-top:2.5rem;}
    .foot {text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:2rem; padding:1.5rem;
        border-top:1px solid #f0e6e3;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
logo_html = "<span class='tri'>◢</span>"
if os.path.exists("logo_hopital.png"):
    with open("logo_hopital.png","rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    logo_html = f"<img src='data:image/png;base64,{b64}' style='height:52px;'>"

st.markdown(f"""
<div class='hdr'>
  <div class='hdr-logo'>{logo_html}
    <div class='txt'><div class='t1'>HOPITAL UNIVERSITAIRE INTERNATIONAL</div>
      <div class='t2'>CHEIKH ZAID · IA Radiologie</div></div>
  </div>
  <div class='hdr-urg'><div class='u1'>URGENCES 24H/7J</div>
    <div class='u2'>+212 (08) 02 00 06 06</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class='hero'>
  <span class='badge'>● Systeme d'aide au diagnostic radiologique</span>
  <h1>Diagnostic <span class='r'>Radiologique</span><br>assiste par Intelligence Artificielle</h1>
  <p>Detection automatique de 14 pathologies thoraciques a partir d'une radiographie du thorax.
     Analyse en quelques secondes, avec carte d'attention (Grad-CAM) et rapport medical telechargeable.</p>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.markdown("<div class='kpi'><div class='v'>14</div><div class='l'>Pathologies detectees</div></div>", unsafe_allow_html=True)
k2.markdown("<div class='kpi'><div class='v'>0.77</div><div class='l'>AUC moyen (fiabilite)</div></div>", unsafe_allow_html=True)
k3.markdown("<div class='kpi'><div class='v'>~3s</div><div class='l'>Temps d'analyse</div></div>", unsafe_allow_html=True)
k4.markdown("<div class='kpi'><div class='v'>DenseNet</div><div class='l'>Architecture 121 couches</div></div>", unsafe_allow_html=True)

# ---------------- MODELE ----------------
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Chargement du modele (une seule fois)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dense_base = m.layers[0]
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = dense_base(inp); x = feat
    for layer in m.layers[1:]: x = layer(x)
    return m, tf.keras.Model(inp, [feat, x])

@st.cache_resource
def charger_seuils():
    try:
        if not os.path.exists(THRESH_PATH):
            urllib.request.urlretrieve(THRESH_URL, THRESH_PATH)
        with open(THRESH_PATH) as f: return json.load(f)
    except Exception: return {p:0.5 for p in PATHOLOGIES}

model, grad_model = charger_modele()
SEUILS = charger_seuils()

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def preparer(pil):
    g = _clahe.apply(np.array(pil.convert("L")).astype("uint8"))
    g = cv2.resize(g,(IMG_SIZE,IMG_SIZE)); rgb = np.stack([g]*3,-1)
    return np.expand_dims(preprocess_input(rgb.astype("float32")),0), rgb

def est_radio(pil):
    a = np.array(pil.convert("RGB")).astype("float32")
    ec = (np.abs(a[...,0]-a[...,1]).mean()+np.abs(a[...,1]-a[...,2]).mean())/2
    h,w = a.shape[:2]
    return ec<15 and max(h,w)/min(h,w)<1.6 and a.mean(-1).std()>25

def gradcam(arr, idx):
    arr = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        co, pr = grad_model(arr, training=False); tape.watch(co); loss = pr[:,idx]
    gr = tape.gradient(loss, co); w = tf.reduce_mean(gr, axis=(0,1,2))
    h = np.maximum(tf.reduce_sum(co[0]*w, -1).numpy(), 0)
    return cv2.resize(h/(h.max()+1e-8),(IMG_SIZE,IMG_SIZE))

def rapport(positifs, ordre, est_rx):
    d = datetime.now().strftime("%d/%m/%Y a %H:%M")
    ref = "RX-"+datetime.now().strftime("%Y%m%d-%H%M%S")
    t  = "#"*60+"\n"
    t += "   HOPITAL UNIVERSITAIRE INTERNATIONAL CHEIKH ZAID\n"
    t += "   Service d'Imagerie Medicale - Unite IA\n"
    t += "   RAPPORT D'AIDE AU DIAGNOSTIC RADIOLOGIQUE\n"
    t += "#"*60+"\n\n"
    t += f"Reference   : {ref}\n"
    t += f"Date        : {d}\n"
    t += f"Modele IA   : DenseNet121 (NIH ChestX-ray14, AUC moyen 0.77)\n"
    t += f"Type examen : Radiographie thoracique frontale\n"
    if not est_rx:
        t += "ALERTE      : image possiblement non-radiographique (fiabilite reduite)\n"
    t += "\n"+"="*60+"\n  CONCLUSION DE L'ANALYSE\n"+"="*60+"\n\n"
    if positifs:
        t += "Pathologie(s) signalee(s) au-dessus du seuil de confiance :\n\n"
        for n,p in positifs:
            niv = "ELEVEE" if p>0.7 else "MODEREE" if p>0.5 else "FAIBLE"
            t += f"   [+] {FR[n]:<26} {p*100:5.1f}%   (suspicion {niv})\n"
    else:
        t += "   Aucune pathologie detectee au-dessus des seuils de confiance.\n"
        t += "   Radiographie sans anomalie signalee par le systeme.\n"
    t += "\n"+"="*60+"\n  DETAIL COMPLET DES 14 PATHOLOGIES\n"+"="*60+"\n\n"
    for n,p in ordre:
        barre = "#"*int(p*20) + "-"*(20-int(p*20))
        t += f"   {FR[n]:<26} |{barre}| {p*100:5.1f}%\n"
    t += "\n"+"#"*60+"\n"
    t += "  AVERTISSEMENT MEDICAL\n"+"#"*60+"\n"
    t += "  Ce rapport est genere par un systeme d'intelligence\n"
    t += "  artificielle a but d'aide au pre-signalement et de\n"
    t += "  demonstration pedagogique. Il ne constitue PAS un\n"
    t += "  diagnostic medical valide et ne remplace en aucun cas\n"
    t += "  l'interpretation d'un medecin radiologue qualifie.\n"
    t += "  Aucune decision clinique ne doit reposer sur ce document.\n"
    t += "#"*60+"\n"
    return t, ref

# ---------------- UPLOAD ----------------
st.markdown("<div class='sect'>Deposer une radiographie</div>", unsafe_allow_html=True)
fichier = st.file_uploader("Radiographie thoracique frontale (PA/AP) - PNG ou JPG",
                           type=['png','jpg','jpeg'])

if fichier is not None:
    pil = Image.open(fichier)
    rx_ok = est_radio(pil)
    if not rx_ok:
        st.warning("⚠️ Image possiblement non-radiographique. Le resultat peut etre peu fiable.")
    x, rgb = preparer(pil)
    probs = model.predict(x, verbose=0)[0]
    ordre = sorted(zip(PATHOLOGIES, probs), key=lambda t:-t[1])
    positifs = [(n,p) for n,p in ordre if p >= SEUILS.get(n,0.5)]

    r1,r2,r3 = st.columns(3)
    r1.markdown(f"<div class='kpi'><div class='v'>{len(positifs)}</div><div class='l'>Pathologie(s) signalee(s)</div></div>", unsafe_allow_html=True)
    r2.markdown(f"<div class='kpi'><div class='v' style='font-size:1.4rem'>{FR[ordre[0][0]]}</div><div class='l'>Suspicion principale</div></div>", unsafe_allow_html=True)
    r3.markdown(f"<div class='kpi'><div class='v'>{ordre[0][1]*100:.0f}%</div><div class='l'>Confiance maximale</div></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sect'>Radiographie analysee</div>", unsafe_allow_html=True)
        st.image(rgb, use_container_width=True)
    with c2:
        st.markdown("<div class='sect'>Carte d'attention (Grad-CAM)</div>", unsafe_allow_html=True)
        try:
            top = int(np.argmax(probs)); heat = gradcam(x, top)
            hc = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            st.image(cv2.addWeighted(rgb,0.6,hc,0.4,0), use_container_width=True,
                     caption=f"Zone influencant la decision : {FR[PATHOLOGIES[top]]}")
        except Exception as e:
            st.image(rgb, use_container_width=True); st.warning(f"Grad-CAM: {e}")

    st.markdown("<div class='sect'>Conclusion</div>", unsafe_allow_html=True)
    if positifs:
        for n,p in positifs:
            st.markdown(f"<div class='carte'><span class='nom'>{FR[n]}</span>"
                        f"<span class='pct'>{p*100:.0f}%</span></div>", unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie detectee au-dessus du seuil de confiance.")

    st.markdown("**Top 5 des probabilites**")
    for n,p in ordre[:5]:
        st.write(f"{FR[n]} — {p*100:.1f}%"); st.progress(float(p))

    st.markdown("<div class='sect'>📄 Rapport medical</div>", unsafe_allow_html=True)
    txt, ref = rapport(positifs, ordre, rx_ok)
    st.code(txt, language=None)
    d1,d2 = st.columns(2)
    d1.download_button("⬇️ Telecharger (.txt)", txt,
        file_name=f"{ref}.txt", mime="text/plain", use_container_width=True)
    try:
        from fpdf import FPDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Courier", size=9)
        for l in txt.split("\n"):
            pdf.cell(0,4.5,l.encode('latin-1','replace').decode('latin-1'), ln=1)
        d2.download_button("⬇️ Telecharger (.pdf)", pdf.output(dest='S').encode('latin-1'),
            file_name=f"{ref}.pdf", mime="application/pdf", use_container_width=True)
    except Exception:
        d2.info("PDF : ajoutez 'fpdf2' au requirements.txt")

st.markdown("<div class='avert'>⚠️ <b>Avertissement medical.</b> Ce systeme est un outil d'aide au "
            "pre-signalement a but pedagogique. Il ne constitue pas un dispositif medical valide et ne "
            "remplace en aucun cas l'interpretation d'un radiologue. Aucune decision clinique ne doit "
            "reposer sur ce resultat.</div>", unsafe_allow_html=True)
st.markdown("<div class='foot'>Hopital Universitaire International Cheikh Zaid · Unite IA Imagerie "
            "Medicale · Projet de stage</div>", unsafe_allow_html=True)
