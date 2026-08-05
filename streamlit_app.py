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

    .hdr {display:flex; align-items:center; justify-content:space-between;
        padding:1rem 1.5rem; background:#fff; border-radius:0 0 16px 16px;
        box-shadow:0 2px 20px rgba(0,0,0,0.04);}
    .hdr-logo {display:flex; align-items:center; gap:12px;}
    .hdr-logo .tri {color:#e8503a; font-size:1.8rem;}
    .hdr-logo .txt {line-height:1.1;}
    .hdr-logo .t1 {font-size:0.7rem; color:#6b7280; letter-spacing:1px; font-weight:600;}
    .hdr-logo .t2 {font-size:1.05rem; color:#1a1a2e; font-weight:800;}
    .hdr-urg {background:#fdeeeb; border-radius:12px; padding:0.5rem 1.2rem; text-align:right;}
    .hdr-urg .u1 {font-size:0.62rem; color:#e8503a; font-weight:700; letter-spacing:1px;}
    .hdr-urg .u2 {font-size:0.95rem; color:#1a1a2e; font-weight:800;}

    .hero {background:linear-gradient(135deg,#fdf6f4 0%,#ffffff 60%);
        border-radius:24px; padding:3rem 2.5rem; margin:1.5rem 0; border:1px solid #f5e9e6;}
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

    .stButton>button, .stDownloadButton>button {background:#e8503a !important; color:#fff !important;
        border:none !important; border-radius:30px !important; padding:0.65rem 2rem !important;
        font-weight:600 !important; font-family:'Poppins',sans-serif !important;
        box-shadow:0 4px 14px rgba(232,80,58,0.3) !important;}
    .stButton>button:hover, .stDownloadButton>button:hover {background:#d13f2a !important;}
    [data-testid="stFileUploader"] {background:#fdf6f4; border:2px dashed #f4a698;
        border-radius:18px; padding:1.2rem;}
    .avert {background:#fdf6f4; border-left:4px solid #e8503a; padding:1.1rem 1.4rem;
        border-radius:12px; font-size:0.85rem; color:#8a3a2a; margin-top:2.5rem;}
    .foot {text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:2rem; padding:1.5rem;
        border-top:1px solid #f0e6e3;}

    .rap, .rap * {color:#1a1a2e !important;}
    .rap td, .rap th {color:#374151 !important;}
    .rap .badge-niv {color:#ffffff !important;}
    .diag, .diag * {color:#1a1a2e !important;}
    .diag .badge-niv, .diag .gros {color:inherit !important;}
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

st.markdown("""
<div class='hero'>
  <span class='badge'>● Systeme d'aide au diagnostic radiologique</span>
  <h1>Diagnostic <span class='r'>Radiologique</span><br>assiste par Intelligence Artificielle</h1>
  <p>Detection automatique de 14 pathologies thoraciques a partir d'une radiographie du thorax.
     Analyse en quelques secondes, avec carte d'attention (Grad-CAM) et compte-rendu medical telechargeable.</p>
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

def evaluer(ordre):
    n1, p1 = ordre[0]; n2, p2 = ordre[1]
    ecart = p1 - p2
    if p1 >= 0.75 and ecart >= 0.15:
        return ("Suspicion élevée", "#c0392b",
                "Une seule pathologie se détache nettement. Résultat cohérent.")
    if p1 >= 0.60 and ecart >= 0.10:
        return ("Suspicion modérée", "#e8503a",
                "Orientation probable, à confirmer par un radiologue.")
    if p1 >= 0.45:
        return ("Résultat incertain", "#f39c12",
                f"Le modèle ne tranche pas clairement : plusieurs pathologies ont des "
                f"scores proches ({FR[n2]} à {p2*100:.1f}%). Lecture humaine indispensable.")
    return ("Aucune anomalie signalée", "#27ae60",
            "Aucune pathologie au-dessus du seuil de confiance.")

def rapport_html(positifs, ordre, est_rx):
    d = datetime.now().strftime("%d/%m/%Y à %H:%M")
    ref = "RX-"+datetime.now().strftime("%Y%m%d-%H%M%S")
    niv, coul_niv, msg = evaluer(ordre)
    n1, p1 = ordre[0]
    lignes = ""
    for n,p in ordre:
        if p >= 0.75: coul, lab = "#c0392b", "Élevée"
        elif p >= 0.60: coul, lab = "#e8503a", "Modérée"
        elif p >= 0.45: coul, lab = "#f39c12", "Incertaine"
        else: coul, lab = "#9ca3af", "Faible"
        signal = "background:#fdeeeb;" if p >= 0.60 else ""
        lignes += (f"<tr style='{signal}'><td style='padding:9px 14px;border-bottom:1px solid #f0e6e3;"
                   f"color:#1a1a2e;font-weight:500;'>{FR[n]}</td>"
                   f"<td style='padding:9px 14px;border-bottom:1px solid #f0e6e3;text-align:right;"
                   f"font-weight:700;color:{coul};'>{p*100:.1f}%</td>"
                   f"<td style='padding:9px 14px;border-bottom:1px solid #f0e6e3;text-align:center;'>"
                   f"<span class='badge-niv' style='background:{coul};padding:3px 12px;"
                   f"border-radius:20px;font-size:0.75rem;font-weight:600;'>{lab}</span></td></tr>")
    alerte = ""
    if not est_rx:
        alerte = ("<div style='background:#fff4e5;border-left:4px solid #ff9800;padding:8px 14px;"
                  "border-radius:6px;margin:10px 0;color:#7a4b00;font-size:0.85rem;'>"
                  "⚠️ Image possiblement non-radiographique — fiabilité réduite.</div>")
    return f"""
    <div class='rap' style='background:#fff;border:1px solid #f0e6e3;border-radius:16px;padding:2rem;
        box-shadow:0 6px 24px rgba(0,0,0,0.05);font-family:Poppins,sans-serif;'>
      <div style='border-bottom:3px solid #e8503a;padding-bottom:1rem;margin-bottom:1.2rem;'>
        <div style='font-size:1.2rem;font-weight:800;color:#1a1a2e;'>Hôpital Universitaire International Cheikh Zaïd</div>
        <div style='color:#6b7280;font-size:0.9rem;'>Service d'Imagerie Médicale — Unité d'Intelligence Artificielle</div>
        <div style='color:#e8503a;font-weight:700;margin-top:6px;'>COMPTE-RENDU D'AIDE AU DIAGNOSTIC RADIOLOGIQUE</div>
      </div>
      <table style='width:100%;font-size:0.9rem;color:#374151;margin-bottom:1rem;'>
        <tr><td style='padding:3px 0;'><b>Référence</b></td><td>{ref}</td>
            <td style='padding:3px 0;'><b>Date</b></td><td>{d}</td></tr>
        <tr><td style='padding:3px 0;'><b>Examen</b></td><td>Radiographie thoracique frontale</td>
            <td style='padding:3px 0;'><b>Modèle IA</b></td><td>DenseNet121 (AUC 0.77)</td></tr>
      </table>
      {alerte}
      <div style='font-weight:700;color:#1a1a2e;margin:1rem 0 0.5rem 0;'>Orientation principale</div>
      <div style='background:#fdf6f4;border-radius:10px;padding:14px 18px;'>
        <span style='font-size:1.15rem;font-weight:700;color:#1a1a2e;'>{FR[n1]}</span>
        <span style='color:{coul_niv};font-weight:800;margin-left:10px;'>{p1*100:.1f}%</span>
        <span class='badge-niv' style='background:{coul_niv};padding:3px 12px;border-radius:20px;
            font-size:0.75rem;font-weight:600;margin-left:10px;'>{niv}</span>
        <div style='color:#6b7280;font-size:0.85rem;margin-top:8px;'>{msg}</div>
      </div>
      <div style='font-weight:700;color:#1a1a2e;margin:1.4rem 0 0.5rem 0;'>Détail des 14 pathologies</div>
      <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
        <tr style='background:#f9fafb;'>
          <th style='padding:9px 14px;text-align:left;color:#6b7280;'>Pathologie</th>
          <th style='padding:9px 14px;text-align:right;color:#6b7280;'>Probabilité</th>
          <th style='padding:9px 14px;text-align:center;color:#6b7280;'>Suspicion</th></tr>
        {lignes}
      </table>
      <div style='background:#fdf6f4;border-radius:10px;padding:12px 16px;margin-top:1.5rem;
          font-size:0.8rem;color:#8a3a2a;'>
        <b>Avertissement.</b> Ce compte-rendu est généré par un système d'intelligence artificielle
        à visée d'aide au pré-signalement. Il ne constitue pas un diagnostic médical et ne remplace pas
        l'interprétation d'un médecin radiologue. La validation par un praticien qualifié est requise.
      </div>
    </div>
    """, ref

def rapport_texte(positifs, ordre, est_rx):
    d = datetime.now().strftime("%d/%m/%Y a %H:%M")
    ref = "RX-"+datetime.now().strftime("%Y%m%d-%H%M%S")
    niv, _, msg = evaluer(ordre)
    n1, p1 = ordre[0]
    t  = "HOPITAL UNIVERSITAIRE INTERNATIONAL CHEIKH ZAID\n"
    t += "Service d'Imagerie Medicale - Unite IA\n"
    t += "COMPTE-RENDU D'AIDE AU DIAGNOSTIC RADIOLOGIQUE\n\n"
    t += f"Reference : {ref}\nDate      : {d}\n"
    t += "Examen    : Radiographie thoracique frontale\n"
    t += "Modele IA : DenseNet121 (AUC 0.77)\n\n"
    t += "ORIENTATION PRINCIPALE\n"
    t += f"  {FR[n1]} : {p1*100:.1f}%  [{niv}]\n"
    t += f"  {msg}\n\n"
    t += "DETAIL DES 14 PATHOLOGIES\n"
    for n,p in ordre: t += f"  {FR[n]:<26} {p*100:5.1f}%\n"
    t += "\nAvertissement : aide au pre-signalement. Validation par un\n"
    t += "radiologue requise. Ne constitue pas un diagnostic medical.\n"
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
    positifs = [(n,p) for n,p in ordre if p >= 0.60]

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

    # ---------- DIAGNOSTIC UNIQUE ----------
    st.markdown("<div class='sect'>Résultat de l'analyse</div>", unsafe_allow_html=True)
    n1, p1 = ordre[0]
    niv, coul, msg = evaluer(ordre)
    st.markdown(f"""
    <div class='diag' style='background:#fff;border:1px solid #f0e6e3;border-top:5px solid {coul};
        border-radius:18px;padding:2.4rem 2rem;text-align:center;
        box-shadow:0 6px 28px rgba(0,0,0,0.06);'>
      <div style='color:#6b7280 !important;font-size:0.82rem;letter-spacing:1.5px;
          text-transform:uppercase;margin-bottom:10px;'>Orientation principale</div>
      <div style='font-size:2.3rem;font-weight:800;color:#1a1a2e !important;margin-bottom:4px;'>{FR[n1]}</div>
      <div class='gros' style='font-size:1.7rem;font-weight:800;color:{coul} !important;
          margin-bottom:16px;'>{p1*100:.1f}%</div>
      <span class='badge-niv' style='background:{coul};color:#fff !important;padding:6px 20px;
          border-radius:20px;font-size:0.85rem;font-weight:600;'>{niv}</span>
      <div style='color:#6b7280 !important;font-size:0.9rem;margin-top:18px;max-width:540px;
          margin-left:auto;margin-right:auto;line-height:1.5;'>{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- DETAIL REPLIABLE ----------
    with st.expander("Voir le détail des 14 pathologies"):
        det = ""
        for n,p in ordre:
            larg = int(p*100)
            cc = "#c0392b" if p>=0.75 else "#e8503a" if p>=0.60 else "#f39c12" if p>=0.45 else "#f4a698"
            det += (f"<div style='margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                    f"<span style='color:#1a1a2e;font-weight:600;font-size:0.92rem;'>{FR[n]}</span>"
                    f"<span style='color:{cc};font-weight:700;'>{p*100:.1f}%</span></div>"
                    f"<div style='background:#f0e6e3;border-radius:20px;height:8px;overflow:hidden;'>"
                    f"<div style='background:linear-gradient(90deg,{cc},#f4a698);width:{larg}%;"
                    f"height:100%;border-radius:20px;'></div></div></div>")
        st.markdown(f"<div class='rap'>{det}</div>", unsafe_allow_html=True)

    # ---------- COMPTE-RENDU ----------
    st.markdown("<div class='sect'>Compte-rendu médical</div>", unsafe_allow_html=True)
    html_rap, ref = rapport_html(positifs, ordre, rx_ok)
    txt_rap, _ = rapport_texte(positifs, ordre, rx_ok)
    st.markdown(html_rap, unsafe_allow_html=True)
    st.write("")
    d1,d2 = st.columns(2)
    d1.download_button("⬇️ Télécharger le compte-rendu (.txt)", txt_rap,
        file_name=f"{ref}.txt", mime="text/plain", use_container_width=True)
    try:
        from fpdf import FPDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Courier", size=9)
        for l in txt_rap.split("\n"):
            pdf.cell(0,5,l.encode('latin-1','replace').decode('latin-1'), ln=1)
        d2.download_button("⬇️ Télécharger le compte-rendu (.pdf)", pdf.output(dest='S').encode('latin-1'),
            file_name=f"{ref}.pdf", mime="application/pdf", use_container_width=True)
    except Exception:
        d2.info("PDF : ajoutez 'fpdf2' au requirements.txt")

st.markdown("<div class='avert'>⚠️ <b>Avertissement medical.</b> Ce systeme est un outil d'aide au "
            "pre-signalement. Il ne constitue pas un dispositif medical valide et ne remplace en aucun "
            "cas l'interpretation d'un radiologue. La validation par un praticien qualifie est requise "
            "pour toute decision clinique.</div>", unsafe_allow_html=True)
st.markdown("<div class='foot'>Hôpital Universitaire International Cheikh Zaïd · "
            "Unité d'Intelligence Artificielle · Service d'Imagerie Médicale</div>", unsafe_allow_html=True)
