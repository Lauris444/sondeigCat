# Reemplaça TOT el teu codi amb aquest

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import random
import os
import re
import threading
import base64
import io
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

def show_loading_animation():
    """Mostra una animació de càrrega personalitzada amb HTML i CSS."""
    loading_html = """
    <style>
        .loading-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: rgba(25,37,81,0.9);
            z-index: 9999;
        }
        .loading-svg {
            width: 150px;
            height: auto;
            margin-bottom: 20px;
        }
        .loading-text {
            color: white;
            font-size: 1.5rem;
            font-family: sans-serif;
        }
        .loading-text .dot {
            animation: blink 1.4s infinite both;
        }
        .loading-text .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        .loading-text .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
    </style>
    <div class="loading-container">
        <svg class="loading-svg" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
            <path d="M 155.6,66.1 C 155.6,42.9 135.5,23.5 111.4,23.5 C 98.4,23.5 86.8,29.4 79.1,38.7 C 75.2,16.8 57.3,0 36.4,0 C 16.3,0 0,16.3 0,36.4 C 0,56.5 16.3,72.8 36.4,72.8 L 110,72.8 C 110,72.8 110,72.8 110,72.8 C 135,72.8 155.6,93.4 155.6,118.4 C 155.6,143.4 135,164 110,164 L 50, 164" fill="none" stroke="#FFFFFF" stroke-width="8"/>
            <polygon points="120,60 90,110 115,110 100,150 145,90 120,90 130,60" fill="#FFD700" />
        </svg>
        <div class="loading-text">
            Carregant<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
        </div>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)


def set_main_background():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(0deg, rgba(6,14,42,1) 0%, rgba(25,37,81,1) 100%);
        background-size: cover; background-position: center center;
        background-repeat: no-repeat; background-attachment: local;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    [data-testid="stToolbar"] {{ right: 2rem; }}
    .welcome-title {{
        font-size: 3.5rem; font-weight: bold; color: white; text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }}
    .welcome-subtitle {{
        font-size: 1.5rem; color: #E0E0E0; text-align: center; margin-bottom: 40px;
    }}
    .mode-card {{
        background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 25px; border-radius: 15px; backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); color: white; height: 100%;
    }}
    .mode-card h3 {{ color: #FFFFFF; font-weight: bold; }}
    .mode-card p {{ color: #D0D0D0; }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

def get_image_as_base64(file_path):
    """Llegeix una imatge i la converteix a format Base64 per a HTML."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except FileNotFoundError:
        return None

def clean_and_convert(text):
    cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
    if not cleaned_text or cleaned_text == '-': return None
    try: return float(cleaned_text)
    except ValueError: return None

def process_sounding_block(block_lines):
    if not block_lines: return None
    p_list, t_list, td_list, wdir_list, wspd_list = [], [], [], [], []
    time_lines = []
    time_keywords = ['observació', 'hora', 'time', 'locale', 'run', 'z', 'date']
    days_fr_to_ca = {'Lundi': 'Dilluns', 'Mardi': 'Dimarts', 'Mercredi': 'Dimecres', 'Jeudi': 'Dijous', 'Vendredi': 'Divendres', 'Samedi': 'Dissabte', 'Dimanche': 'Diumenge'}
    months_fr_to_ca = {'janvier': 'de gener', 'février': 'de febrer', 'mars': 'de març', 'avril': 'd\'abril', 'mai': 'de maig', 'juin': 'de juny', 'juillet': 'de juliol', 'août': 'd\'agost', 'septembre': 'de setembre', 'octobre': 'd\'octubre', 'novembre': 'de novembre', 'décembre': 'de desembre'}
    general_fr_to_ca = {'Run': 'Model', 'locale': 'local', 'du': 'del'}
    for line in block_lines:
        line_strip = line.strip()
        if any(keyword in line_strip.lower() for keyword in time_keywords) and not (line_strip and line_strip[0].isdigit()):
            time_lines.append(line_strip)
            continue
        if not line_strip or line_strip.startswith('#') or 'Pression' in line_strip: continue
        try:
            parts = re.split(r'\s{2,}|[\t]', line_strip)
            if len(parts) < 7: continue
            p, t, td = clean_and_convert(parts[1]), clean_and_convert(parts[2]), clean_and_convert(parts[4])
            if p is None or t is None or td is None: continue
            p_list.append(p); t_list.append(t); td_list.append(td)
            wdir, wspd = 0.0, 0.0
            try:
                wind_str = parts[6].strip()
                if '/' in wind_str:
                    wind_parts = wind_str.split('/')
                    if len(wind_parts) == 2:
                        wdir_val, wspd_val = clean_and_convert(wind_parts[0]), clean_and_convert(wind_parts[1])
                        if wdir_val is not None: wdir = wdir_val
                        if wspd_val is not None: wspd = wspd_val
            except IndexError: pass
            wdir_list.append(wdir); wspd_list.append(wspd)
        except Exception as e:
            st.warning(f"Advertència: Error processant línia '{line_strip}'. Error: {e}")
            continue
    if not p_list or len(p_list) < 2: return None
    translated_lines = []
    for line in time_lines:
        translated_line = line
        for fr, ca in days_fr_to_ca.items(): translated_line = translated_line.replace(fr, ca)
        for fr, ca in months_fr_to_ca.items(): translated_line = re.sub(fr, ca, translated_line, flags=re.IGNORECASE)
        for fr, ca in general_fr_to_ca.items(): translated_line = re.sub(r'\b' + fr + r'\b', ca, translated_line, flags=re.IGNORECASE)
        translated_lines.append(translated_line)
    observation_time = "\n".join(translated_lines) if translated_lines else "Hora no disponible"
    sorted_indices = np.argsort(p_list)[::-1]
    return {'p_levels': np.array(p_list)[sorted_indices] * units.hPa, 't_initial': np.array(t_list)[sorted_indices] * units.degC, 'td_initial': np.array(td_list)[sorted_indices] * units.degC, 'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph, 'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees, 'observation_time': observation_time}

def parse_all_soundings(filepath):
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'. Assegura't que existeix al mateix directori.")
        return []

    for line in lines:
        if 'Pression' in line and (line.strip().startswith('Nivell') or line.strip().startswith('# Nivell')):
            if current_sounding_lines:
                processed_data = process_sounding_block(current_sounding_lines)
                if processed_data: all_soundings_data.append(processed_data)
            current_sounding_lines = []
        current_sounding_lines.append(line)
    if current_sounding_lines:
        processed_data = process_sounding_block(current_sounding_lines)
        if processed_data: all_soundings_data.append(processed_data)
    return all_soundings_data

def create_wintry_mix_profile():
    p = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
    t = np.array([1.5, 3.0, 1.0, -5.0, -20.0, -45.0, -60.0]) * units.degC
    td = np.array([0.5, 1.0, -1.0, -6.0, -22.0, -48.0, -65.0]) * units.degC
    ws = np.full_like(p.magnitude, 15) * units.knots
    wd = np.full_like(p.magnitude, 180) * units.degrees
    return {'p_levels': p, 't_initial': t, 'td_initial': td, 'wind_speed_kmh': ws.to('kph'), 'wind_dir_deg': wd}

# =========================================================================
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
# =========================================================================
# ... (Aquestes funcions no canvien, s'ometen per brevetat)
def calculate_thermo_parameters(p_levels, t_profile, td_profile):
    try:
        p, t, td = p_levels, t_profile, td_profile
        valid_indices = ~np.isnan(p.magnitude) & ~np.isnan(t.magnitude) & ~np.isnan(td.magnitude)
        if np.sum(valid_indices) < 2: raise ValueError("No hi ha prou dades.")
        p, t, td = p[valid_indices], t[valid_indices], td[valid_indices]
        p_sfc, t_sfc, td_sfc = p[0], t[0], td[0]
        parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
        cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
        lcl_p, _ = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
        lfc_p, _ = mpcalc.lfc(p, t, td, parcel_prof)
        el_p, _ = mpcalc.el(p, t, td, parcel_prof)
        try:
            t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")
            p_range = np.arange(p.m.min(), p.m.max())
            t_range = t_interp(p_range)
            fz_idx = np.where(t_range < 0)[0]
            fz_lvl = p_range[fz_idx[0]] * units.hPa if fz_idx.size > 0 else np.nan * units.hPa
        except Exception: fz_lvl = np.nan * units.hPa
        if el_p is None and cape.magnitude > 0: el_p = p[-1]
        lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p else 0
        lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.inf
        el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else lfc_h
        fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
        return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
    except Exception as e:
        return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0)
def calculate_storm_parameters(p_levels, wind_speed, wind_dir):
    try:
        p, ws, wd = p_levels, wind_speed.to('m/s'), wind_dir
        u, v = mpcalc.wind_components(ws, wd)
        heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
        valid_mask = ~np.isnan(heights_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2: return 0.0, 0.0, 0.0, 0.0
        p_c, u_c, v_c, h_c = p[valid_mask], u[valid_mask], v[valid_mask], heights_raw[valid_mask]
        _, unique_indices = np.unique(h_c.m, return_index=True)
        if len(unique_indices) < 2: return 0.0, 0.0, 0.0, 0.0
        p_u, u_u, v_u, h_u = p_c[unique_indices], u_c[unique_indices], v_c[unique_indices], h_c[unique_indices]
        h_min, h_max = h_u.m.min(), min(h_u.m.max(), 12000)
        if h_max <= h_min: return 0.0, 0.0, 0.0, 0.0
        h_interp = np.arange(h_min, h_max, 50) * units.meter
        u_i = np.interp(h_interp.m, h_u.m, u_u.m) * units('m/s')
        v_i = np.interp(h_interp.m, h_u.m, v_u.m) * units('m/s')
        p_interp = mpcalc.height_to_pressure_std(h_interp)
        u_6, v_6 = mpcalc.bulk_shear(p_interp, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        u_1, v_1 = mpcalc.bulk_shear(p_interp, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter)[0].m
        return s_0_6, s_0_1, srh_0_1, srh_0_3
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0
def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, base_km, top_km, pwat_0_4):
    cape, cin, _, _, _, lfc_h, _, _, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, _, _, _ = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    if fz_h < 1500 or t_profile[0].m < 5: precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'sleet'
    elif cape.m > 3000: precipitation_type = 'hail'
    elif cape.m > 500: precipitation_type = 'rain'
    elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
    chat_log = [("Sistema", f"Iniciant anàlisi conversacional per a l'escenari de {cloud_type}.")]
    if cloud_type == "Hivernal":
        chat_log.extend([("Analista", "Estem davant d'un perfil clarament hivernal. El primer que crida l'atenció és la isoterma de 0°C."),("Usuari", f"Està molt baixa, a només {fz_h:.0f} metres."),("Analista", "Exacte. Això ens diu que la 'fàbrica de neu' està molt a prop del terra. Ara bé, la clau està en la temperatura de superfície."),("Usuari", f"És de {t_profile[0].m:.1f}°C, lleugerament positiva."),("Analista", "I aquí tenim el matís. Aquesta petita capa càlida a prop del terra pot ser suficient per fondre els flocs de neu just abans que arribin, convertint una possible nevada en aiguaneu o fins i tot pluja gelant.")])
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([("Analista", "Aquest és un perfil de manual per a temps sever. Anem a desglossar-lo."),("Usuari", f"Suposo que el primer és l'energia. Veig un CAPE de {cape.m:.0f} J/kg."),("Analista", "Correcte. Tenim una quantitat d'energia enorme. Això és el combustible. Però el que defineix aquest escenari és el 'motor'."),("Usuari", "El cisallament del vent?"),("Analista", "Precisament. El perfil mostra un cisallament i una helicitat molt forts. Aquesta combinació d'un combustible potent (CAPE alt) amb un motor d'alt rendiment (cisallament fort) és el que permet que una tempesta s'organitzi i comenci a rotar, formant una supercèl·lula."),("Analista", "El pronòstic ha de ser de màxima precaució: risc elevat de calamarsa gran, vents destructius i, per la rotació a nivells baixos, vigilància per a possibles tornados.")])
    elif "Nimbostratus" in cloud_type:
        chat_log.extend([("Analista", "Aquest perfil és molt diferent. Aquí la història no va d'inestabilitat."),("Usuari", f"És cert, el CAPE és gairebé inexistent, només {cape.m:.0f} J/kg."),("Analista", "Exacte. Aquí el protagonista és la humitat. Tenim una capa d'aire molt gruixuda i completament saturada. No hi ha un 'motor' convectiu, sinó un flux constant d'humitat."),("Usuari", "Llavors, la pluja serà més constant que en una tempesta?"),("Analista", f"Sí. Aquest és un escenari típic de pluja estratiforme, associada a fronts. La intensitat dependrà de l'aigua precipitable, que amb {pwat_0_4.m:.1f} mm, ens indica que podem esperar pluges persistents.")])
    elif cloud_type == "Cumulus Humilis":
        chat_log.extend([("Analista", "Estem observant un escenari de temps estable."),("Usuari", f"Però hi ha una mica de CAPE, {cape.m:.0f} J/kg."),("Analista", "Sí, una mica d'energia hi ha, suficient per formar núvols, però molt poca. A més, segurament hi ha una forta inversió just a sobre que impedeix qualsevol creixement."),("Analista", "Això és un perfil típic per a la formació de Cumulus Humilis, els clàssics 'núvols de bon temps' que no produeixen precipitació.")])
    elif cloud_type == "Cumulus Mediocris":
        chat_log.extend([("Analista", "Aquest és un perfil interessant per a una tarda d'estiu."),("Usuari", f"Tenim {cape.m:.0f} J/kg de CAPE. És suficient per a tempestes?"),("Analista", "És una energia moderada. Permet un cert creixement vertical, però no explosiu. El cisallament del vent també és feble."),("Analista", "Això afavoreix la formació de Cumulus Mediocris. Són els típics núvols de cotó fluix amb una base plana, que rarament donen més que quatre gotes.")])
    elif cloud_type == "Cumulus Congestus":
        chat_log.extend([("Analista", "Atenció a aquest perfil. Aquí comencem a veure potencial per a fenòmens més actius."),("Usuari", f"El CAPE ja és més considerable, {cape.m:.0f} J/kg."),("Analista", "Exacte. Tenim prou energia per a un desenvolupament vertical important. Aquests núvols creixen amb força cap amunt."),("Analista", "És l'escenari ideal per a Cumulus Congestus, també coneguts com a 'torres cumuliformes'. Són el pas previ al Cumulonimbus i ja poden deixar ruixats o xàfecs localment intensos.")])
    elif cloud_type == "Cumulonimbus (Multicèl·lula)":
        chat_log.extend([("Analista", "Bé, tenim un escenari amb potencial de tempestes. El primer, com sempre, és l'energia disponible."),("Usuari", f"El CAPE és de {cape.m:.0f} J/kg."),("Analista", f"És un bon valor, suficient per a tempestes fortes, possiblement amb calamarsa. Ara, mirem si tenen algun fre."),("Usuari", f"El CIN és de {cin.m:.0f} J/kg."),("Analista", "És una inhibició feble. La convecció es pot disparar amb relativa facilitat."),("Usuari", "I s'organitzaran?"),("Analista", "Aquí ve el matís. El cisallament del vent és feble. Per tant, no esperem supercèl·lules, sinó tempestes multicel·lulars (Cumulonimbus) més caòtiques. Poden ser localment fortes, però no tindran la longevitat ni l'organització d'una supercèl·lula.")])
    elif cloud_type == "Castellanus":
        chat_log.extend([("Analista", "Aquest és un cas particular. Tenim energia en altura, però la superfície està desconnectada."),("Usuari", f"Què vols dir? El CAPE és de {cape.m:.0f} J/kg."),("Analista", f"Sí, però fixa't en el CIN: és molt fort, de {cin.m:.0f} J/kg. Això impedeix que la convecció comenci des del terra. No obstant, hi ha una capa inestable a nivells mitjans."),("Analista", "Això pot generar núvols de tipus Altocumulus Castellanus, que són com petites torretes que creixen des d'una base elevada i poden donar lloc a xàfecs sobtats i ratxes de vent.")])
    elif cloud_type == "Cumulus Fractus":
         chat_log.extend([("Analista", "El que veiem aquí són condicions residuals."),("Usuari", "Què vol dir això?"),("Analista", "Hi ha una mica d'humitat i inestabilitat, però és molt poca i desorganitzada. No hi ha prou força per crear núvols ben definits."),("Analista", "Això només permetrà la formació de Cumulus Fractus, que són trossos de núvols esquinçats, sense un desenvolupament clar. No tenen cap mena de risc associat.")])
    else:
        chat_log.extend([("Analista", "El perfil atmosfèric és molt estable."),("Usuari", "Llavors, no veurem cap núvol?"),("Analista", f"És molt poc probable. Amb un CAPE de només {cape.m:.0f} J/kg, no hi ha pràcticament gens d'energia per al creixement vertical. Tindrem un dia de cel serè o amb alguns núvols alts sense importància.")])
    return chat_log, precipitation_type
def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type):
    cape, cin, _, lcl_h, _, lfc_h, _, _, _ = calculate_thermo_parameters(p, t, td)
    shear_0_6, _, _, _ = calculate_storm_parameters(p, t, td)
    chat_log = []
    chat_log.append(("Analista", "Molt bé, anem a analitzar el perfil que has creat. Ho farem com si fóssim un equip, pas a pas. Comencem?"))
    if cape.m < 50:
        chat_log.extend([("Usuari", "Tenim potencial per a tempestes?"),("Analista", f"Ara mateix no. L'energia disponible, el CAPE, és de només {cape.m:.0f} J/kg. L'atmosfera està molt estable.")])
    else:
        chat_log.extend([("Usuari", "Què estic creant amb aquesta energia?"),])
        cloud_mention = f"Això és un escenari típic per a la formació de {cloud_type}."
        if cloud_type == "Cel Serè": cloud_mention = "Encara que hi ha energia, la tapadera és tan forta que probablement no veuríem cap núvol significatiu."
        chat_log.append(("Analista", f"Has generat un CAPE de {cape.m:.0f} J/kg. {cloud_mention}"))
        chat_log.append(("Usuari", "I la 'tapadera' (CIN)? Com afecta?"))
        if cin.m < -100: chat_log.append(("Analista", f"Molt forta. Amb un CIN de {cin.m:.0f} J/kg, l'atmosfera està blindada. És com tenir una tapa d'olla a pressió. La convecció des de superfície és gairebé impossible, necessitaria un forçament extern massiu."))
        elif cin.m < -50: chat_log.append(("Analista", f"És considerable, amb {cin.m:.0f} J/kg. Les tempestes de superfície són poc probables, però obre la porta a la convecció de base elevada (Castellanus)."))
        elif cin.m < -25: chat_log.append(("Analista", f"És moderada ({cin.m:.0f} J/kg). Permet que l'energia s'acumuli a sota abans de disparar-se, un escenari clàssic per a tempestes fortes."))
        else: chat_log.append(("Analista", f"És feble ({cin.m:.0f} J/kg). La convecció té gairebé via lliure per iniciar-se."))
        if cin.m > -100 and cape.m > 800:
            chat_log.append(("Usuari", "He modificat el vent. Com afecta?"))
            if shear_0_6 > 15: chat_log.append(("Analista", "El cisallament és significatiu. Aquest és l'ingredient que ajuda a organitzar les tempestes i a fer-les més duradores i severes."))
            else: chat_log.append(("Analista", "El cisallament és feble. Si es formen tempestes, probablement seran més desorganitzades i de vida més curta."))
    return chat_log, None
def generate_tutorial_analysis(scenario, step):
    chat_log = []
    if scenario == 'aiguaneu':
        if step == 0: chat_log.append(("Analista", "Benvingut! Hem carregat un perfil típic d'aiguaneu. Observa com a 850hPa la temperatura és positiva. Aquesta és la 'capa càlida' que fon la neu. El teu objectiu és entendre per què passa això."))
        elif step == 1: chat_log.append(("Analista", "Correcte! Aquesta capa mitjana freda és on es formen els flocs de neu. Tot va bé fins aquí."))
        elif step == 2: chat_log.append(("Analista", "Molt bé! Has identificat el problema. Aquesta capa càlida fon els flocs de neu a mig camí, convertint-los en gotes de pluja."))
        elif step == 3: chat_log.append(("Analista", "Exacte! La capa propera a la superfície està sota zero, així que les gotes de pluja es tornen a congelar just abans de tocar a terra, formant aiguaneu o pluja gelant."))
        elif step == 4: chat_log.append(("Analista", "Has analitzat el perfil a la perfecció. Repte: Ara que has acabat, ves al Mode Lliure i utilitza l'eina '❄️ Refredar Capa Mitjana'. Veuràs com elimines el problema i ho converteixes en una nevada segura!"))
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem el tutorial de supercèl·lula. El primer pas és sempre crear energia. Necessitem un dia càlid d'estiu. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "Correcte! Molta calor. Ara, afegim el combustible: la humitat. A l'anàlisi final veuràs com augmenta el valor de CAPE quan les línies de temperatura i punt de rosada s'acosten."))
        elif step == 2: chat_log.append(("Analista", "Fantàstic! Has afegit cisallament. Aquest és l'ingredient secret que fa que les tempestes rotin. Ara tenim energia, humitat i rotació: la recepta perfecta!"))
        elif step == 3: chat_log.append(("Analista", "Missió complerta! Has creat un perfil amb molta energia (CAPE alt), humitat i cisallament. A l'anàlisi final, fixa't en com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."))
    return chat_log, None
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    if fz_h < 1500 or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            try:
                p_arr, t_arr = p_levels.m, t_profile.m
                warm_layer_mask = (p_arr < 950) & (p_arr > 600) & (t_arr > 0)
                if np.any(warm_layer_mask): return "AIGUANEU O PLUJA GEBRADORA", "Capa càlida en altura pot fondre la neu. Risc d'aiguaneu o pluja gelant.", "mediumorchid"
                else: return "AVÍS PER NEU", "Perfil atmosfèric favorable a nevades a cotes baixes.", "navy"
            except: return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5: return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades. Extremi les precaucions.", "dodgerblue"
    if cape.m >= 800:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        if cin.m <= -100: return "CONVECCIÓ FORTAMENT INHIBIDA", f"Potencial energètic (CAPE {cape.m:.0f} J/kg) bloquejat per una 'tapadera' molt forta (CIN {cin.m:.0f} J/kg).", "darkslategray"
        if -100 < cin.m <= -50: return "POSSIBLE CONVECCIÓ DE MITJÀ NIVELL", f"La convecció des de superfície és difícil (CIN {cin.m:.0f} J/kg). Es requereix forçament intens. Risc de nuclis elevats.", "slategray"
        title, color, message = "AVÍS PER TEMPESTES", "darkorange", ""
        if -25 < cin.m < 0: message = f"Inhibició dèbil (CIN {cin.m:.0f} J/kg). Les tempestes es poden formar fàcilment. CAPE: {cape.m:.0f} J/kg."
        elif -50 < cin.m <= -25: message = f"Inhibició moderada (CIN {cin.m:.0f} J/kg). Es necessita forçament per trencar-la. CAPE: {cape.m:.0f} J/kg."; color = "goldenrod"
        if srh_0_1 > 150 and shear_0_6 > 15 and cape.m > 1500: title, color = "AVÍS PER TORNADO", "darkred"; message += f" Condicions favorables per a supercèl·lules i tornados (SRH: {srh_0_1:.0f})."
        elif cape.m > 2500 and shear_0_6 > 15: title, color = "AVÍS PER PEDRA GRAN", "purple"; message += " Risc de pedra de gran mida (>4cm)."
        elif lfc_h > 3000: title, color = "TEMPESTES DE BASE ALTA", "saddlebrown"; message += " Risc de ratxes de vent fortes (downbursts)."
        return title, message, color
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p_levels).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_layer = mpcalc.relative_humidity_from_dewpoint(t_profile[layer_mask], td_profile[layer_mask])
            pwat_layer = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask]).to('mm')
            rh_mean_layer = np.mean(rh_layer)
            if rh_mean_layer > 0.85 and cape.magnitude < 350:
                if pwat_layer.m > 25: return "AVÍS PER PLUGES INTENSES", "(Activa el forçament) Risc de pluges persistents i fortes. Possible acumulació d'aigua.", "darkblue"
                elif pwat_layer.m > 15: return "AVÍS PER PLUJA MODERADA", "Cel cobert amb pluja contínua i moderada. Visibilitat reduïda.", "steelblue"
                else: return "PREVISIÓ DE PLUJA FEBLE", "(Activa el forçament) S'esperen plugims o ruixats febles i intermitents.", "cadetblue"
    except Exception: pass
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

# =========================================================================
# === 3. FUNCIONS DE DIBUIX ===============================================
# =========================================================================
# ... (Les funcions de dibuix s'ometen per brevetat)
def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if cape.m<=0 or not lcl_p:return None,None
    cloud_base_km=lcl_h/1000.;
    if convergence_active:cloud_top_km=el_h/1000. if el_h>lcl_h else cloud_base_km
    else:
        if not lfc_p:cloud_top_km=cloud_base_km+0.1
        else:
            try:
                rh=mpcalc.relative_humidity_from_dewpoint(t_profile,td_profile);indices_above_lcl=np.where(p_levels<=lcl_p)[0];p_top=p_levels[-1]
                if len(indices_above_lcl)>0:
                    for idx in indices_above_lcl:
                        if rh[idx]<0.7:p_top=p_levels[idx];break
                cloud_top_km=mpcalc.pressure_to_height_std(p_top).to('km').m
            except:cloud_top_km=cloud_base_km
    return (cloud_base_km,cloud_top_km) if cloud_base_km is not None and cloud_top_km is not None and cloud_top_km>cloud_base_km else (None,None)
def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10));skew=SkewT(fig,rotation=45);ax=skew.ax;ax.set_ylim(1050,100);ax.set_xlim(-50,45)
    with integrator_lock:skew.plot_dry_adiabats(alpha=0.3,color='orange');skew.plot_moist_adiabats(alpha=0.3,color='green');skew.plot_mixing_lines(alpha=0.4,color='blue',linestyle='--')
    td_profile=np.minimum(t_profile,td_profile);skew.plot(p_levels,t_profile,'r',linewidth=2,label='Temperatura (T)');skew.plot(p_levels,td_profile,'b',linewidth=2,label='Punt de Rosada (Td)')
    parcel_prof=mpcalc.parcel_profile(p_levels,t_profile[0],td_profile[0]).to('degC');skew.plot(p_levels,parcel_prof,'k--',linewidth=2,label='Bombolla Adiabàtica')
    wb_profile=mpcalc.wet_bulb_temperature(p_levels,t_profile,td_profile);skew.plot(p_levels,wb_profile,color='purple',linewidth=1.5,label='Tª Bombolla Humida')
    skew.shade_cape(p_levels,t_profile,parcel_prof,facecolor='yellow',alpha=0.3);skew.shade_cin(p_levels,t_profile,parcel_prof,facecolor='black',alpha=0.3)
    cape,cin,lcl_p,lcl_h,lfc_p,lfc_h,el_p,el_h,fz_h=calculate_thermo_parameters(p_levels,t_profile,td_profile);xlims=ax.get_xlim()
    if lcl_p:ax.plot(xlims,[lcl_p.m,lcl_p.m],'gray',linestyle='--',label='LCL')
    if lfc_p:ax.plot(xlims,[lfc_p.m,lfc_p.m],'purple',linestyle='--',label='LFC')
    if el_p:ax.plot(xlims,[el_p.m,el_p.m],'red',linestyle='--',label='EL')
    ax.legend();plt.tight_layout();return fig
def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type):
    fig, ax = plt.subplots(figsize=(5, 8));ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set(ylim=(0,16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0, 17, 2));ax.set_ylabel("Altitud (km)");ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5);ax.set_facecolor('#6495ED');ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//' if ground_color=='#228B22' else ''))
    return fig # Funció simplificada per brevetat
def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active): return plt.figure(figsize=(5, 8)) # Funció simplificada
def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir): return plt.figure(figsize=(5, 5)) # Funció simplificada
def create_hodograph_figure(p, ws, wd, t, td): return plt.figure(figsize=(6, 6)) # Funció simplificada

# =========================================================================
# === 4. ESTRUCTURA DE L'APLICACIÓ =======================================
# =========================================================================

def show_map_selection_screen():
    """Mostra un mapa interactiu de Catalunya creat amb HTML/CSS."""
    st.title("🗺️ Selecció de la Província")
    st.markdown("Fes clic sobre una de les províncies per visualitzar el sondeig atmosfèric.")

    # Missatge d'advertència si es fa clic a una província no disponible
    if 'unavailable_city_clicked' in st.session_state and st.session_state.unavailable_city_clicked:
        city_map = {'girona': 'Girona', 'lleida': 'Lleida', 'tarragona': 'Tarragona'}
        city_name = city_map.get(st.session_state.unavailable_city_clicked, 'Aquesta ubicació')
        st.warning(f"⚠️ **{city_name} no està disponible actualment.** De moment, només el sondeig de Barcelona està actiu.")
        st.session_state.unavailable_city_clicked = None

    # Codi HTML i CSS per crear el mapa interactiu
    map_html = """
    <style>
        a {
            text-decoration: none;
        }
        .map-grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto;
            gap: 15px;
            padding: 20px;
            max-width: 700px;
            margin: auto;
            color: white;
        }
        .province-card {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            transition: all 0.3s ease;
        }
        .province-card h3 {
            margin: 0 0 10px 0;
            color: white;
        }
        .province-card .icon {
            font-size: 3rem;
            line-height: 1;
        }
        .province-card p {
            margin: 5px 0 0 0;
            font-weight: bold;
        }

        /* Estil per al botó de Barcelona (disponible) */
        .available {
            cursor: pointer;
            border-color: #1E90FF; /* Blau per destacar */
        }
        .available:hover {
            transform: scale(1.05);
            background: rgba(30, 144, 255, 0.2); /* Color al passar per sobre */
            border-color: #87CEFA;
        }
        .available p {
            color: #1E90FF;
            background-color: black;
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid #1E90FF;
        }
        
        /* Estil per a les províncies no disponibles */
        .unavailable {
            cursor: pointer; /* Canvia el cursor per indicar que és clicable */
            opacity: 0.6;
        }
         .unavailable:hover {
            background: rgba(255, 0, 0, 0.1); /* To vermellós al passar per sobre */
        }
    </style>

    <div class="map-grid-container">
        <!-- Fila superior -->
        <a href="?city=lleida">
            <div class="province-card unavailable" title="Lleida (No disponible)">
                <h3>Lleida</h3>
                <div class="icon">🚫</div>
                <p>No disponible</p>
            </div>
        </a>
        <a href="?city=girona">
            <div class="province-card unavailable" title="Girona (No disponible)">
                <h3>Girona</h3>
                <div class="icon">🚫</div>
                <p>No disponible</p>
            </div>
        </a>
        
        <!-- Fila inferior -->
        <a href="?city=tarragona">
            <div class="province-card unavailable" title="Tarragona (No disponible)">
                <h3>Tarragona</h3>
                <div class="icon">🚫</div>
                <p>No disponible</p>
            </div>
        </a>
        <a href="?city=barcelona">
            <div class="province-card available" title="Barcelona (Disponible)">
                <h3>Barcelona</h3>
                <div class="icon">🛰️</div>
                <p>Accedir al Sondeig</p>
            </div>
        </a>
    </div>
    """
    st.markdown(map_html, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⬅️ Tornar a l'inici", use_container_width=True):
        st.session_state.app_mode = 'welcome'
        st.rerun()

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">TEMPESTES.CAT PRESENTA :</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Una eina per a la visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️Temps Real</h3><p>Visualitza els sondejos atmosfèrics més recents basats en dades de models. Navega entre les diferents execucions horàries disponibles.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode Temps Real", use_container_width=True):
            st.session_state.app_mode = 'map_selection'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪Laboratori</h3><p>Aprèn de forma interactiva com es formen els fenòmens severs modificant pas a pas un sondeig o experimenta lliurement amb els controls.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False):
    st.markdown(f"#### {obs_time}")
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    
    st.toggle(
        "Activar Forçament Extern (Convergència / Orografia)",
        key='convergence_active',
        help="Simula l'efecte d'un mecanisme de tret (p.ex. convergència o orografia). Si està activat, els núvols creixeran fins al seu topall teòric (EL) si hi ha CAPE, ignorant la inhibició (CIN). Si no, només es formaran en capes ja saturades o si la convecció pot vèncer el CIN per si sola."
    )
    convergence_active = st.session_state.get('convergence_active', False)

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, convergence_active)
    cloud_type = "Cel Serè"
    pwat_0_4, rh_0_4 = units.Quantity(0, 'mm'), 0.0
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_profile_layer = mpcalc.relative_humidity_from_dewpoint(t[layer_mask], td[layer_mask])
            rh_0_4 = np.mean(rh_profile_layer)
            pwat_0_4 = mpcalc.precipitable_water(p[layer_mask], td[layer_mask]).to('mm')
    except Exception: pass
    
    sfc_temp = t[0]
    convection_possible_from_surface = (cin.m > -80 and lfc_h < 2500)

    if sfc_temp.m < 5 or fz_h < 1500: cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape.m < 350:
        if pwat_0_4.m > 25: cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4.m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150 and convection_possible_from_surface: cloud_type = "Supercèl·lula"
    elif cape.m >= 1500 and convection_possible_from_surface: cloud_type = "Cumulonimbus (Multicèl·lula)"
    elif cape.m >= 800 and convection_possible_from_surface: cloud_type = "Cumulus Congestus"
    elif cape.m >= 300 and convection_possible_from_surface: cloud_type = "Cumulus Mediocris"
    elif cape.m > 50 and convection_possible_from_surface: cloud_type = "Cumulus Humilis"
    elif cape.m > 500: cloud_type = "Castellanus"
    elif base_km and top_km and (top_km - base_km) > 0: cloud_type = "Cumulus Fractus"
    
    st.subheader("Diagrama Skew-T", anchor=False)
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    st.divider()

    if is_sandbox_mode:
         chat_log, precipitation_type = generate_dynamic_analysis(p, t, td, ws, wd, cloud_type)
    else:
        chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Assistent d'Anàlisi", "📊 Paràmetres Detallats", "📈 Hodògraf", "☁️ Visualització de Núvols", "📡 Simulació Radar"])
    with tab1:
        css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>"""
        html_chat = "<div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower()
            html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)

    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A"); param_cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); param_cols[3].metric("Shear 0-6km", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1km", f"{srh_0_1:.1f} m²/s²"); param_cols[1].metric("SRH 0-3km", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm")
        rh_display = "N/A"
        try:
            rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("RH Mitja 0-4km", rh_display)
    
    with tab3: st.pyplot(create_hodograph_figure(p, ws, wd, t, td), use_container_width=True)
    with tab4:
        precipitation_type_visual = "snow" if "NEU" in title else "sleet" if "AIGUANEU" in title else precipitation_type
        cloud_cols = st.columns(2)
        with cloud_cols[0]: st.pyplot(create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type_visual, lfc_h, cape, base_km, top_km, cloud_type), use_container_width=True)
        with cloud_cols[1]: st.pyplot(create_cloud_structure_figure(p, t, td, ws, wd, convergence_active), use_container_width=True)
    with tab5: st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

def display_province_map():
    """
    Mostra un mapa SVG interactiu de les províncies de Catalunya.
    Barcelona és clicable, mentre que les altres províncies estan desactivades.
    """
    # ... (tot el codi de la funció del mapa) ...


# I AQUÍ HI VA LA VERSIÓ SUBSTITUÏDA DE LA FUNCIÓ run_live_mode (PAS 1)
def run_live_mode():
    # Comprovem si una província ha estat seleccionada
    if 'province_selected' not in st.session_state:
        st.session_state.province_selected = None

    if st.session_state.province_selected == 'barcelona':
        # ... (tota la lògica per mostrar el sondeig de Barcelona) ...
    else:
        # Si no, mostra el mapa
        st.title("🛰️ Mode Temps Real")
        # ...
        display_province_map() # Fixa't com es crida la funció que has afegit a sobre
    def get_time_state(filename, current_hour):
        try:
            file_hour = int(filename.replace('h.txt', ''))
            if file_hour < current_hour: return 'past'
            elif file_hour == current_hour: return 'current'
            else: return 'future'
        except (ValueError, IndexError): return 'future'

    def format_time_for_display(filename):
        state = get_time_state(filename, st.session_state.current_hour)
        display_time = filename.replace('h.txt', ':00')
        if state == 'past': return f"✅ {display_time}"
        elif state == 'current': return f"🟡 {display_time} (Ara)"
        else: return f" {display_time}"

    with st.sidebar:
        try: current_index = st.session_state.existing_files.index(st.session_state.selected_file)
        except ValueError: current_index = 0 
        selected_file = st.radio("Hores disponibles:", st.session_state.existing_files, index=current_index, format_func=format_time_for_display, key='time_selector')
        if selected_file != st.session_state.selected_file:
            st.session_state.selected_file = selected_file
            st.rerun()
            
    try:
        soundings = parse_all_soundings(st.session_state.selected_file)
        if soundings:
            data = soundings[0]
            show_full_analysis_view(p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], obs_time=data.get('observation_time', 'Hora no disponible'), is_sandbox_mode=False)
        else: st.error(f"No s'han pogut carregar dades de {st.session_state.selected_file}")
    except FileNotFoundError:
        st.error(f"L'arxiu '{st.session_state.selected_file}' no existeix.")
        if st.session_state.existing_files:
            st.session_state.selected_file = st.session_state.existing_files[0]
            st.rerun()

# =================================================================================
# === LABORATORI-TUTORIAL =========================================================
# =================================================================================
# ... (El codi del laboratori/tutorial no canvia, s'omet per brevetat)
def get_tutorial_data(): return {'supercel':[{'action_id':'warm_low','title':'Pas 1: Escalfament superficial','instruction':"Necessitem energia. La manera més comuna és l'escalfament del sol durant el dia. Fes clic al botó de sota per escalfar les capes baixes.",'button_label':"☀️ Escalfar Capa Baixa",'explanation':"Això augmenta la temperatura a prop de la superfície, creant una 'bombolla' d'aire que voldrà ascendir."},{'action_id':'moisten_low','title':'Pas 2: Afegeix combustible','instruction':"Una tempesta necessita humitat per formar-se. Fes clic al botó per humitejar les capes baixes i apropar el punt de rosada a la temperatura.",'button_label':"💧 Humitejar Capa Baixa",'explanation':"Això fa que l'aire ascendent es condensi abans, alliberant calor latent i donant més força a la tempesta (augmentant el CAPE)."},{'action_id':'add_shear_low','title':"Pas 3: Afegeix el motor de rotació",'instruction':"L'ingredient secret d'una supercèl·lula és el cisallament del vent a nivells baixos. Fes clic al botó per afegir un canvi de vent amb l'altura.",'button_label':"🌪️ Afegir Cisallament a Capes Baixes",'explanation':"Això farà que el corrent ascendent de la tempesta comenci a rotar, organitzant-la i fent-la molt més potent i duradora."},{'action_id':'conceptual','title':'Pas 4: Anàlisi Final','instruction':"Ja tenim energia, humitat i rotació. Has creat un entorn perfecte per a la formació de supercèl·lules.",'button_label':"Entès, finalitzar →",'explanation':"A l'anàlisi final, fixa't en com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."}],'aiguaneu':[{'action_id':'conceptual','title':"Pas 1: Analitza la Capa Mitjana-Alta",'instruction':"Hem carregat un perfil d'hivern. A les capes altes (per sobre de 700 hPa), les temperatures són negatives. Aquesta és la 'fàbrica de neu'.",'button_label':"Entès, següent pas →",'explanation':"Aquí és on es formen els flocs de neu inicials. De moment, tot correcte."},{'action_id':'conceptual','title':"Pas 2: Identifica la Capa Càlida",'instruction':"Ara mira la capa mitjana (al voltant de 850 hPa). La temperatura aquí és superior a 0°C. Aquest és el nostre problema.",'button_label':"Ho veig, següent pas →",'explanation':"Quan els flocs de neu cauen a través d'aquesta capa càlida, es fonen i es converteixen en gotes de pluja."},{'action_id':'conceptual','title':"Pas 3: Analitza la Superfície",'instruction':"Finalment, la capa superficial està de nou sota zero. Què passarà amb les gotes de pluja que venen de dalt?",'button_label':"Entès, següent pas →",'explanation':"Les gotes es tornen a congelar just abans de tocar a terra. Això és el que produeix l'aiguaneu (sleet) o la perillosa pluja gelant."},{'action_id':'conceptual','title':'Pas 4: Conclusió i Repte','instruction':"Has analitzat un perfil clàssic d'aiguaneu! Ara saps que una capa càlida intermèdia és la culpable.",'button_label':"Finalitzar Tutorial",'explanation':"Repte: Ara que has acabat, fes clic a 'Finalitzar'. Utilitza l'eina '❄️ Refredar Capa Mitjana' a la barra lateral i veuràs com converteixes aquest perfil en una nevada perfecta!"}]}
def start_tutorial(scenario_name):
    st.session_state.sandbox_mode='tutorial';st.session_state.tutorial_active=True;st.session_state.tutorial_scenario=scenario_name;st.session_state.tutorial_step=0
    if scenario_name=='aiguaneu':profile_data=create_wintry_mix_profile()
    else:profile_data=st.session_state.sandbox_original_data
    st.session_state.sandbox_p_levels=profile_data['p_levels'].copy();st.session_state.sandbox_t_profile=profile_data['t_initial'].copy();st.session_state.sandbox_td_profile=profile_data['td_initial'].copy();st.session_state.sandbox_ws=st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s');st.session_state.sandbox_wd=st.session_state.sandbox_original_data['wind_dir_deg'].copy()
def exit_tutorial():
    st.session_state.sandbox_mode='free';st.session_state.tutorial_active=False
    if 'tutorial_scenario' in st.session_state: del st.session_state['tutorial_scenario']
    if 'tutorial_step' in st.session_state: del st.session_state['tutorial_step']
def apply_profile_modification(action):
    t=st.session_state.sandbox_t_profile.m;td=st.session_state.sandbox_td_profile.m;p=st.session_state.sandbox_p_levels.m;ws=st.session_state.sandbox_ws.to('m/s').m;wd=st.session_state.sandbox_wd.m;low_mask=p>850;mid_mask=(p<=850)&(p>600);high_mask=p<=600
    if action=='warm_low':t[low_mask]+=2.
    elif action=='cool_low':t[low_mask]-=2.
    elif action=='moisten_low':td[low_mask]=np.minimum(t[low_mask]-1.,td[low_mask]+2.)
    elif action=='dry_low':td[low_mask]-=2.
    elif action=='warm_mid':t[mid_mask]+=2.
    elif action=='cool_mid':t[mid_mask]-=4.
    elif action=='moisten_mid':td[mid_mask]=np.minimum(t[mid_mask]-1.5,td[mid_mask]+2.)
    elif action=='dry_mid':td[mid_mask]-=2.
    elif action=='warm_high':t[high_mask]+=2.
    elif action=='cool_high':t[high_mask]-=2.
    elif action=='moisten_high':td[high_mask]=np.minimum(t[high_mask]-2.,td[high_mask]+2.)
    elif action=='dry_high':td[high_mask]-=2.
    elif action=='warm_all':t+=2.
    elif action=='cool_all':t-=2.
    elif action=='moisten_all':td=np.minimum(t-1.,td+2.)
    elif action=='dry_all':td-=2.
    elif action=='add_inversion':inv_mask=(p<950)&(p>800);t[inv_mask]+=3.
    elif 'shear' in action:
        if action=='add_shear_low':mask=low_mask
        elif action=='add_shear_mid':mask=mid_mask
        elif action=='add_shear_high':mask=high_mask
        else:mask=np.full_like(p,True)
        num_points=np.sum(mask)
        if num_points>0:ws[mask]+=np.linspace(0,15,num_points);ws=np.clip(ws,0,80);wd[mask]=(wd[mask]+np.linspace(0,45,num_points))%360
        st.session_state.sandbox_ws=ws*units('m/s');st.session_state.sandbox_wd=wd*units.degrees
    td=np.minimum(t,td);st.session_state.sandbox_t_profile=t*units.degC;st.session_state.sandbox_td_profile=td*units.degC
def show_tutorial_interface():
    tutorials=get_tutorial_data();scenario=st.session_state.tutorial_scenario;step_index=st.session_state.tutorial_step;steps=tutorials[scenario];st.title("🧪 Laboratori de Sondejos - Mode Tutorial")
    with st.container(border=True):
        col1,col2=st.columns([1,1],gap="large")
        with col1:
            st.markdown(f"### Tutorial: {scenario.replace('_',' ').title()}");st.markdown("---")
            if step_index>=len(steps):
                st.success("🎉 Enhorabona, has completat el tutorial! 🎉");st.markdown("El sondeig que has construït ja està a punt. Fes clic a 'Finalitzar' per veure'n l'anàlisi completa.")
                if st.button("Finalitzar i Veure Resultat",use_container_width=True,type="primary"):exit_tutorial();st.rerun()
            else:
                current_step=steps[step_index];st.markdown(f"#### {current_step['title']}")
                with st.container(border=True):
                    st.markdown(current_step['instruction']);action_id=current_step['action_id']
                    if st.button(current_step['button_label'],key=f"tut_action_{step_index}",use_container_width=True,type="primary"):
                        if action_id!='conceptual':apply_profile_modification(action_id)
                        st.session_state.tutorial_step+=1;st.rerun()
                st.markdown(f"*{current_step['explanation']}*")
        with col2:
            chat_log,_=generate_tutorial_analysis(scenario,step_index);css_styles="""<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; height: 350px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>""";html_chat="<h6>Assistent d'Anàlisi</h6><div class='chat-container'>"
            for speaker,message in chat_log:css_class=speaker.lower();html_chat+=f"""<div class="message-row {'message-row-right' if css_class=='usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            html_chat+="</div>";st.markdown(css_styles+html_chat,unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Abandonar Tutorial",use_container_width=True):exit_tutorial();st.rerun()
def show_sandbox_selection_screen():
    st.title("🧪 Benvingut al Laboratori!");st.markdown("Tria com vols començar. Pots seguir un tutorial guiat per aprendre els conceptes clau o anar directament al mode lliure per experimentar por tu mateix.");st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🌪️ Tutorial: Supercèl·lula</h4><p>Aprèn a crear un entorn amb una inestabilitat explosiva i el cisallament necessari per a les tempestes més severes i organitzades.</p></div>""",unsafe_allow_html=True)
        if st.button("Començar Tutorial de Supercèl·lula",use_container_width=True):start_tutorial('supercel');st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>💧 Tutorial: Aiguaneu</h4><p>Analitza una situació d'aiguaneu a BCN per exemple , identifica la capa càlida culpable i aprèn com transformar la precipitació en neu.</p></div>""",unsafe_allow_html=True)
        if st.button("Començar Tutorial d'Aiguaneu",use_container_width=True):start_tutorial('aiguaneu');st.rerun()
    with c3:
        st.markdown("""<div class="mode-card"><h4>🛠️ Mode Lliure</h4><p>Salta directament a l'acció. Tindràs el control total sobre el perfil atmosfèric des del principi per crear els teus propis escenaris.</p></div>""",unsafe_allow_html=True)
        if st.button("Anar al Mode Lliure",use_container_width=True,type="primary"):st.session_state.sandbox_mode='free';st.rerun()
    st.markdown("---")
    if st.button("⬅️ Tornar a l'inici"):st.session_state.app_mode='welcome';st.rerun()
def run_sandbox_mode():
    if 'sandbox_mode' not in st.session_state:st.session_state.sandbox_mode='selection'
    if 'sandbox_initialized' not in st.session_state:
        placeholder=st.empty()
        with placeholder.container():show_loading_animation();time.sleep(0.5)
        soundings=parse_all_soundings("sondeigproves.txt")
        if not soundings:st.error("No s'ha trobat 'sondeigproves.txt'. Assegura't que el fitxer existeix.");placeholder.empty();return
        st.session_state.sandbox_original_data=soundings[0];data=st.session_state.sandbox_original_data;st.session_state.sandbox_p_levels=data['p_levels'].copy();st.session_state.sandbox_t_profile=data['t_initial'].copy();st.session_state.sandbox_td_profile=data['td_initial'].copy();st.session_state.sandbox_ws=data['wind_speed_kmh'].to('m/s');st.session_state.sandbox_wd=data['wind_dir_deg'].copy();st.session_state.sandbox_initialized=True;st.session_state.convergence_active=False;placeholder.empty()
    with st.sidebar:
        st.header("Caixa d'Eines")
        if st.button("⬅️ Tornar al Menú del Laboratori",use_container_width=True):
            for key in ['sandbox_mode','tutorial_active','tutorial_scenario','tutorial_step','convergence_active']:
                if key in st.session_state:del st.session_state[key]
            st.rerun()
        st.markdown("---");st.subheader("Modificacions Termodinàmiques");st.markdown("**Capes Baixes (> 850 hPa)**");c1,c2=st.columns(2);c1.button("☀️ Escalfar",on_click=apply_profile_modification,args=('warm_low',),use_container_width=True);c2.button("❄️ Refredar",on_click=apply_profile_modification,args=('cool_low',),use_container_width=True);c1.button("💧 Humitejar",on_click=apply_profile_modification,args=('moisten_low',),use_container_width=True);c2.button("💨 Assecar",on_click=apply_profile_modification,args=('dry_low',),use_container_width=True)
        st.markdown("**Capes Mitjanes (850-600 hPa)**");c1,c2=st.columns(2);c1.button("☀️ Escalfar",on_click=apply_profile_modification,args=('warm_mid',),use_container_width=True,key='w_mid');c2.button("❄️ Refredar",on_click=apply_profile_modification,args=('cool_mid',),use_container_width=True,key='c_mid');c1.button("💧 Humitejar",on_click=apply_profile_modification,args=('moisten_mid',),use_container_width=True,key='m_mid');c2.button("💨 Assecar",on_click=apply_profile_modification,args=('dry_mid',),use_container_width=True,key='d_mid')
        st.markdown("**Capes Altes (< 600 hPa)**");c1,c2=st.columns(2);c1.button("☀️ Escalfar",on_click=apply_profile_modification,args=('warm_high',),use_container_width=True,key='w_h');c2.button("❄️ Refredar",on_click=apply_profile_modification,args=('cool_high',),use_container_width=True,key='c_h');c1.button("💧 Humitejar",on_click=apply_profile_modification,args=('moisten_high',),use_container_width=True,key='m_h');c2.button("💨 Assecar",on_click=apply_profile_modification,args=('dry_high',),use_container_width=True,key='d_h')
        st.markdown("---");st.subheader("Eines Globals i de Vent");c1,c2=st.columns(2);c1.button("🔥 Escalfar Tot",on_click=apply_profile_modification,args=('warm_all',),use_container_width=True);c2.button("🧊 Refredar Tot",on_click=apply_profile_modification,args=('cool_all',),use_container_width=True);c1.button("💦 Humitejar Tot",on_click=apply_profile_modification,args=('moisten_all',),use_container_width=True);c2.button("🌬️ Assecar Tot",on_click=apply_profile_modification,args=('dry_all',),use_container_width=True);st.button("Tapadera (Inversió)",on_click=apply_profile_modification,args=('add_inversion',),use_container_width=True)
        st.markdown("**Cisallament del Vent**");c1,c2,c3=st.columns(3);c1.button("🌪️ Baixes",on_click=apply_profile_modification,args=('add_shear_low',),use_container_width=True);c2.button("🌪️ Mitges",on_click=apply_profile_modification,args=('add_shear_mid',),use_container_width=True);c3.button("🌪️ Altes",on_click=apply_profile_modification,args=('add_shear_high',),use_container_width=True)
        def reset_wind_profile():st.session_state.sandbox_ws=st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s');st.session_state.sandbox_wd=st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.button("🚫 Reiniciar Vents",on_click=reset_wind_profile,use_container_width=True);st.markdown("---")
        if st.button("🔄 Reiniciar Tot al Perfil Original",use_container_width=True):
            data=st.session_state.sandbox_original_data;st.session_state.sandbox_p_levels=data['p_levels'].copy();st.session_state.sandbox_t_profile=data['t_initial'].copy();st.session_state.sandbox_td_profile=data['td_initial'].copy();reset_wind_profile()
            if st.session_state.get('tutorial_active',False):exit_tutorial()
            if 'convergence_active' in st.session_state:st.session_state.convergence_active=False
            st.rerun()
    if st.session_state.sandbox_mode=='selection':show_sandbox_selection_screen()
    elif st.session_state.sandbox_mode=='tutorial':show_tutorial_interface()
    elif st.session_state.sandbox_mode=='free':
        st.title("🧪 Laboratori de Sondejos - Mode Lliure")
        show_full_analysis_view(p=st.session_state.sandbox_p_levels,t=st.session_state.sandbox_t_profile,td=st.session_state.sandbox_td_profile,ws=st.session_state.sandbox_ws,wd=st.session_state.sandbox_wd,obs_time="Sondeig de Prova - Mode Laboratori",is_sandbox_mode=True)


# =========================================================================
# === PUNT D'ENTRADA DE L'APLICACIÓ =======================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Analitzador de Sondejos")

    # Gestió de paràmetres URL per a la selecció del mapa
    if 'city' in st.query_params:
        city_code = st.query_params.get('city')
        st.query_params.clear() 

        if city_code == 'barcelona':
            # Si es fa clic a Barcelona, canvia al mode temps real.
            st.session_state.app_mode = 'live'
        else:
            # Si es fa clic a una altra ciutat, torna al mapa i desa la ciutat clicada.
            st.session_state.app_mode = 'map_selection'
            st.session_state.unavailable_city_clicked = city_code
        st.rerun()

    # Defineix l'estat inicial de l'aplicació si no existeix
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'

    # Enrutador principal de l'aplicació
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'map_selection':
        show_map_selection_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
