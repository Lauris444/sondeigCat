# =========================================================================
# === ANALITZADOR DE SONDEJOS ATMOSFÈRICS v2.0 ============================
# === Autor: TEMPESTES.CAT / Desenvolupament assistit per IA =============
# =========================================================================

# --- 1. IMPORTACIONS DE LLIBRERIES ---------------------------------------
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
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

# Noves llibreries per a la càrrega de dades web
import requests
from bs4 import BeautifulSoup

# --- 2. CONFIGURACIÓ GLOBAL ----------------------------------------------
# Pany de concurrència per a càlculs intensius de MetPy
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

def show_loading_animation(message="Carregant"):
    """Mostra una animació de càrrega personalitzada amb HTML i CSS."""
    loading_html = f"""
    <style>
        .loading-container {{
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            background: rgba(25,37,81,0.9); z-index: 9999;
        }}
        .loading-svg {{ width: 150px; height: auto; margin-bottom: 20px; }}
        .loading-text {{ color: white; font-size: 1.5rem; font-family: sans-serif; }}
        .loading-text .dot {{ animation: blink 1.4s infinite both; }}
        .loading-text .dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .loading-text .dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{ 0%, 80%, 100% {{ opacity: 0; }} 40% {{ opacity: 1; }} }}
    </style>
    <div class="loading-container">
        <svg class="loading-svg" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
            <path d="M 155.6,66.1 C 155.6,42.9 135.5,23.5 111.4,23.5 C 98.4,23.5 86.8,29.4 79.1,38.7 C 75.2,16.8 57.3,0 36.4,0 C 16.3,0 0,16.3 0,36.4 C 0,56.5 16.3,72.8 36.4,72.8 L 110,72.8 C 110,72.8 110,72.8 110,72.8 C 135,72.8 155.6,93.4 155.6,118.4 C 155.6,143.4 135,164 110,164 L 50, 164" fill="none" stroke="#FFFFFF" stroke-width="8"/>
            <polygon points="120,60 90,110 115,110 100,150 145,90 120,90 130,60" fill="#FFD700" />
        </svg>
        <div class="loading-text">{message}<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

def set_main_background():
    """Estableix el fons fosc estàndard per a les pantalles d'anàlisi."""
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

def fetch_sounding_from_url(lat, lon, forecast_hour):
    """
    Obté i processa un sondeig de Meteociel per a una ubicació i hora de pronòstic.
    """
    url = f"https://www.meteociel.fr/modeles/sondage2arome.php?archive=0&ech={forecast_hour}&map=8&wrf=0&y1={lat}&x1={lon}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        pre_tag = soup.find('pre', class_='pre')
        if not pre_tag:
            st.warning(f"No s'ha pogut trobar el bloc de dades del sondeig a la URL per a +{forecast_hour}h.")
            return None
        sounding_text = pre_tag.get_text()
        sounding_lines = sounding_text.splitlines()
        processed_data = process_sounding_block(sounding_lines)
        if processed_data:
            model_run_info = soup.find('h3').get_text(strip=True) if soup.find('h3') else f"Model AROME {forecast_hour:02d}Z"
            processed_data['observation_time'] = f"{model_run_info}\nPronòstic: +{forecast_hour}h"
        return processed_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error de xarxa en intentar obtenir el sondeig: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperat processant el sondeig de la URL: {e}")
        return None

def get_image_as_base64(file_path):
    """Llegeix una imatge i la converteix a format Base64 per a HTML."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        return None

def clean_and_convert(text):
    """Neteja i converteix text a un float."""
    cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
    if not cleaned_text or cleaned_text == '-': return None
    try: return float(cleaned_text)
    except ValueError: return None

def process_sounding_block(block_lines):
    """Processa un bloc de text de sondeig i el converteix en dades estructurades."""
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
    """Llegeix un fitxer de text local amb múltiples sondejos."""
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'. Aquest fitxer és necessari per al Mode Laboratori.")
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
    """Genera un perfil de dades sintètic per a un escenari d'aiguaneu."""
    p = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
    t = np.array([1.5, 3.0, 1.0, -5.0, -20.0, -45.0, -60.0]) * units.degC
    td = np.array([0.5, 1.0, -1.0, -6.0, -22.0, -48.0, -65.0]) * units.degC
    ws = np.full_like(p.magnitude, 15) * units.knots
    wd = np.full_like(p.magnitude, 180) * units.degrees
    return {'p_levels': p, 't_initial': t, 'td_initial': td, 'wind_speed_kmh': ws.to('kph'), 'wind_dir_deg': wd}

# =========================================================================
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
# =========================================================================

def calculate_thermo_parameters(p_levels, t_profile, td_profile):
    """Calcula paràmetres termodinàmics clau d'un sondeig."""
    with integrator_lock:
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
    """Calcula paràmetres de cisallament i helicitat."""
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
        
        with integrator_lock:
            srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter)[0].m
            srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter)[0].m
        
        return s_0_6, s_0_1, srh_0_1, srh_0_3
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, base_km, top_km, pwat_0_4):
    """Genera l'anàlisi conversacional per al mode 'Live', amb més diàleg i per a totes les condicions."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, _, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    chat_log = []

    if t_profile[0].m < 7.0:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'rain'
        chat_log.append(("Sistema", "Iniciant anàlisi de perfil hivernal (T < 7°C)."))
        chat_log.append(("Analista", f"D'acord, tenim una temperatura en superfície de {t_profile[0].m:.1f}°C. Això canvia les regles del joc. Ja no busquem tempestes, sinó que analitzem el potencial de neu."))
        chat_log.append(("Usuari", "Perfecte. Quins són els factors decisius per veure neu?"))
        p_array, t_array = p_levels.m, t_profile.m
        warm_layer_mask = (p_array < p_array[0]) & (p_array > 700) & (t_array > 0.5)
        if not np.any(warm_layer_mask):
            chat_log.append(("Analista", "Bones notícies per als amants del fred. He revisat tota la columna d'aire i sembla que es manté per sota o molt a prop de 0°C en tot el recorregut."))
            if t_profile[0].m > 1.5:
                chat_log.append(("Usuari", f"Llavors, tot i que no hi ha capes càlides, la temperatura a la superfície ({t_profile[0].m:.1f}°C) no és massa alta?"))
                chat_log.append(("Analista", f"Bona observació. Una T de {t_profile[0].m:.1f}°C pot fer que els flocs es fonguin just en arribar o donin una neu molt humida. La cota estaria just per sobre de la nostra ubicació."))
                precipitation_type = 'rain'
            else:
                chat_log.append(("Usuari", "Això vol dir que si precipita, serà en forma de neu?"))
                chat_log.append(("Analista", "Exacte. Aquest és un 'perfil de nevada'. Si hi ha precipitació, serà neu."))
                precipitation_type = 'snow'
        else:
            max_temp_in_layer = np.max(t_array[warm_layer_mask])
            chat_log.append(("Analista", f"Alerta! He detectat una 'capa càlida' en altura. La temperatura puja fins a {max_temp_in_layer:.1f}°C."))
            chat_log.append(("Usuari", "I això què significa? Adéu a la neu?"))
            if t_profile[0].m <= 0.5:
                chat_log.append(("Analista", "Aquesta capa fon els flocs, però com que a prop del terra fa fred, es tornen a congelar. El resultat més probable és **aiguaneu (sleet)** o la perillosa **pluja gelant**."))
                precipitation_type = 'sleet'
            else:
                 chat_log.append(("Analista", "Exactament. Aquesta capa fon la neu i, com que la temperatura en superfície és positiva, arribarà com a pluja freda."))
                 precipitation_type = 'rain'
    else:
        if "Tornàdica" in cloud_type or "Tuba" in cloud_type or "Mur" in cloud_type: precipitation_type = 'hail'
        elif cape.m > 500: precipitation_type = 'rain'
        elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
        chat_log.append(("Sistema", f"Iniciant anàlisi conversacional per a l'escenari de {cloud_type}."))
        
        # Lògica conversacional per a cada tipus de núvol
        if "Tornàdica" in cloud_type:
            chat_log.extend([("Analista", "ALERTA MÀXIMA. El perfil presenta característiques tornàdiques clàssiques."), ("Usuari", "Què ho fa tan perillós?"), ("Analista", f"Base del núvol molt baixa ({lcl_h:.0f} m) i rotació extrema a nivells baixos (SRH 0-1km: {srh_0_1:.0f} m²/s²). Alt risc que la rotació arribi a terra.")])
        elif "Tuba/Funnel" in cloud_type:
            chat_log.extend([("Analista", "Molt de compte. Perfil de supercèl·lula amb alt potencial per a embuts."), ("Usuari", "Què indica aquest potencial?"), ("Analista", f"Combinació d'helicitat (SRH 0-1km: {srh_0_1:.0f} m²/s²) i base de núvol baixa ({lcl_h:.0f} m). La rotació té facilitat per baixar i condensar-se.")])
        elif "Mur de núvols" in cloud_type:
            chat_log.extend([("Analista", "Aquest és un perfil de supercèl·lula clàssic i organitzat."), ("Usuari", "Què el fa especial?"), ("Analista", f"La rotació a nivells mitjans és molt intensa (SRH 0-3km: {srh_0_3:.0f} m²/s²), cosa que pot formar un mur de núvols, la regió principal des d'on s'originen els tornados.")])
        elif "Shelf Cloud" in cloud_type:
             chat_log.extend([("Analista", "Atenció. Aquest perfil és perillós per vents lineals forts."), ("Usuari", "No és una supercèl·lula?"), ("Analista", f"No exactament. Tenim molta energia (CAPE: {cape.m:.0f} J/kg) però poca rotació. Això afavoreix un 'reventón' (downburst) que en arribar a terra crea un núvol de prestatge.")])
        elif "Cumulonimbus" in cloud_type:
            chat_log.extend([("Analista", "Bé, tenim un escenari amb potencial de tempestes."), ("Usuari", f"El CAPE és de {cape.m:.0f} J/kg."), ("Analista", f"És un bon valor, suficient per a tempestes fortes, possiblement amb calamarsa."), ("Usuari", "I com és el cisallament?"), ("Analista", f"El cisallament de {shear_0_6:.1f} m/s és moderat. Permet que les tempestes s'organitzin en sistemes multicel·lulars duradors.")])
        elif "Nimbostratus" in cloud_type:
            chat_log.extend([("Analista", "Aquest perfil no va d'inestabilitat explosiva."), ("Usuari", f"És cert, el CAPE és gairebé inexistent."), ("Analista", f"Exacte. El protagonista és la humitat. Tenim una capa molt gruixuda i saturada. Espera pluges persistents i generalitzades, amb {pwat_0_4.m:.1f} mm d'aigua precipitable.")])
        else:
            chat_log.extend([("Analista", "El perfil atmosfèric és estable."), ("Usuari", "Llavors, no veurem cap núvol significatiu?"), ("Analista", f"És poc probable. Amb un CAPE de només {cape.m:.0f} J/kg, no hi ha energia per al creixement vertical. Cel serè o amb núvols residuals.")])
    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type):
    """Genera anàlisi conversacional per al mode laboratori, amb més diàleg."""
    cape, cin, _, lcl_h, _, lfc_h, _, _, _ = calculate_thermo_parameters(p, t, td)
    shear_0_6, _, _, _ = calculate_storm_parameters(p, ws, wd)
    chat_log = [("Analista", "Molt bé, anem a analitzar el perfil que has creat pas a pas.")]
    if cape.m < 50:
        chat_log.extend([("Usuari", "Tenim potencial per a tempestes?"), ("Analista", f"Ara mateix no. El CAPE és de només {cape.m:.0f} J/kg. L'atmosfera està molt estable.")])
    else:
        chat_log.extend([("Usuari", "Què estic creant amb aquesta energia?"), ("Analista", f"Has generat un CAPE de {cape.m:.0f} J/kg. Això és un escenari típic per a la formació de {cloud_type}.")])
        chat_log.append(("Usuari", "I la 'tapadera' (CIN)? Com afecta?"))
        if cin.m < -50: chat_log.append(("Analista", f"Molt forta ({cin.m:.0f} J/kg). La convecció des de superfície és gairebé impossible."))
        else: chat_log.append(("Analista", f"És feble ({cin.m:.0f} J/kg). La convecció té gairebé via lliure per iniciar-se."))
        if cin.m > -100 and cape.m > 800:
            chat_log.append(("Usuari", "He modificat el vent. Com afecta?"))
            if shear_0_6 > 18: chat_log.append(("Analista", f"El cisallament és fort ({shear_0_6:.1f} m/s). Pot fer que les tempestes rotin (supercèl·lules)."))
            else: chat_log.append(("Analista", f"El cisallament és feble ({shear_0_6:.1f} m/s). Les tempestes seran més desorganitzades."))
    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Genera l'anàlisi del xat per a un pas específic d'un tutorial."""
    chat_log = []
    if scenario == 'aiguaneu':
        if step == 0: chat_log.extend([("Analista", "Benvingut! Anem a analitzar un perfil clàssic d'aiguaneu."), ("Usuari", "Perfecte. Què he de mirar?"), ("Analista", "Observa la 'fàbrica de neu' a les capes altes. Per sobre de 700 hPa fa prou fred per formar flocs.")])
        elif step == 1: chat_log.extend([("Analista", "Ara ve la part clau. Fixa't en la capa al voltant de 850 hPa. La temperatura puja per sobre dels 0°C."), ("Usuari", "Això és la 'capa càlida'?"), ("Analista", "Exacte. Aquesta capa fon els flocs i els converteix en pluja.")])
        elif step == 2: chat_log.extend([("Analista", "Finalment, mira la temperatura a prop del terra..."), ("Usuari", "Torna a estar per sota de 0°C!"), ("Analista", "Precisament! Les gotes es tornen a congelar, convertint-se en aiguaneu (sleet).")])
        else: chat_log.extend([("Analista", "Has analitzat el perfil a la perfecció! Ara, quan acabis, prova de refredar la capa mitjana al Mode Lliure i crea una nevada.")])
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem! El primer pas és sempre crear energia. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "Correcte! Ara, afegim el combustible: la humitat. Veuràs com augmenta el valor de CAPE."))
        elif step == 2: chat_log.append(("Analista", "Fantàstic! Has afegit cisallament. Aquest és l'ingredient secret que fa que les tempestes rotin."))
        else: chat_log.append(("Analista", "Missió complerta! Has creat un perfil amb energia (CAPE), humitat i cisallament (Shear/SRH)."))
    return chat_log, None
    
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """Genera un títol d'avís meteorològic basat en els paràmetres del sondeig."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    if sfc_temp.m < 7.0: # Anàlisi hivernal
        if sfc_temp.m <= 0.5:
            warm_layer_mask = (p_levels.m < 950) & (p_levels.m > 600) & (t_profile.m > 0.5)
            if np.any(warm_layer_mask): return "AIGUANEU O PLUJA GEBRADORA", "Risc d'aiguaneu o pluja gelant.", "mediumorchid"
            else: return "AVÍS PER NEU", "Perfil favorable a nevades a cotes baixes.", "navy"
        else: return "AMBIENT FRED I HUMIT", "Precipitació en forma de pluja o neu molt humida.", "steelblue"

    if cape.m >= 1200:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        if cin.m <= -100: return "CONVECCIÓ FORTAMENT INHIBIDA", f"Potencial energètic (CAPE {cape.m:.0f} J/kg) bloquejat per una forta tapadera (CIN {cin.m:.0f} J/kg).", "darkslategray"
        
        message = f"CAPE: {cape.m:.0f} J/kg. "
        if srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18:
            return "AVÍS PER TORNADO", message + f"Alt risc de tornados (SRH 0-1km: {srh_0_1:.0f}, LCL: {lcl_h:.0f}m).", "darkred"
        elif srh_0_3 > 250 and shear_0_6 > 18:
            return "AVÍS PER TEMPS SEVER", message + f"Supercèl·lules probables. Risc de calamarsa gran.", "purple"
        elif cape.m > 1500 and shear_0_6 > 12:
            return "AVÍS PER VENTS FORTS", message + "Risc de ratxes de vent severes (downbursts).", "saddlebrown"
        return "AVÍS PER TEMPESTES SEVERES", message, "darkorange"

    try:
        heights_agl = (mpcalc.pressure_to_height_std(p_levels) - mpcalc.pressure_to_height_std(p_levels[0])).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_mean_layer = np.mean(mpcalc.relative_humidity_from_dewpoint(t_profile[layer_mask], td_profile[layer_mask]))
            if rh_mean_layer > 0.85 and cape.magnitude < 350:
                pwat_layer = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask])
                if pwat_layer.to('mm').m > 25: return "AVÍS PER PLUGES INTENSES", "Risc de pluges persistents i fortes.", "darkblue"
    except: pass

    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

def determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p):
    """Determina una llista de possibles tipus de núvols basant-se en les condicions del sondeig."""
    potential_clouds = []
    try:
        if len(p) < 2: return ["Dades insuficients"]
        heights, rh = mpcalc.pressure_to_height_std(p).to('m'), mpcalc.relative_humidity_from_dewpoint(t, td)
        t_interp_func = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")
        
        mask_high = (heights.m > 6000) & (heights.m < 18000)
        if np.any(mask_high) and np.mean(rh[mask_high]) > 0.75: potential_clouds.append("Cirrus")
        mask_mid = (heights.m > 2000) & (heights.m < 7000)
        if np.any(mask_mid) and np.mean(rh[mask_mid]) > 0.90 and cape.m < 50: potential_clouds.append("Altostratus")
        
        mask_nimbostratus = (heights.m > 500) & (heights.m < 5000)
        if np.any(mask_nimbostratus) and np.mean(rh[mask_nimbostratus]) > 0.95 and cape.m <= 100:
            potential_clouds.append("Nimbostratus")
            
        if cape.m > 100 and lcl_h is not None and lcl_h > 0:
            if lfc_h is not None and lfc_h < 3000:
                if (100 < cape.m < 2500) and (lfc_h > lcl_h): potential_clouds.append("Cumulus (diversos tipus)")
                if cape.m > 1000:
                    t_at_el = t_interp_func(el_p.m) if el_p else -10
                    if t_at_el < 0: potential_clouds.append("Cumulonimbus")
            else:
                mask_mid_castellanus = (heights.m > 2000) & (heights.m < 7000)
                if np.any(mask_mid_castellanus) and np.mean(rh[mask_mid_castellanus]) > 0.60:
                     potential_clouds.append("Altocumulus Castellanus")

        if not potential_clouds: return ["Cel Serè o Núvols residuals"]
        return sorted(list(set(potential_clouds)))
    except Exception as e:
        return [f"No s'ha pogut determinar la nuvolositat. Error: {e}"]

# =========================================================================
# === 3. FUNCIONS DE DIBUIX (Aquestes funcions no canvien) ================
# =========================================================================
def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if cape.m <= 0 or not lcl_p: return None, None
    cloud_base_km = lcl_h / 1000.0
    if convergence_active:
        cloud_top_km = el_h / 1000.0 if el_h > lcl_h else cloud_base_km
    else:
        if not lfc_p:
            cloud_top_km = cloud_base_km + 0.1
        else:
            try:
                rh = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile)
                indices_above_lcl = np.where(p_levels <= lcl_p)[0]
                p_top = p_levels[-1]
                if len(indices_above_lcl) > 0:
                    for idx in indices_above_lcl:
                        if rh[idx] < 0.7: p_top = p_levels[idx]; break
                cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
            except: cloud_top_km = cloud_base_km
    return (cloud_base_km, cloud_top_km) if cloud_base_km is not None and cloud_top_km is not None and cloud_top_km > cloud_base_km else (None, None)

def _draw_cumulonimbus(ax, base_km, top_km):
    center_x, num_points = 0, 20
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    r_pts = [(center_x + widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    l_pts = [(center_x - widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    ax.add_patch(Polygon([(l_pts[0][0], l_pts[0][1])] + r_pts + l_pts[::-1], facecolor='#d8d8d8', lw=0, zorder=10))

def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100); ax.set_xlim(-50, 45)
    td_profile = np.minimum(t_profile, td_profile)
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
        skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
        parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
        skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
        skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
        skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    _, _, lcl_p, _, lfc_p, _, el_p, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    xlims = ax.get_xlim()
    if lcl_p: ax.plot(xlims, [lcl_p.m, lcl_p.m], 'gray', linestyle='--', label='LCL')
    if lfc_p: ax.plot(xlims, [lfc_p.m, lfc_p.m], 'purple', linestyle='--', label='LFC')
    if el_p: ax.plot(xlims, [el_p.m, el_p.m], 'red', linestyle='--', label='EL')
    ax.legend()
    plt.tight_layout()
    return fig

def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type):
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud (km)"); ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5); ax.set_facecolor('#6495ED')
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color=ground_color, alpha=0.8, zorder=3))
    if base_km is not None and top_km is not None and (top_km - base_km > 0.1):
        if "Cumulonimbus" in cloud_type:
             _draw_cumulonimbus(ax, base_km, top_km)
    return fig

def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray'); ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.grid(True, linestyle=':', alpha=0.3, color='white')
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if cape.m < 100:
        ax.text(0, 0, "Sense precipitació significativa", ha='center', va='center', color='white', fontsize=9)
        return fig
    shear_0_6, *_ = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    max_dbz = np.clip(20 + (cape.m / 3000) * 55, 20, 75)
    elongation = np.clip(1 + (shear_0_6 / 20), 1, 2.5)
    x, y = np.linspace(-50, 50, 150), np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    sigma_x, sigma_y = 15, 15 / elongation
    Z = max_dbz * np.exp(-((xx**2 / (2 * sigma_x**2)) + (yy**2 / (2 * sigma_y**2))))
    Z += gaussian_filter(np.random.randn(150, 150), sigma=6) * (max_dbz * 0.1)
    radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff']
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 75]
    radar_cmap = ListedColormap(radar_colors)
    radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    return fig

def create_hodograph_figure(p, ws, wd, t, td):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=40.)
    h.add_grid(increment=10, ls='--', color='gray')
    ax.set_xlabel('kt'); ax.set_ylabel('kt')
    try:
        u, v = mpcalc.wind_components(ws.to('kt'), wd.to('deg'))
        heights = mpcalc.pressure_to_height_std(p).to('km')
        h_interp = np.arange(0, min(12, heights.m.max()), 0.1) * units.km
        u_interp = np.interp(h_interp.m, heights.m, u.m) * units.kt
        v_interp = np.interp(h_interp.m, heights.m, v.m) * units.kt
        levels, colors = [0, 1, 3, 5, 8], ['green', 'orange', 'red', 'purple']
        cmap, norm = ListedColormap(colors), BoundaryNorm(levels, len(colors))
        for i in range(len(h_interp) - 1):
            ax.plot(u_interp[i:i+2].m, v_interp[i:i+2].m, color=cmap(norm(h_interp[i].m)), linewidth=2)
    except Exception:
        ax.text(0.5, 0.5, "Dades de vent insuficients.", ha='center', va='center', transform=ax.transAxes)
    return fig

# =========================================================================
# === 4. VISTA PRINCIPAL I INTERFICIE D'USUARI ============================
# =========================================================================

def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False):
    """
    Construeix la pàgina completa d'anàlisi amb tots els gràfics, paràmetres i xat.
    """
    st.markdown(f"#### {obs_time}")
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:10px;"><h3 style="color:white;text-align:center;">{title}</h3><p style="color:white;text-align:center;font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    
    st.toggle("Activar Forçament (Convergència)", key='convergence_active', help="Simula un mecanisme de tret. Si està activat, els núvols creixeran fins al seu topall teòric (EL) si hi ha CAPE.")
    convergence_active = st.session_state.get('convergence_active', False)

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, convergence_active)
    
    # Lògica de determinació del tipus de núvol principal
    cloud_type = "Cel Serè"
    if t[0].m < 7.0: cloud_type = "Hivernal"
    elif cape.m > 1500 and srh_0_1 > 150 and lcl_h < 1000: cloud_type = "Supercèl·lula (Tornàdica)"
    elif cape.m > 1200: cloud_type = "Cumulonimbus (Multicèl·lula)"
    elif cape.m > 300: cloud_type = "Cumulus Congestus"

    st.subheader("Diagrama Skew-T", anchor=False)
    st.pyplot(create_skewt_figure(p, t, td, ws, wd), use_container_width=True)
    st.divider()

    if is_sandbox_mode: chat_log, precipitation_type = generate_dynamic_analysis(p, t, td, ws, wd, cloud_type)
    else: chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, units.Quantity(0, 'mm'))

    potential_clouds = determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Assistent d'Anàlisi", "📊 Paràmetres", "📈 Hodògraf", "☁️ Visualització", "📡 Radar"])
    
    with tab1:
        css_styles = """<style>.chat-container { ... }</style>""" # Ometut per brevetat
        html_chat = "<div class='chat-container'>" + "".join([f"<div ...><strong>{s}</strong>{m}</div>" for s, m in chat_log]) + "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)

    with tab2:
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("Shear 0-6km", f"{shear_0_6:.1f} m/s")

    with tab3:
        st.pyplot(create_hodograph_figure(p, ws, wd, t, td), use_container_width=True)
        
    with tab4:
        st.pyplot(create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type), use_container_width=True)

    with tab5:
        st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

# =========================================================================
# === 5. MODES DE L'APLICACIÓ (Welcome, Live, Sandbox) ====================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">TEMPESTES.CAT PRESENTA:</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Eina de visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️ Temps real</h3><p>Visualitza sondejos actualitzats del model AROME per a diferents localitzacions.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode temps real", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪 Laboratori</h3><p>Aprèn de forma interactiva modificant un sondeig o experimenta lliurement.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def show_province_selection_screen():
    """
    Mostra una pantalla de selecció de província amb un fons animat.
    """
    animation_html = """
    <style>
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        @keyframes day-night-cycle { 0%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)}20%{background:linear-gradient(to top,#020111 20%,#2c3e50,#f39c12,#f1c40f)}35%{background:linear-gradient(to top,#87CEEB 20%,#B2FFFF)}60%{background:linear-gradient(to top,#87CEEB 20%,#B2FFFF)}75%{background:linear-gradient(to top,#1e2531 20%,#e74c3c,#f39c12)}90%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)}100%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)} }
        @keyframes sun-moon-move { 0%{transform:translate(-10vw,30vh);background:#F5F3CE;box-shadow:0 0 20px #F5F3CE;opacity:1}20%{transform:translate(50vw,-20vh);background:#FFD700;box-shadow:0 0 40px #FFD700;opacity:1}60%{transform:translate(110vw,30vh);opacity:1}75%{transform:translate(50vw,80vh);opacity:0}80%{transform:translate(-10vw,30vh);background:#F5F3CE;box-shadow:0 0 20px #F5F3CE;opacity:1}100%{transform:translate(-10vw,30vh);opacity:1} }
        @keyframes stars-opacity { 0%{opacity:1}20%{opacity:0}75%{opacity:0}90%{opacity:1}100%{opacity:1} }
        @keyframes shooting-star-anim { 0%{transform:translate(120vw,-30vh);opacity:0}94%{transform:translate(120vw,-30vh);opacity:0}95%{transform:translate(80vw,10vh) scale(1);opacity:1}100%{transform:translate(-40vw,90vh) scale(1);opacity:0} }
        @keyframes twinkle { 0%,100%{opacity:1}50%{opacity:.3} }
        .animation-container { position:fixed;top:0;left:0;width:100%;height:100vh;overflow:hidden;z-index:-1 }
        .animated-sky { position:absolute;top:0;left:0;width:100%;height:100%;animation:day-night-cycle 40s linear infinite }
        .sun-moon { position:absolute;width:clamp(60px,8vw,120px);height:clamp(60px,8vw,120px);border-radius:50%;animation:sun-moon-move 40s linear infinite }
        .stars-wrapper { position:absolute;top:0;left:0;width:100%;height:60%;animation:stars-opacity 40s linear infinite }
        .star { position:absolute;background:white;border-radius:50%;animation:twinkle 4s ease-in-out infinite }
        .shooting-star { position:absolute;width:3px;height:100px;background:linear-gradient(to bottom,rgba(255,255,255,1),rgba(255,255,255,0));border-radius:50%;transform-origin:top center;filter:drop-shadow(0 0 6px white);animation:shooting-star-anim 40s linear infinite }
        .landscape { position:absolute;bottom:0;left:0;width:100%;height:35vh;background:#000 }
        .city-silhouette,.mountain-silhouette { position:absolute;bottom:0;width:100%;height:100%;background-repeat:no-repeat;background-position:bottom }
        .mountain-silhouette { background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 175"><path d="M0 175 H 500 V 100 L 450 120 L 400 60 L 350 110 L 300 30 L 250 100 L 200 80 L 150 130 L 100 90 L 50 140 L 0 110 Z" fill="%2312182c"/></svg>');background-size:100% 100%;z-index:2 }
        .city-silhouette { background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 150"><path d="M0 150 V 100 L 20 105 L 40 80 L 60 90 L 100 40 L 140 110 L 180 100 L 220 50 L 250 120 L 280 110 L 320 20 L 360 130 L 400 110 L 450 10 L 500 140 L 550 120 L 600 70 L 650 130 L 700 90 L 750 120 L 800 80 V 150 H 0 Z" fill="%230a0a0a"/></svg>');background-size:100% 100%;z-index:3 }
        .city-lights-wrapper { position:absolute;bottom:0;left:0;width:100%;height:100%;z-index:4;animation:stars-opacity 40s linear infinite }
        .light { position:absolute;background:#fde9a0;border-radius:50%;width:.3%;height:1.5%;animation:twinkle 3s ease-in-out infinite }
    </style>
    <div class="animation-container"><div class="animated-sky"><div class="sun-moon"></div><div class="stars-wrapper">
    """ + ''.join([f'<div class="star" style="top:{random.uniform(0,70)}%;left:{random.uniform(0,100)}%;width:{random.uniform(1,2.5)}px;height:{random.uniform(1,2.5)}px;animation-delay:{random.uniform(0,4)}s;"></div>' for _ in range(200)]) + """
    </div><div class="shooting-star"></div></div><div class="landscape"><div class="mountain-silhouette"></div><div class="city-silhouette"></div><div class="city-lights-wrapper">
    """ + ''.join([f'<div class="light" style="bottom:{random.uniform(5,50)}%;left:{random.uniform(2,98)}%;animation-delay:{random.uniform(0,3)}s;"></div>' for _ in range(80)]) + """
    </div></div></div>
    """
    st.markdown(animation_html, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:white;text-shadow:2px 2px 6px #000;padding-top:25vh;'>Selecciona una Província</h2>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        def select_province(name):
            st.session_state.province_selected = name
        st.button("Barcelona", on_click=select_province, args=('barcelona',), use_container_width=True, type="primary")
        st.button("Girona", on_click=select_province, args=('girona',), use_container_width=True)
        st.button("Lleida", on_click=select_province, args=('lleida',), use_container_width=True)
        st.button("Tarragona", on_click=select_province, args=('tarragona',), use_container_width=True)

def run_live_mode():
    # Diccionari amb les coordenades de cada lloc. Fàcil d'ampliar!
    # Aquestes són aproximades. Pots trobar les teves clicant al mapa de Meteociel.
    LOCATIONS = {
        'barcelona': {'name': 'Barcelona', 'lat': 668, 'lon': 467},
        'girona': {'name': 'Girona', 'lat': 668, 'lon': 437},
        'lleida': {'name': 'Lleida', 'lat': 738, 'lon': 477},
        'tarragona': {'name': 'Tarragona', 'lat': 698, 'lon': 497}
    }

    selected_province_key = st.session_state.get('province_selected')

    # Si no hi ha cap província seleccionada, mostrem la pantalla de selecció
    if not selected_province_key:
        with st.sidebar:
            st.header("Controls")
            if st.button("⬅️ Tornar a l'inici", use_container_width=True):
                st.session_state.app_mode = 'welcome'
                st.rerun()
        show_province_selection_screen()
        return

    # Si ja tenim una província, obtenim les seves dades
    location_data = LOCATIONS.get(selected_province_key)
    set_main_background() 
    st.title(location_data['name'].upper())
    
    with st.sidebar:
        st.header("Controls")
        
        def back_to_selection():
            st.session_state.province_selected = None
        st.button("⬅️ Tornar a la selecció", use_container_width=True, on_click=back_to_selection)
        
        st.markdown("---")
        st.subheader("Selecciona pronòstic")
        st.info("Dades del model AROME obtingudes en temps real des de Meteociel.fr")

        # === CANVI CLAU AQUÍ ===
        # El rang de pronòstics ara comença a 1 i arriba fins a 48 hores.
        forecast_hours = list(range(1, 49, 1))
        
        # Selector per a les hores de pronòstic
        selected_hour = st.selectbox(
            "Hora del pronòstic:",
            options=forecast_hours,
            format_func=lambda h: f"+ {h} hores",
            key=f"hour_selector_{selected_province_key}"
        )

    # Obtenim les dades de la URL cada vegada que canvia l'hora seleccionada
    # Streamlit desa a la memòria cau el resultat si els paràmetres no canvien
    @st.cache_data(ttl=600) # Cau de 10 minuts per no saturar Meteociel
    def get_data_for_hour(lat, lon, hour):
        return fetch_sounding_from_url(lat, lon, hour)

    # Mostrem una animació de càrrega mentre es descarreguen les dades
    with st.spinner(f"Obtenint dades per a {location_data['name']} a +{selected_hour}h..."):
        sounding_data = get_data_for_hour(location_data['lat'], location_data['lon'], selected_hour)
    
    if sounding_data:
        show_full_analysis_view(
            p=sounding_data['p_levels'], 
            t=sounding_data['t_initial'], 
            td=sounding_data['td_initial'], 
            ws=sounding_data['wind_speed_kmh'].to('m/s'), 
            wd=sounding_data['wind_dir_deg'], 
            obs_time=sounding_data.get('observation_time', 'Hora no disponible'), 
            is_sandbox_mode=False
        )
    else:
        st.error(f"No s'han pogut obtenir les dades del sondeig per a {location_data['name']} a +{selected_hour}h. Prova-ho més tard o selecciona una altra hora.")

def run_sandbox_mode():
    """Executa el mode Laboratori/Tutorial."""
    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings: 
            st.error("No s'ha trobat 'sondeigproves.txt'. Aquest fitxer és essencial per al Mode Laboratori.")
            return
        data = soundings[0]
        st.session_state.sandbox_original_data = data
        st.session_state.sandbox_p_levels = data['p_levels'].copy()
        st.session_state.sandbox_t_profile = data['t_initial'].copy()
        st.session_state.sandbox_td_profile = data['td_initial'].copy()
        st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True

    with st.sidebar:
        st.header("Caixa d'Eines")
        if st.button("⬅️ Tornar al Menú Principal", use_container_width=True):
            st.session_state.app_mode = 'welcome'
            st.rerun()

    # Aquí aniria la lògica del sandbox: show_sandbox_selection_screen, show_tutorial_interface, etc.
    # Ometut per brevetat en aquesta vista general, però és crucial.
    st.title("🧪 Laboratori de Sondejos")
    st.info("El Mode Laboratori està en construcció en aquesta versió simplificada del codi.")


# =========================================================================
# === PUNT D'ENTRADA DE L'APLICACIÓ =======================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Analitzador de Sondejos")

    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'

    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()

