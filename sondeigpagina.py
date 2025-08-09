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

# El pany segueix sent crucial per evitar errors de concurrència.
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

def show_loading_animation(message="Carregant"):
    """Mostra una animació de càrrega personalitzada amb HTML i CSS."""
    loading_html = f"""
    <style>
        .loading-container {{
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
        }}
        .loading-svg {{
            width: 150px;
            height: auto;
            margin-bottom: 20px;
        }}
        .loading-text {{
            color: white;
            font-size: 1.5rem;
            font-family: sans-serif;
        }}
        .loading-text .dot {{
            animation: blink 1.4s infinite both;
        }}
        .loading-text .dot:nth-child(2) {{
            animation-delay: 0.2s;
        }}
        .loading-text .dot:nth-child(3) {{
            animation-delay: 0.4s;
        }}
        @keyframes blink {{
            0%, 80%, 100% {{ opacity: 0; }}
            40% {{ opacity: 1; }}
        }}
    </style>
    <div class="loading-container">
        <svg class="loading-svg" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
            <path d="M 155.6,66.1 C 155.6,42.9 135.5,23.5 111.4,23.5 C 98.4,23.5 86.8,29.4 79.1,38.7 C 75.2,16.8 57.3,0 36.4,0 C 16.3,0 0,16.3 0,36.4 C 0,56.5 16.3,72.8 36.4,72.8 L 110,72.8 C 110,72.8 110,72.8 110,72.8 C 135,72.8 155.6,93.4 155.6,118.4 C 155.6,143.4 135,164 110,164 L 50, 164" fill="none" stroke="#FFFFFF" stroke-width="8"/>
            <polygon points="120,60 90,110 115,110 100,150 145,90 120,90 130,60" fill="#FFD700" />
        </svg>
        <div class="loading-text">
            {message}<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
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

def create_city_mountain_scape():
    """Crea una figura de Matplotlib amb una escena de ciutat i muntanya."""
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    star_x, star_y = np.random.uniform(0, 100, 200), np.random.uniform(15, 60, 200)
    star_s, star_alpha = np.random.uniform(0.5, 2.5, 200), np.random.uniform(0.5, 1, 200)
    ax.scatter(star_x, star_y, s=star_s, c='white', alpha=star_alpha, edgecolors='none')
    mountain_poly = Polygon([(55, 0), (68, 38), (75, 32), (85, 45), (95, 28), (100, 32), (100, 0)], facecolor='#12182c', edgecolor=None, zorder=5)
    ax.add_patch(mountain_poly)
    city_patches, light_patches = [], []
    for x_base in np.arange(0, 70, 0.5):
        height_factor = 1 - abs(x_base - 35) / 35
        building_height = (random.uniform(2, 12) * (1 + height_factor * 2))
        building_width = random.uniform(0.8, 3)
        color_val = random.uniform(0.05, 0.1)
        building = Rectangle((x_base, 0), building_width, building_height, facecolor=(color_val, color_val, color_val), edgecolor=None, zorder=10)
        city_patches.append(building)
        if random.random() < 0.08:
            light_x, light_y = x_base + random.uniform(0, building_width), random.uniform(1, building_height * 0.5)
            light = Circle((light_x, light_y), radius=0.15, color='#fde9a0', alpha=0.9)
            light_patches.append(light)
    ax.add_collection(PatchCollection(city_patches, match_original=True))
    ax.add_collection(PatchCollection(light_patches, match_original=True, zorder=11))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f: data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError: return None

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
    # L'extracció de la isoterma des de l'arxiu s'ha eliminat per usar el valor calculat, que és més fiable.
    return {'p_levels': np.array(p_list)[sorted_indices] * units.hPa, 
            't_initial': np.array(t_list)[sorted_indices] * units.degC, 
            'td_initial': np.array(td_list)[sorted_indices] * units.degC, 
            'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph, 
            'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees, 
            'observation_time': observation_time}

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

def calculate_thermo_parameters(p_levels, t_profile, td_profile):
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
    """Genera l'anàlisi conversacional per al mode 'Live'."""
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
        warm_layer_present = np.any(warm_layer_mask)
        if not warm_layer_present:
            chat_log.append(("Analista", "Bones notícies. He revisat tota la columna d'aire i sembla que es manté per sota o molt a prop de 0°C en tot el recorregut."))
            if t_profile[0].m > 1.5:
                chat_log.append(("Usuari", f"Llavors, la temperatura a la superfície ({t_profile[0].m:.1f}°C) no és massa alta?"))
                chat_log.append(("Analista", f"És una bona observació. Una temperatura de {t_profile[0].m:.1f}°C pot fer que els flocs es fonguin just en arribar o donin una neu molt humida."))
                precipitation_type = 'rain'
            else:
                chat_log.append(("Usuari", "Això vol dir que si precipita, serà en forma de neu?"))
                chat_log.append(("Analista", "Exacte. Aquest és un 'perfil de nevada'. Si hi ha precipitació, serà en forma de neu."))
                precipitation_type = 'snow'
        else:
            max_temp_in_layer = np.max(t_array[warm_layer_mask])
            chat_log.append(("Analista", f"Alerta! He detectat una 'capa càlida' en altura. La temperatura puja fins a {max_temp_in_layer:.1f}°C."))
            chat_log.append(("Usuari", "I això què significa exactament? Adéu a la neu?"))
            if t_profile[0].m <= 0.5:
                chat_log.append(("Analista", "Aquesta capa fon els flocs, però com que el terra està sota zero, les gotes es tornen a congelar. El resultat més probable és **aiguaneu (sleet)** o la perillosa **pluja gelant**."))
                precipitation_type = 'sleet'
            else:
                 chat_log.append(("Analista", "Exactament. La capa càlida fon la neu i, com que la superfície és positiva, arribarà com a pluja freda. És el típic escenari de 'plou i fa fred'."))
                 precipitation_type = 'rain'
    else:
        if "Tornàdica" in cloud_type or "Tuba" in cloud_type or "Mur" in cloud_type: precipitation_type = 'hail'
        elif cape.m > 500: precipitation_type = 'rain'
        elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
        
        chat_log = [("Sistema", f"Iniciant anàlisi conversacional per a l'escenari de {cloud_type}.")]

        if "Tornàdica" in cloud_type:
            chat_log.extend([
                ("Analista", "ALERTA MÀXIMA. El perfil no només és de supercèl·lula, sinó que presenta característiques tornàdiques clàssiques."),
                ("Usuari", "Què ho fa tan perillós?"),
                ("Analista", f"Tenim tres ingredients clau: una base del núvol molt baixa ({lcl_h:.0f} m), una rotació a nivells baixos extremadament forta (SRH 0-1km: {srh_0_1:.0f} m²/s²), i una forta inestabilitat. La rotació té moltes probabilitats d'arribar a terra."),
            ])
        elif "Tuba/Funnel" in cloud_type:
            chat_log.extend([
                ("Analista", "Molt de compte. Aquest és un perfil de supercèl·lula amb un alt potencial per desenvolupar embuts."),
                ("Usuari", "Què indica aquest potencial?"),
                ("Analista", f"La combinació d'una helicitat significativa a nivells baixos (SRH 0-1km: {srh_0_1:.0f} m²/s²) i una base de núvol relativament baixa ({lcl_h:.0f} m). La rotació pot baixar i condensar-se fàcilment."),
            ])
        elif "Mur de núvols" in cloud_type:
            chat_log.extend([
                ("Analista", "Aquest és un perfil de supercèl·lula clàssic i molt organitzat."),
                ("Usuari", "Què el fa especial?"),
                ("Analista", f"La rotació a nivells mitjans és molt intensa (SRH 0-3km: {srh_0_3:.0f} m²/s²). Això pot provocar una baixada localitzada de la base, formant un mur de núvols, que és la regió principal d'on neixen els tornados."),
            ])
        elif "Shelf Cloud" in cloud_type:
             chat_log.extend([
                ("Analista", "Atenció. Aquest perfil és perillós, però per un motiu diferent a la rotació."),
                ("Usuari", "No és una supercèl·lula?"),
                ("Analista", f"No exactament. Tot i la gran energia (CAPE: {cape.m:.0f} J/kg), la clau aquí és el potencial per a una corrent descendent (downdraft) molt violenta. El principal perill són els vents lineals destructius (reventón o downburst)."),
            ])
        elif "Base Rugosa" in cloud_type:
            chat_log.extend([
                ("Analista", "Tenim una tempesta forta en marxa."),
                ("Usuari", "Quin és el detall més significatiu?"),
                ("Analista", f"L'energia és alta (CAPE: {cape.m:.0f} J/kg) i hi ha una forta entrada d'aire humit. Això fa que es condensi humitat per sota de la base principal, creant una base rugosa amb fragments (scud). És un senyal que la tempesta s'està alimentant amb força."),
            ])
        elif "Supercèl·lula" in cloud_type:
            chat_log.extend([
                ("Analista", "Aquest és un perfil de manual per a temps sever."),
                ("Usuari", f"Veig molta energia (CAPE: {cape.m:.0f} J/kg) i cisallament ({shear_0_6:.1f} m/s)."),
                ("Analista", "Exacte. Aquesta combinació crea un motor que permet que una tempesta s'organitzi i roti. El pronòstic ha de ser de precaució per calamarsa gran i vents forts."),
            ])
        elif cloud_type == "Nimbostratus":
            chat_log.extend([
                ("Analista", "Aquest perfil és molt diferent. Aquí la història no va d'inestabilitat explosiva."),
                ("Usuari", f"És cert, el CAPE és gairebé inexistent, només {cape.m:.0f} J/kg."),
                ("Analista", f"Exacte. El protagonista aquí és la humitat. Tenim una capa d'aire molt gruixuda i saturada. És un escenari típic de pluja estratiforme, associada a sistemes frontals. La intensitat dependrà de l'aigua precipitable ({pwat_0_4.m:.1f} mm)."),
            ])
        elif cloud_type == "Cirrus":
            chat_log.extend([
                ("Analista", "Interessant. El perfil està molt sec a les capes baixes i mitjanes."),
                ("Usuari", "Llavors no hi haurà núvols?"),
                ("Analista", "A gran altura, per sobre dels 6-7 km, hi ha una fina capa d'humitat. Això crea les condicions ideals per als Cirrus, núvols alts de vidres de gel que no produeixen precipitació."),
            ])
        elif cloud_type == "Altostratus / Altocumulus":
            chat_log.extend([
                ("Analista", "Tenim un cas de nuvolositat a nivells mitjans."),
                ("Usuari", f"Per què? No hi ha gairebé CAPE ({cape.m:.0f} J/kg)."),
                ("Analista", "Correcte, la convecció des de superfície està inhibida. No obstant, observa la marcada capa d'humitat entre els 3 i 6 km. Això formarà una capa de núvols mitjans, com Altostratus o Altocumulus."),
            ])
        elif cloud_type == "Cumulus Humilis":
            chat_log.extend([
                ("Analista", "Estem observant un escenari de temps estable."),
                ("Usuari", f"Però hi ha una mica de CAPE, {cape.m:.0f} J/kg."),
                ("Analista", "Sí, una mica d'energia hi ha, però molt poca i amb una forta 'tapadera' just a sobre que impedeix qualsevol creixement. És un perfil típic per als 'núvols de bon temps'."),
            ])
        elif cloud_type == "Cumulus Mediocris":
            chat_log.extend([
                ("Analista", "Aquest és un perfil interessant per a una tarda d'estiu."),
                ("Usuari", f"Tenim {cape.m:.0f} J/kg de CAPE. És suficient per a tempestes?"),
                ("Analista", "És una energia moderada amb cisallament feble. Permet un cert creixement vertical, però no explosiu. Afavoreix la formació de Cumulus Mediocris, que rarament donen més que quatre gotes."),
            ])
        elif cloud_type == "Cumulus Congestus":
            chat_log.extend([
                ("Analista", "Atenció, aquí comencem a veure potencial per a fenòmens més actius."),
                ("Usuari", f"El CAPE ja és més considerable, {cape.m:.0f} J/kg."),
                ("Analista", "Exacte. Tenim prou energia per a un desenvolupament vertical important. Són el pas previ al Cumulonimbus i ja poden deixar ruixats o xàfecs localment intensos."),
            ])
        elif cloud_type == "Cumulonimbus (Multicèl·lula)":
            chat_log.extend([
                ("Analista", "Bé, tenim un escenari amb potencial de tempestes."),
                ("Usuari", f"El CAPE és de {cape.m:.0f} J/kg."),
                ("Analista", f"És un bon valor, suficient per a tempestes fortes. A més, el LFC ({lfc_h:.0f} m) és prou baix per permetre que la convecció arrenqui."),
                ("Usuari", "I s'organitzaran? Com és el cisallament?"),
            ])
            if shear_0_6 < 10: shear_analysis_message = f"El cisallament ({shear_0_6:.1f} m/s) és feble. Les tempestes seran probablement desorganitzades i de cicle de vida curt."
            elif shear_0_6 < 18: shear_analysis_message = f"El cisallament ({shear_0_6:.1f} m/s) és moderat. Permetrà sistemes multicel·lulars més duradors."
            else: shear_analysis_message = f"El cisallament ({shear_0_6:.1f} m/s) és fort. Hi ha una organització considerable i les tempestes seran robustes."
            chat_log.append(("Analista", shear_analysis_message))
        elif cloud_type == "Castellanus":
            chat_log.extend([
                ("Analista", "Aquest és un cas particular. Tenim energia en altura, però la superfície està desconnectada."),
                ("Usuari", f"Què vols dir?"),
                ("Analista", f"El CIN és molt fort ({cin.m:.0f} J/kg), el que impedeix que la convecció comenci des del terra. No obstant, hi ha una capa inestable a nivells mitjans que pot generar Altocumulus Castellanus."),
            ])
        elif cloud_type == "Cumulus Fractus":
             chat_log.extend([
                ("Analista", "El que veiem aquí són condicions residuals."),
                ("Usuari", "Què vol dir això?"),
                ("Analista", "Hi ha una mica d'humitat i inestabilitat, però és molt poca i desorganitzada. Només permetrà la formació de trossos de núvols esquinçats, sense cap risc associat."),
            ])
        else:
            chat_log.extend([
                ("Analista", "El perfil atmosfèric és molt estable."),
                ("Usuari", "Llavors, no veurem cap núvol?"),
                ("Analista", f"És molt poc probable. Amb un CAPE de només {cape.m:.0f} J/kg, no hi ha pràcticament gens d'energia per al creixement vertical."),
            ])

    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type):
    """Genera anàlisi conversacional per al mode laboratori."""
    cape, cin, _, lcl_h, _, lfc_h, _, _, _ = calculate_thermo_parameters(p, t, td)
    shear_0_6, _, _, _ = calculate_storm_parameters(p, ws, wd)
    chat_log = []
    
    chat_log.append(("Analista", "Molt bé, anem a analitzar el perfil que has creat. Comencem?"))

    if cape.m < 50:
        chat_log.extend([
            ("Usuari", "Tenim potencial per a tempestes?"),
            ("Analista", f"Ara mateix no. El CAPE és de només {cape.m:.0f} J/kg. L'atmosfera està molt estable.")
        ])
    else:
        chat_log.extend([("Usuari", "Què estic creant amb aquesta energia?")])
        cloud_mention = f"Això és un escenari típic per a la formació de {cloud_type}."
        if cloud_type == "Cel Serè":
             cloud_mention = "Encara que hi ha energia, la tapadera és tan forta que probably no veuríem cap núvol significatiu."
        chat_log.append(("Analista", f"Has generat un CAPE de {cape.m:.0f} J/kg. {cloud_mention}"))

        chat_log.append(("Usuari", "I la 'tapadera' (CIN)? Com afecta?"))
        if cin.m < -100:
            chat_log.append(("Analista", f"Molt forta ({cin.m:.0f} J/kg). La convecció des de superfície és gairebé impossible."))
        elif cin.m < -50:
            chat_log.append(("Analista", f"És considerable ({cin.m:.0f} J/kg). Obre la porta a la convecció de base elevada (Castellanus)."))
        elif cin.m < -25:
            chat_log.append(("Analista", f"És moderada ({cin.m:.0f} J/kg). Permet que l'energia s'acumuli, un escenari clàssic per a tempestes fortes."))
        else:
             chat_log.append(("Analista", f"És feble ({cin.m:.0f} J/kg). La convecció té gairebé via lliure."))
        
        if cin.m > -100 and cape.m > 800:
            chat_log.append(("Usuari", "He modificat el vent. Com afecta?"))
            if shear_0_6 > 18:
                chat_log.append(("Analista", f"El cisallament és fort ({shear_0_6:.1f} m/s). Aquest és l'ingredient clau per organitzar les tempestes en supercèl·lules."))
            elif shear_0_6 > 10:
                chat_log.append(("Analista", f"El cisallament és moderat ({shear_0_6:.1f} m/s). Ajuda a organitzar les tempestes en sistemes multicel·lulars."))
            else:
                chat_log.append(("Analista", f"El cisallament és feble ({shear_0_6:.1f} m/s). Les tempestes probablement seran més desorganitzades."))

    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Genera l'anàlisi del xat per a un pas específic d'un tutorial."""
    chat_log = []
    if scenario == 'aiguaneu':
        if step == 0: chat_log.extend([("Analista", "Benvingut! Anem a analitzar un perfil clàssic d'aiguaneu."), ("Usuari", "Perfecte. Què és el primer que he de mirar?"), ("Analista", "Observa la 'fàbrica de neu' a les capes altes. Per sobre de 700 hPa fa prou fred per formar flocs de neu.")])
        elif step == 1: chat_log.extend([("Analista", "Molt bé. Ara ve la part clau. Fixa't en la capa al voltant de 850 hPa. La temperatura puja per sobre dels 0°C."), ("Usuari", "Això és la 'capa càlida', oi? Què provoca?"), ("Analista", "Exacte. Aquesta capa actua com un 'bufador' i fon els flocs, convertint-los en gotes de pluja.")])
        elif step == 2: chat_log.extend([("Analista", "Ja gairebé ho tenim. Ara tenim gotes de pluja caient cap a la superfície. Però mira la temperatura a prop del terra..."), ("Usuari", "Torna a estar per sota de 0°C!"), ("Analista", "Precisament! Aquestes gotes es tornen a congelar just abans d'arribar a terra. Això és l'aiguaneu (sleet).")])
        elif step == 3: chat_log.extend([("Analista", "Has analitzat el perfil a la perfecció."), ("Usuari", "Entès. Llavors, com ho podria convertir en una nevada?"), ("Analista", "Aquest és el repte! Ara, quan finalitzis el tutorial, ves al Mode Lliure i utilitza l'eina '❄️ Refredar Capa Mitjana'. Veuràs com el perfil es converteix en una nevada perfecta.")])
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem el tutorial de supercèl·lula. El primer pas és sempre crear energia. Necessitem un dia càlid d'estiu. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "Correcte! Ara, afegim el combustible: la humitat. Veuràs com augmenta el valor de CAPE quan les línies de temperatura i punt de rosada s'acosten."))
        elif step == 2: chat_log.append(("Analista", "Fantàstic! Has afegit cisallament. Aquest és l'ingredient secret que fa que les tempestes rotin. Ara tenim la recepta perfecta!"))
        elif step == 3: chat_log.append(("Analista", "Missió complerta! Has creat un perfil amb molta energia (CAPE), humitat i cisallament. Fixa't com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."))

    return chat_log, None
    
# MODIFICAT: Ara la funció utilitza el `fz_h` calculat, no el que ve de l'arxiu
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    if sfc_temp.m < 7.0: # Anàlisi hivernal
        if sfc_temp.m <= 0.5:
            try:
                p_arr, t_arr = p_levels.m, t_profile.m
                warm_layer_mask = (p_arr < 950) & (p_arr > 600) & (t_arr > 0.5)
                if np.any(warm_layer_mask):
                    return "AIGUANEU O PLUJA GEBRADORA", "Capa càlida en altura pot fondre la neu. Risc d'aiguaneu o pluja gelant.", "mediumorchid"
                else:
                    return "AVÍS PER NEU", "Perfil atmosfèric favorable a nevades a cotes baixes.", "navy"
            except:
                return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades. Extremi les precaucions.", "dodgerblue"
            else:
                 return "AMBIENT FRED I HUMIT", "Condicions de fred. La precipitació seria en forma de pluja o neu molt humida.", "steelblue"

    if cape.m >= 1200:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)

        if cin.m <= -100: return "CONVECCIÓ FORTAMENT INHIBIDA", f"Potencial energètic (CAPE {cape.m:.0f} J/kg) bloquejat per una 'tapadera' molt forta (CIN {cin.m:.0f} J/kg).", "darkslategray"
        if -100 < cin.m <= -50: return "POSSIBLE CONVECCIÓ DE MITJÀ NIVELL", f"La convecció des de superfície és difícil (CIN {cin.m:.0f} J/kg). Risc de nuclis elevats.", "slategray"

        title = "AVÍS PER TEMPESTES SEVERES"; color = "darkorange"; message = f"CAPE: {cape.m:.0f} J/kg. "

        if srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18:
            title, color = "AVÍS PER TORNADO", "darkred"; message += f"Alt risc de tornados (SRH 0-1km: {srh_0_1:.0f}, LCL: {lcl_h:.0f}m)."
        elif srh_0_3 > 250 and shear_0_6 > 18:
            title, color = "AVÍS PER TEMPS SEVER", "purple"; message += f"Supercèl·lules probables. Risc de calamarsa gran i/o murs de núvols (SRH 0-3km: {srh_0_3:.0f})."
        elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150):
            title, color = "AVÍS PER VENTS FORTS", "saddlebrown"; message += "Risc de ratxes de vent lineals severes (downbursts/shelf cloud)."
        
        # === MODIFICAT: Nova lògica per a l'avís de pedra gran basada en la isoterma CALCULADA ===
        elif cape.m > 2000 and fz_h < 4200:
            title, color = "AVÍS PER PEDRA GRAN", "mediumvioletred"
            message += f"Condicions favorables per a calamarsa de gran mida (Iso 0°C calculada: {int(fz_h)} m)."
        elif cape.m > 2500: # Avís de fallback si la isoterma és molt alta
            title, color = "AVÍS PER PEDRA GRAN", "mediumvioletred"
            message += "Condicions favorables per a calamarsa de gran mida."
        
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
                if pwat_layer.m > 25: return "AVÍS PER PLUGES INTENSES", "(Activa el forçament) Risc de pluges persistents i fortes.", "darkblue"
                elif pwat_layer.m > 15: return "AVÍS PER PLUJA MODERADA", "Cel cobert amb pluja contínua i moderada.", "steelblue"
                else: return "PREVISIÓ DE PLUJA FEBLE", "(Activa el forçament) S'esperen plugims o ruixats febles.", "cadetblue"
    except Exception: pass

    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

def determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p):
    """Determina una llista de possibles tipus de núvols basant-se en les condicions del sondeig."""
    potential_clouds = []
    
    try:
        if len(p) < 2: return ["Dades insuficients"]
            
        heights = mpcalc.pressure_to_height_std(p).to('m')
        rh = mpcalc.relative_humidity_from_dewpoint(t, td)
        t_interp_func = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")

        mask_high = (heights.m > 6000) & (heights.m < 18000)
        if np.any(mask_high) and np.sum(mask_high) > 1:
            rh_high = rh[mask_high]
            if np.mean(rh_high) > 0.85: potential_clouds.append("Cirrostratus")
            elif np.mean(rh_high) > 0.80: potential_clouds.append("Cirrocumulus")
            elif np.mean(rh_high) > 0.75: potential_clouds.append("Cirrus")

        mask_mid = (heights.m > 2000) & (heights.m < 7000)
        if np.any(mask_mid) and np.sum(mask_mid) > 1:
            mask_as = (heights.m > 2000) & (heights.m < 6000)
            if np.any(mask_as) and np.sum(mask_as) > 1 and np.mean(rh[mask_as]) > 0.90 and cape.m < 50:
                 potential_clouds.append("Altostratus")
            elif (0.70 < np.mean(rh[mask_mid]) < 0.90) and cape.m <= 100:
                potential_clouds.append("Altocumulus")

        mask_low = (heights.m < 2500)
        if np.any(mask_low) and np.sum(mask_low) > 1:
            rh_low = rh[mask_low]
            mask_st = (heights.m < 1500)
            if np.any(mask_st) and np.sum(mask_st) > 1 and np.mean(rh[mask_st]) > 0.95 and cape.m < 10:
                potential_clouds.append("Stratus (Boira ascendent)")
            elif np.mean(rh_low) > 0.85 and cape.m <= 50:
                 potential_clouds.append("Stratocumulus")

        mask_nimbostratus = (heights.m > 500) & (heights.m < 5000)
        if np.any(mask_nimbostratus) and np.sum(mask_nimbostratus) > 1 and np.mean(rh[mask_nimbostratus]) > 0.95 and cape.m <= 100:
            potential_clouds.append("Nimbostratus")
            
        has_convective_potential = cape.m > 100 and lcl_h is not None and lcl_h > 0

        if has_convective_potential:
            if lfc_h is not None and lfc_h < 3000:
                if (100 < cape.m < 2500) and (lfc_h > lcl_h):
                    potential_clouds.append("Cumulus (Humilis, Mediocris o Congestus)")

                if cape.m > 1000:
                    is_iced_top = False
                    if el_p is not None and not np.isnan(el_p.m):
                        try:
                            t_at_el = t_interp_func(el_p.m)
                            if t_at_el < 0: is_iced_top = True
                        except: pass
                    if is_iced_top:
                        potential_clouds.append("Cumulonimbus")
            
            else:
                mask_mid_castellanus = (heights.m > 2000) & (heights.m < 7000)
                if np.any(mask_mid_castellanus) and np.sum(mask_mid_castellanus) > 1:
                    if np.mean(rh[mask_mid_castellanus]) > 0.60:
                         potential_clouds.append("Altocumulus Castellanus")

        final_clouds = []
        has_cb = "Cumulonimbus" in potential_clouds
        has_cu = "Cumulus (Humilis, Mediocris o Congestus)" in potential_clouds
        has_ns = "Nimbostratus" in potential_clouds
        has_castellanus = "Altocumulus Castellanus" in potential_clouds
        
        for cloud in potential_clouds:
            if cloud == "Cumulus (Humilis, Mediocris o Congestus)" and has_cb: continue
            if cloud == "Stratocumulus" and (has_cb or has_cu or has_ns): continue
            if cloud == "Altocumulus" and has_castellanus: continue
            if cloud not in final_clouds:
                final_clouds.append(cloud)

        if not final_clouds: return ["Cel Serè o Núvols residuals"]
            
        return sorted(list(set(final_clouds)))
        
    except Exception as e:
        return [f"No s'ha pogut determinar la nuvolositat. Error: {e}"]

# =========================================================================
# === 4. ESTRUCTURA DE L'APLICACIÓ =======================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">TEMPESTES.CAT PRESENTA :</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Una eina per a la visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️Temps real</h3><p>Visualitza els sondejos atmosfèrics més recents basats en dades de models per a les zones més actives del dia.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode temps real", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪Laboratori</h3><p>Aprèn de forma interactiva com es formen els fenòmens severs modificant pas a pas un sondeig o experimenta lliurement.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

# MODIFICAT: S'ha eliminat `iso_0_from_file_m` dels arguments
def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False):
    st.markdown(f"#### {obs_time}")
    
    # La isoterma calculada (`fz_h`) es passa ara implícitament dins de `generate_public_warning`
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    
    st.toggle(
        "Activar Forçament (Convergència)",
        key='convergence_active',
        help="Simula l'efecte d'un mecanisme de tret (p.ex. convergència o orografia). Si està activat, els núvols creixeran fins al seu topall teòric (EL) si hi ha CAPE, ignorant la inhibició (CIN). Si no, només es formaran en capes ja saturades o si la convecció pot vèncer el CIN per si sola."
    )
    convergence_active = st.session_state.get('convergence_active', False)

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, convergence_active)
    
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

    surface_height_m = mpcalc.pressure_to_height_std(p[0]).to('m').m
    if lfc_h is not None and lfc_h != np.inf:
        lfc_above_ground = lfc_h - surface_height_m
        convection_possible_from_surface = (cin.m > -100 and lfc_above_ground < 3000)
    else:
        convection_possible_from_surface = False

    if sfc_temp.m < 7.0: cloud_type = "Hivernal"
    elif cape.m > 1500 and srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercèl·lula (Tornàdica)"
    elif cape.m > 1500 and srh_0_1 > 120 and lcl_h < 1200 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercèl·lula (Tuba/Funnel)"
    elif cape.m > 1800 and srh_0_3 > 250 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercèl·lula (Mur de núvols)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150 and convection_possible_from_surface: cloud_type = "Supercèl·lula"
    elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150): cloud_type = "Cumulonimbus (Shelf Cloud)"
    elif cape.m > 1200 and s_0_1 > 8 and convection_possible_from_surface: cloud_type = "Cumulonimbus (Base Rugosa)"
    elif cape.m >= 1200 and convection_possible_from_surface: cloud_type = "Cumulonimbus (Multicèl·lula)"
    elif cape.m > 500 and cin.m < -75: cloud_type = "Castellanus"
    elif cape.m >= 800 and convection_possible_from_surface: cloud_type = "Cumulus Congestus"
    elif rh_0_4 > 0.85 and cape.m < 250 and pwat_0_4.m > 15: cloud_type = "Nimbostratus"
    elif cape.m >= 300 and convection_possible_from_surface: cloud_type = "Cumulus Mediocris"
    elif cape.m > 50 and convection_possible_from_surface:
        try:
            p_lcl_val = lcl_p.m if lcl_p else p[0].m - 100; p_cap_level = p_lcl_val - 50
            t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value='extrapolate')
            gradient = (t_interp(p_cap_level) - t_interp(p_lcl_val)) / (p_cap_level - p_lcl_val)
            cloud_type = "Cumulus Humilis" if gradient > 0 else "Cumulus Mediocris"
        except: cloud_type = "Cumulus Humilis"
    elif np.any(p.m < 400) and np.mean(mpcalc.relative_humidity_from_dewpoint(t[p.m < 400], td[p.m < 400])) > 0.7 and cape.m < 50: cloud_type = "Cirrus"
    elif np.any((p.m < 650) & (p.m > 400)) and np.mean(mpcalc.relative_humidity_from_dewpoint(t[(p.m < 650) & (p.m > 400)], td[(p.m < 650) & (p.m > 400)])) > 0.85 and cape.m < 100: cloud_type = "Altostratus / Altocumulus"
    elif cape.m > 5 and convection_possible_from_surface: cloud_type = "Cumulus Fractus"
    else: cloud_type = "Cel Serè"

    if cloud_type == "Cel Serè" and base_km and top_km and (top_km - base_km) > 0.05:
        cloud_type = "Cumulus Fractus"

    if "Supercèl·lula" in cloud_type or "Cumulonimbus" in cloud_type or "Congestus" in cloud_type or "Castellanus" in cloud_type:
        if lfc_h and base_km is not None and (lfc_h / 1000.0) > base_km:
            base_km = lfc_h / 1000.0
    
    st.subheader("Diagrama Skew-T", anchor=False)
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    st.divider()

    if is_sandbox_mode:
         chat_log, precipitation_type = generate_dynamic_analysis(p, t, td, ws, wd, cloud_type)
    else:
        chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)

    potential_clouds = determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Assistent d'Anàlisi", "📊 Paràmetres Detallats", "📈 Hodògraf", "☁️ Visualització de Núvols", "📋 Tipus de Núvols", "📡 Simulació Radar"])
    # ... (El contingut de les pestanyes roman igual)

# MODIFICAT: Pantalla de selecció principal simplificada
def show_province_selection_screen():
    set_main_background()
    fig_scape = create_city_mountain_scape()
    st.pyplot(fig_scape, use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>Anàlisi de Zones Meteorològiques</h2>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.button("Segueix la zona de canvis d'avui", on_click=lambda: st.session_state.update(province_selected='seguiment_menu'), use_container_width=True, type="primary")

# MODIFICAT: Pantalla de selecció de zona de seguiment actualitzada
def show_seguiment_selection_screen():
    st.title("Zona de Canvis d'Avui")
    st.markdown("Selecciona la comarca que vols analitzar. Cada zona representa un perfil atmosfèric diferent basat en les previsions més recents.")
    
    with st.sidebar:
        st.header("Controls")
        if st.button("⬅️ Tornar", use_container_width=True):
            st.session_state.province_selected = None
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🔥 Zona Més Destacable</h4><p>El perfil amb el major potencial per a fenòmens significatius.</p></div>""", unsafe_allow_html=True)
        if st.button("Pallars Jussà", use_container_width=True, type="primary"):
            st.session_state.province_selected = 'seguiment_destacable'
            st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>🤔 Zona Interessant</h4><p>Un perfil que presenta algunes característiques d'interès.</p></div>""", unsafe_allow_html=True)
        if st.button("Alt Urgell", use_container_width=True):
            st.session_state.province_selected = 'seguiment_interessant'
            st.rerun()

def run_single_sounding_mode(mode):
    seguiment_map = {
        'seguiment_destacable': {'file': 'sondeig_destacable.txt', 'title': "ZONA MÉS DESTACABLE", 'comarca': "Pallars Jussà"},
        'seguiment_interessant': {'file': 'sondeig_interessant.txt', 'title': "ZONA INTERESSANT", 'comarca': "Alt Urgell"}
    }
    
    config = seguiment_map[mode]
    comarca = config['comarca']
    
    st.title(f"{config['title']} - {comarca.upper()}")
    
    with st.sidebar:
        st.header("Controls")
        st.button("⬅️ Tornar a la selecció", use_container_width=True, on_click=lambda: st.session_state.update(province_selected='seguiment_menu'))

    content_placeholder = st.empty()
    with content_placeholder.container():
        show_loading_animation(message=f"Carregant {config['title']}")
        time.sleep(0.1) 

    try:
        soundings = parse_all_soundings(config['file'])
        content_placeholder.empty()

        if soundings:
            data = soundings[0]
            obs_time = data.get('observation_time', f"Sondeig de la {config['title'].lower()}")
            show_full_analysis_view(
                p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                obs_time=obs_time, 
                is_sandbox_mode=False
            )
        else:
            content_placeholder.empty()
            st.error(f"No s'han pogut carregar dades del sondeig '{config['file']}'.")
    
    except FileNotFoundError:
        content_placeholder.empty()
        st.error(f"L'arxiu '{config['file']}' no existeix. Aquest mode requereix que l'arxiu estigui present.")

def run_live_mode():
    selection = st.session_state.get('province_selected')

    if selection == 'seguiment_menu':
        show_seguiment_selection_screen()
    elif selection and selection.startswith('seguiment_'):
        run_single_sounding_mode(selection)
    else: 
        with st.sidebar:
            st.header("Controls")
            if st.button("⬅️ Tornar a l'inici", use_container_width=True):
                st.session_state.app_mode = 'welcome'
                if 'province_selected' in st.session_state: del st.session_state.province_selected
                st.rerun()
        show_province_selection_screen()

# =================================================================================
# === LABORATORI-TUTORIAL (Sense canvis en aquesta secció) ========================
# =================================================================================
# ... (El codi del Laboratori roman exactament igual) ...
def get_tutorial_data():
    return {
        'supercel': [
            {'action_id': 'warm_low', 'title': 'Pas 1: Escalfament superficial', 'instruction': "Necessitem energia. La manera més comuna és l'escalfament del sol durant el dia. Fes clic al botó de sota per escalfar les capes baixes.", 'button_label': "☀️ Escalfar Capa Baixa", 'explanation': "Això augmenta la temperatura a prop de la superfície, creant una 'bombolla' d'aire que voldrà ascendir."},
            {'action_id': 'moisten_low', 'title': 'Pas 2: Afegeix combustible', 'instruction': "Una tempesta necessita humitat per formar-se. Fes clic al botó per humitejar les capes baixes i apropar el punt de rosada a la temperatura.", 'button_label': "💧 Humitejar Capa Baixa", 'explanation': "Això fa que l'aire ascendent es condensi abans, alliberant calor latent i donant més força a la tempesta (augmentant el CAPE)."},
            {'action_id': 'add_shear_low', 'title': "Pas 3: Afegeix el motor de rotació", 'instruction': "L'ingredient secret d'una supercèl·lula és el cisallament del vent a nivells baixos. Fes clic al botó per afegir un canvi de vent amb l'altura.", 'button_label': "🌪️ Afegir Cisallament a Capes Baixes", 'explanation': "Això farà que el corrent ascendent de la tempesta comenci a rotar, organitzant-la i fent-la molt més potent i duradora."},
            {'action_id': 'conceptual', 'title': 'Pas 4: Anàlisi Final', 'instruction': "Ja tenim energia, humitat i rotació. Has creat un entorn perfecte per a la formació de supercèl·lules.", 'button_label': "Entès, finalitzar →", 'explanation': "A l'anàlisi final, fixa't en com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."},
        ],
        'aiguaneu': [
            {'action_id': 'conceptual', 'title': "Pas 1: La Fàbrica de Neu", 'instruction': "Hem carregat un perfil d'aiguaneu. Observa a les capes altes (sobre 700 hPa). Les temperatures són negatives. Aquí es formen els flocs de neu.", 'button_label': "Entès, pas 1/3 →", 'explanation': "Aquí és on es formen els flocs de neu inicials. De moment, tot correcte."},
            {'action_id': 'conceptual', 'title': "Pas 2: La Capa Càlida que ho fon tot", 'instruction': "Ara mira la capa mitjana (~850 hPa). La temperatura supera els 0°C. Aquest és el problema: els flocs es fonen i es converteixen en pluja.", 'button_label': "Ho veig, pas 2/3 →", 'explanation': "Quan els flocs de neu cauen a través d'aquesta capa càlida, es fonen i es converteixen en gotes de pluja."},
            {'action_id': 'conceptual', 'title': "Pas 3: Recongelació a Superfície", 'instruction': "Finalment, a prop de terra, la temperatura torna a ser negativa. Les gotes de pluja es tornen a congelar just abans de tocar el terra.", 'button_label': "Entès, pas 3/3 →", 'explanation': "Això és el que produeix l'aiguaneu (sleet) o la perillosa pluja gelant."},
            {'action_id': 'conceptual', 'title': 'Conclusió i Repte Final', 'instruction': "Has analitzat un perfil clàssic d'aiguaneu! Ara saps que una capa càlida intermèdia és la culpable.", 'button_label': "Finalitzar Tutorial", 'explanation': "Repte: Ara que has acabat, fes clic a 'Finalitzar'. Utilitza l'eina '❄️ Refredar Capa Mitjana' a la barra lateral i veuràs com converteixes aquest perfil en una nevada perfecta!"},
        ]
    }

def start_tutorial(scenario_name):
    st.session_state.sandbox_mode = 'tutorial'
    st.session_state.tutorial_active = True
    st.session_state.tutorial_scenario = scenario_name
    st.session_state.tutorial_step = 0
    if scenario_name == 'aiguaneu':
        profile_data = create_wintry_mix_profile()
    else:
        profile_data = st.session_state.sandbox_original_data
    st.session_state.sandbox_p_levels = profile_data['p_levels'].copy()
    st.session_state.sandbox_t_profile = profile_data['t_initial'].copy()
    st.session_state.sandbox_td_profile = profile_data['td_initial'].copy()
    st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
    st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()

def exit_tutorial():
    st.session_state.sandbox_mode = 'free'
    st.session_state.tutorial_active = False
    if 'tutorial_scenario' in st.session_state: del st.session_state['tutorial_scenario']
    if 'tutorial_step' in st.session_state: del st.session_state['tutorial_step']

def apply_profile_modification(action):
    t = st.session_state.sandbox_t_profile.m
    td = st.session_state.sandbox_td_profile.m
    p = st.session_state.sandbox_p_levels.m
    ws = st.session_state.sandbox_ws.to('m/s').m
    wd = st.session_state.sandbox_wd.m

    low_mask = p > 850
    mid_mask = (p <= 850) & (p > 600)
    high_mask = p <= 600

    if action == 'warm_low': t[low_mask] += 2.0
    elif action == 'cool_low': t[low_mask] -= 2.0
    elif action == 'moisten_low': td[low_mask] = np.minimum(t[low_mask] - 1.0, td[low_mask] + 2.0)
    elif action == 'dry_low': td[low_mask] -= 2.0
    elif action == 'warm_mid': t[mid_mask] += 2.0
    elif action == 'cool_mid': t[mid_mask] -= 4.0 
    elif action == 'moisten_mid': td[mid_mask] = np.minimum(t[mid_mask] - 1.5, td[mid_mask] + 2.0)
    elif action == 'dry_mid': td[mid_mask] -= 2.0
    elif action == 'warm_high': t[high_mask] += 2.0
    elif action == 'cool_high': t[high_mask] -= 2.0
    elif action == 'moisten_high': td[high_mask] = np.minimum(t[high_mask] - 2.0, td[high_mask] + 2.0)
    elif action == 'dry_high': td[high_mask] -= 2.0
    elif action == 'warm_all': t += 2.0
    elif action == 'cool_all': t -= 2.0
    elif action == 'moisten_all': td = np.minimum(t - 1.0, td + 2.0)
    elif action == 'dry_all': td -= 2.0
    elif action == 'add_inversion':
        inv_mask = (p < 950) & (p > 800)
        t[inv_mask] += 3.0
    elif 'shear' in action:
        if action == 'add_shear_low': mask = low_mask
        elif action == 'add_shear_mid': mask = mid_mask
        elif action == 'add_shear_high': mask = high_mask
        else: mask = np.full_like(p, True)
        
        num_points = np.sum(mask)
        if num_points > 0:
            ws[mask] += np.linspace(0, 15, num_points)
            ws = np.clip(ws, 0, 80)
            wd[mask] = (wd[mask] + np.linspace(0, 45, num_points)) % 360
        st.session_state.sandbox_ws = ws * units('m/s')
        st.session_state.sandbox_wd = wd * units.degrees

    td = np.minimum(t, td)
    st.session_state.sandbox_t_profile = t * units.degC
    st.session_state.sandbox_td_profile = td * units.degC

def show_tutorial_interface():
    tutorials = get_tutorial_data()
    scenario = st.session_state.tutorial_scenario
    step_index = st.session_state.tutorial_step
    steps = tutorials[scenario]
    
    st.title("🧪 Laboratori de Sondejos - Mode Tutorial")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"### Tutorial: {scenario.replace('_', ' ').title()}")
            st.markdown("---")
            if step_index >= len(steps):
                st.success("🎉 Enhorabona, has completat el tutorial! 🎉")
                st.markdown("El sondeig que has construït ja està a punt. Fes clic a 'Finalitzar' per veure'n l'anàlisi completa.")
                if st.button("Finalitzar i Veure Resultat", use_container_width=True, type="primary"):
                    exit_tutorial()
                    st.rerun()
            else:
                current_step = steps[step_index]
                st.markdown(f"#### {current_step['title']}")
                
                with st.container(border=True):
                    st.markdown(current_step['instruction'])
                    action_id = current_step['action_id']
                    
                    if st.button(current_step['button_label'], key=f"tut_action_{step_index}", use_container_width=True, type="primary"):
                        if action_id != 'conceptual':
                            apply_profile_modification(action_id)
                        st.session_state.tutorial_step += 1
                        st.rerun()
                st.markdown(f"*{current_step['explanation']}*")

        with col2:
            chat_log, _ = generate_tutorial_analysis(scenario, step_index)
            css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; height: 350px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>"""
            html_chat = "<h6>Assistent d'Anàlisi</h6><div class='chat-container'>"
            for speaker, message in chat_log:
                css_class = speaker.lower()
                html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            html_chat += "</div>"
            st.markdown(css_styles + html_chat, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("Abandonar Tutorial", use_container_width=True):
            exit_tutorial()
            st.rerun()

def show_sandbox_selection_screen():
    st.title("🧪 Benvingut al Laboratori!")
    st.markdown("Tria com vols començar. Pots seguir un tutorial guiat per aprendre els conceptes clau o anar directament al mode lliure per experimentar por tu mateix.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🌪️ Tutorial: Supercèl·lula</h4><p>Aprèn a crear un entorn amb una inestabilitat explosiva i el cisallament necessari per a les tempestes més severes i organitzades.</p></div>""", unsafe_allow_html=True)
        if st.button("Començar Tutorial de Supercèl·lula", use_container_width=True): 
            start_tutorial('supercel')
            st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>💧 Tutorial: Aiguaneu</h4><p>Analitza una situació d'aiguaneu, identifica la capa càlida culpable i aprèn com transformar la precipitació en neu.</p></div>""", unsafe_allow_html=True)
        if st.button("Començar Tutorial d'Aiguaneu", use_container_width=True): 
            start_tutorial('aiguaneu')
            st.rerun()
    with c3:
        st.markdown("""<div class="mode-card"><h4>🛠️ Mode Lliure</h4><p>Salta directament a l'acció. Tindràs el control total sobre el perfil atmosfèric des del principi per crear els teus propis escenaris.</p></div>""", unsafe_allow_html=True)
        if st.button("Anar al Mode Lliure", use_container_width=True, type="primary"):
            st.session_state.sandbox_mode = 'free'
            st.rerun()
    st.markdown("---")
    if st.button("⬅️ Tornar a l'inici"):
        st.session_state.app_mode = 'welcome'
        st.rerun()
        
def run_sandbox_mode():
    if 'sandbox_mode' not in st.session_state:
        st.session_state.sandbox_mode = 'selection'

    if 'sandbox_initialized' not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            show_loading_animation()
            time.sleep(0.5)
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings: 
            st.error("No s'ha trobat 'sondeigproves.txt'. Assegura't que el fitxer existeix.")
            placeholder.empty()
            return
        st.session_state.sandbox_original_data = soundings[0]
        data = st.session_state.sandbox_original_data
        st.session_state.sandbox_p_levels = data['p_levels'].copy()
        st.session_state.sandbox_t_profile = data['t_initial'].copy()
        st.session_state.sandbox_td_profile = data['td_initial'].copy()
        st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True
        st.session_state.convergence_active = False
        placeholder.empty()

    with st.sidebar:
        st.header("Caixa d'Eines")
        if st.button("⬅️ Tornar al Menú del Laboratori", use_container_width=True):
            for key in ['sandbox_mode', 'tutorial_active', 'tutorial_scenario', 'tutorial_step', 'convergence_active']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
        st.markdown("---")
        st.subheader("Modificacions Termodinàmiques")
        st.markdown("**Capes Baixes (> 850 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_low',), use_container_width=True); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_low',), use_container_width=True); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_low',), use_container_width=True); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_low',), use_container_width=True)
        st.markdown("**Capes Mitjanes (850-600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_mid',), use_container_width=True, key='w_mid'); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_mid',), use_container_width=True, key='c_mid'); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_mid',), use_container_width=True, key='m_mid'); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_mid',), use_container_width=True, key='d_mid')
        st.markdown("**Capes Altes (< 600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_high',), use_container_width=True, key='w_h'); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_high',), use_container_width=True, key='c_h'); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_high',), use_container_width=True, key='m_h'); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_high',), use_container_width=True, key='d_h')
        st.markdown("---"); st.subheader("Eines Globals i de Vent")
        c1, c2 = st.columns(2); c1.button("🔥 Escalfar Tot", on_click=apply_profile_modification, args=('warm_all',), use_container_width=True); c2.button("🧊 Refredar Tot", on_click=apply_profile_modification, args=('cool_all',), use_container_width=True)
        c1.button("💦 Humitejar Tot", on_click=apply_profile_modification, args=('moisten_all',), use_container_width=True); c2.button("🌬️ Assecar Tot", on_click=apply_profile_modification, args=('dry_all',), use_container_width=True)
        st.button("Tapadera (Inversió)", on_click=apply_profile_modification, args=('add_inversion',), use_container_width=True)
        st.markdown("**Cisallament del Vent**")
        c1, c2, c3 = st.columns(3); c1.button("🌪️ Baixes", on_click=apply_profile_modification, args=('add_shear_low',), use_container_width=True); c2.button("🌪️ Mitges", on_click=apply_profile_modification, args=('add_shear_mid',), use_container_width=True); c3.button("🌪️ Altes", on_click=apply_profile_modification, args=('add_shear_high',), use_container_width=True)
        def reset_wind_profile():
            st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.button("🚫 Reiniciar Vents", on_click=reset_wind_profile, use_container_width=True)
        
        st.markdown("---")
        if st.button("🔄 Reiniciar Tot al Perfil Original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_p_levels = data['p_levels'].copy(); st.session_state.sandbox_t_profile = data['t_initial'].copy(); st.session_state.sandbox_td_profile = data['td_initial'].copy()
            reset_wind_profile()
            if st.session_state.get('tutorial_active', False): 
                exit_tutorial()
            if 'convergence_active' in st.session_state:
                st.session_state.convergence_active = False
            st.rerun()

    if st.session_state.sandbox_mode == 'selection':
        show_sandbox_selection_screen()
    elif st.session_state.sandbox_mode == 'tutorial':
        show_tutorial_interface()
    elif st.session_state.sandbox_mode == 'free':
        st.title("🧪 Laboratori de Sondejos - Mode Lliure")
        show_full_analysis_view(
            p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile, 
            td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws, 
            wd=st.session_state.sandbox_wd, obs_time="Sondeig de Prova - Mode Laboratori",
            is_sandbox_mode=True
        )

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
