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

# El pany segueix sent crucial per evitar errors de concurrència quan es fan
# càlculs de veritat (p. ex., en canviar l'hora del sondeig).
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
    fig.patch.set_facecolor('#0b0f19') # Cel molt fosc
    ax.set_facecolor('#0b0f19')

    # Dibuixar estrelles
    star_x = np.random.uniform(0, 100, 200)
    star_y = np.random.uniform(15, 60, 200)
    star_s = np.random.uniform(0.5, 2.5, 200)
    star_alpha = np.random.uniform(0.5, 1, 200)
    ax.scatter(star_x, star_y, s=star_s, c='white', alpha=star_alpha, edgecolors='none')

    # Dibuixar la muntanya a la dreta
    mountain_poly = Polygon(
        [(55, 0), (68, 38), (75, 32), (85, 45), (95, 28), (100, 32), (100, 0)],
        facecolor='#12182c', edgecolor=None, zorder=5
    )
    ax.add_patch(mountain_poly)

    # Dibuixar la silueta de la ciutat a l'esquerra
    city_patches = []
    light_patches = []
    for x_base in np.arange(0, 70, 0.5):
        # La ciutat és més alta al centre
        height_factor = 1 - abs(x_base - 35) / 35
        building_height = (random.uniform(2, 12) * (1 + height_factor * 2))
        building_width = random.uniform(0.8, 3)
        
        # Color fosc per als edificis
        color_val = random.uniform(0.05, 0.1)
        color = (color_val, color_val, color_val)
        
        building = Rectangle((x_base, 0), building_width, building_height, facecolor=color, edgecolor=None, zorder=10)
        city_patches.append(building)
        
        # Afegir llums grogues aleatòries
        if random.random() < 0.08:
            light_x = x_base + random.uniform(0, building_width)
            light_y = random.uniform(1, building_height * 0.5)
            light = Circle((light_x, light_y), radius=0.15, color='#fde9a0', alpha=0.9)
            light_patches.append(light)
            
    ax.add_collection(PatchCollection(city_patches, match_original=True))
    ax.add_collection(PatchCollection(light_patches, match_original=True, zorder=11))

    # Definir límits i amagar eixos
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    return fig

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
    """Genera l'anàlisi conversacional per al mode 'Live', amb més diàleg i per a totes les condicions."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, _, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    chat_log = []

    # =================================================================
    # === NOU BLOC: ANÀLISI HIVERNAL SI T_sup < 7°C =====================
    # =================================================================
    if t_profile[0].m < 7.0:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'rain'

        chat_log.append(("Sistema", "Iniciant anàlisi de perfil hivernal (T < 7°C)."))
        chat_log.append(("Analista", f"D'acord, tenim una temperatura en superfície de {t_profile[0].m:.1f}°C. Això canvia les regles del joc. Ja no busquem tempestes, sinó que analitzem el potencial de neu."))
        chat_log.append(("Usuari", "Perfecte. Quins són els factors decisius per veure neu?"))
        
        p_array = p_levels.m
        t_array = t_profile.m
        
        # Analitzar si hi ha una capa càlida per sobre de la superfície
        warm_layer_mask = (p_array < p_array[0]) & (p_array > 700) & (t_array > 0.5)
        warm_layer_present = np.any(warm_layer_mask)
        
        if not warm_layer_present:
            chat_log.append(("Analista", "Bones notícies per als amants del fred. He revisat tota la columna d'aire i sembla que es manté per sota o molt a prop de 0°C en tot el recorregut de la possible precipitació."))
            
            if t_profile[0].m > 1.5:
                chat_log.append(("Usuari", f"Llavors, tot i que no hi ha capes càlides, la temperatura a la superfície ({t_profile[0].m:.1f}°C) no és massa alta?"))
                chat_log.append(("Analista", f"És una bona observació. Encara que no hi ha una capa càlida en altura que fongui la neu, una temperatura en superfície de {t_profile[0].m:.1f}°C pot fer que els flocs es fonguin just en arribar o donin una neu molt humida i de poca qualitat. La cota de neu estaria just per sobre de la nostra ubicació."))
                precipitation_type = 'rain'
            else:
                chat_log.append(("Usuari", "Això vol dir que si precipita, serà en forma de neu?"))
                chat_log.append(("Analista", "Exacte. Aquest és un 'perfil de nevada'. Significa que els flocs de neu que es formin en altura no es fondran durant la seva caiguda. Si hi ha precipitació, serà en forma de neu."))
                precipitation_type = 'snow'
        else: # Hi ha una capa càlida
            max_temp_in_layer = np.max(t_array[warm_layer_mask])
            chat_log.append(("Analista", f"Alerta! Aquí tenim el factor clau que sovint ens roba la neu a cotes baixes. He detectat una 'capa càlida' o 'nas càlid' en altura. La temperatura puja fins a {max_temp_in_layer:.1f}°C."))
            chat_log.append(("Usuari", "I això què significa exactament? Adéu a la neu?"))
            
            if t_profile[0].m <= 0.5:
                chat_log.append(("Analista", "Aquesta capa càlida fon els flocs de neu, convertint-los en gotes de pluja. Però com que la capa just a prop del terra està per sota de 0°C, aquestes gotes es tornen a congelar abans de tocar el terra."))
                chat_log.append(("Analista", "El resultat més probable és **aiguaneu (sleet)**. En casos molt concrets, si la capa freda superficial és molt prima, podríem tenir la perillosa **pluja gelant**."))
                precipitation_type = 'sleet'
            else: # T_sup > 0.5°C
                 chat_log.append(("Analista", "Exactament. Aquesta capa càlida fon la neu i la converteix en pluja. Com que la temperatura a la superfície, encara que freda ({t_profile[0].m:.1f}°C), és positiva, la precipitació arribarà en forma de pluja freda. És el típic escenari de 'plou i fa fred'."))
                 precipitation_type = 'rain'

    # =================================================================
    # === BLOC ANTERIOR: ANÀLISI DE TEMPESTES (SI T_sup >= 7°C) ========
    # =================================================================
    else:
        if "Tornàdica" in cloud_type or "Tuba" in cloud_type or "Mur" in cloud_type: precipitation_type = 'hail'
        elif cape.m > 500: precipitation_type = 'rain'
        elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
        
        chat_log = [("Sistema", f"Iniciant anàlisi conversacional per a l'escenari de {cloud_type}.")]

        if "Tornàdica" in cloud_type:
            chat_log.extend([
                ("Analista", "ALERTA MÀXIMA. El perfil no només és de supercèl·lula, sinó que presenta característiques tornàdiques clàssiques."),
                ("Usuari", "Què ho fa tan perillós?"),
                ("Analista", f"Tenim tres ingredients clau alineats. Primer, una base del núvol molt baixa, a només {lcl_h:.0f} metres del terra. Això és crucial."),
                ("Analista", f"Segon, una rotació a nivells baixos extremadament forta, amb un SRH de 0-1km de {srh_0_1:.0f} m²/s². Això és el 'motor' del tornado."),
                ("Analista", "Aquests factors, combinats amb l'energia i el cisallament generals, creen un entorn d'alt risc on la rotació de la tempesta té moltes probabilitats d'arribar a terra."),
            ])
        elif "Tuba/Funnel" in cloud_type:
            chat_log.extend([
                ("Analista", "Molt de compte. Aquest és un perfil de supercèl·lula amb un alt potencial per desenvolupar embuts."),
                ("Usuari", "Què indica aquest potencial?"),
                ("Analista", f"La combinació d'una helicitat significativa a nivells baixos (SRH 0-1km: {srh_0_1:.0f} m²/s²) i una base de núvol relativament baixa ({lcl_h:.0f} m)."),
                ("Analista", "Això vol dir que la rotació de la tempesta té molta facilitat per baixar i condensar-se en una tuba visible. És el pas previ a un tornado i requereix vigilància constant.")
            ])
        elif "Mur de núvols" in cloud_type:
            chat_log.extend([
                ("Analista", "Aquest és un perfil de supercèl·lula clàssic i molt organitzat."),
                ("Usuari", "Què el fa especial?"),
                ("Analista", f"La rotació a nivells mitjans és molt intensa, com indica el SRH 0-3km de {srh_0_3:.0f} m²/s². Aquesta forta rotació pot provocar una baixada localitzada de la base de la tempesta, formant el que coneixem com a mur de núvols (wall cloud)."),
                ("Analista", "Aquest mur de núvols és la regió principal des d'on es poden originar els tornados. És un senyal visual inequívoc d'una supercèl·lula potent.")
            ])
        elif "Shelf Cloud" in cloud_type:
             chat_log.extend([
                ("Analista", "Atenció. Aquest perfil és perillós, però per un motiu diferent a la rotació."),
                ("Usuari", "No és una supercèl·lula?"),
                ("Analista", f"No exactament. Tot i que tenim molta energia (CAPE: {cape.m:.0f} J/kg), la clau aquí no és la rotació (SRH baix), sinó el potencial per a una corrent descendent (downdraft) molt violenta."),
                ("Analista", "Aquest aire fred i dens s'espargeix en arribar a terra i aixeca l'aire càlid que té davant, formant un impressionant núvol de prestatge o shelf cloud. El principal perill aquí són els vents lineals destructius (reventón o downburst).")
            ])
        elif "Base Rugosa" in cloud_type:
            chat_log.extend([
                ("Analista", "Tenim una tempesta forta en marxa."),
                ("Usuari", "Quin és el detall més significatiu?"),
                ("Analista", f"L'energia és alta (CAPE: {cape.m:.0f} J/kg) i hi ha una forta entrada d'aire humit a nivells baixos (inflow). Això fa que es condensi humitat per sota de la base principal, creant una base rugosa o amb fragments de núvols (scud)."),
                ("Analista", "És un senyal que la tempesta està 'respirant' bé i s'està alimentant amb força. Encara no mostra signes clars de rotació organitzada, però és una tempesta a vigilar.")
            ])
        elif "Supercèl·lula" in cloud_type:
            chat_log.extend([
                ("Analista", "Aquest és un perfil de manual per a temps sever. Anem a desglossar-lo."),
                ("Usuari", f"Veig molta energia (CAPE: {cape.m:.0f} J/kg) i cisallament ({shear_0_6:.1f} m/s)."),
                ("Analista", "Exacte. Aquesta combinació crea un motor d'alt rendiment que permet que una tempesta s'organitzi i comenci a rotar. Tot i que no mostra els trets més extrems per a tornados, és una supercèl·lula en tota regla."),
                ("Analista", "El pronòstic ha de ser de precaució per calamarsa gran i vents forts.")
            ])
        elif cloud_type == "Nimbostratus":
            chat_log.extend([
                ("Analista", "Aquest perfil és molt diferent. Aquí la història no va d'inestabilitat explosiva."),
                ("Usuari", f"És cert, el CAPE és gairebé inexistent, només {cape.m:.0f} J/kg."),
                ("Analista", "Exacte. El protagonista aquí és la humitat. Tenim una capa d'aire molt gruixuda i completament saturada, des de baix fins a nivells mitjans."),
                ("Usuari", "Llavors, la pluja serà més constant que en una tempesta?"),
                ("Analista", f"Sí. Aquest és un escenari típic de pluja estratiforme, associada a sistemes frontals. La intensitat dependrà de l'aigua precipitable, que amb {pwat_0_4.m:.1f} mm, ens indica que podem esperar pluges persistents i generalitzades.")
            ])
        elif cloud_type == "Cirrus":
            chat_log.extend([
                ("Analista", "Interessant. El perfil està molt sec a les capes baixes i mitjanes."),
                ("Usuari", "Llavors no hi haurà núvols?"),
                ("Analista", "No exactament. Si mires a gran altura, per sobre dels 6 o 7 km, veuràs una fina capa on la humitat augmenta."),
                ("Usuari", "I això què forma?"),
                ("Analista", "Això crea les condicions ideals per als Cirrus. Són núvols alts, formats per vidres de gel. No produeixen precipitació, però de vegades són el preludi d'un canvi de temps.")
            ])
        elif cloud_type == "Altostratus / Altocumulus":
            chat_log.extend([
                ("Analista", "Tenim un cas de nuvolositat a nivells mitjans."),
                ("Usuari", f"Per què? Veig que no hi ha gairebé CAPE ({cape.m:.0f} J/kg)."),
                ("Analista", "Correcte, la convecció des de superfície està totalment inhibida. No obstant, observa la marcada capa d'humitat entre els 3 i 6 km aproximadament."),
                ("Usuari", "Ah, ho veig, les línies de temperatura i punt de rosada s'ajunten en aquesta zona."),
                ("Analista", "Exacte. Això formarà una capa de núvols mitjans, com Altostratus o Altocumulus, que poden arribar a cobrir el cel i donar un aspecte tapat o aigualit al sol.")
            ])
        elif cloud_type == "Cumulus Humilis":
            chat_log.extend([
                ("Analista", "Estem observant un escenari de temps estable."),
                ("Usuari", f"Però hi ha una mica de CAPE, {cape.m:.0f} J/kg."),
                ("Analista", "Sí, una mica d'energia hi ha, suficient per formar núvols, però molt poca. A més, segurament hi ha una forta inversió tèrmica (una 'tapadera') just per sobre que impedeix qualsevol creixement."),
                ("Analista", "Això és un perfil típic per a la formació de Cumulus Humilis, els clàssics 'núvols de bon temps' que no produeixen precipitació.")
            ])
        elif cloud_type == "Cumulus Mediocris":
            chat_log.extend([
                ("Analista", "Aquest és un perfil interessant per a una tarda d'estiu."),
                ("Usuari", f"Tenim {cape.m:.0f} J/kg de CAPE. És suficient per a tempestes?"),
                ("Analista", "És una energia moderada. Permet un cert creixement vertical, però no explosiu. El cisallament del vent també és feble."),
                ("Analista", "Això afavoreix la formació de Cumulus Mediocris. Són els típics núvols de cotó fluix amb una base plana, que rarament donen més que quatre gotes.")
            ])
        elif cloud_type == "Cumulus Congestus":
            chat_log.extend([
                ("Analista", "Atenció a aquest perfil. Aquí comencem a veure potencial per a fenòmens més actius."),
                ("Usuari", f"El CAPE ja és més considerable, {cape.m:.0f} J/kg."),
                ("Analista", "Exacte. Tenim prou energia per a un desenvolupament vertical important. Aquests núvols creixen amb força cap amunt."),
                ("Analista", "És l'escenari ideal per a Cumulus Congestus, també coneguts com a 'torres cumuliformes'. Són el pas previ al Cumulonimbus i ja poden deixar ruixats o xàfecs localment intensos.")
            ])
        elif cloud_type == "Cumulonimbus (Multicèl·lula)":
            chat_log.extend([
                ("Analista", "Bé, tenim un escenari amb potencial de tempestes. El primer, com sempre, és l'energia disponible."),
                ("Usuari", f"El CAPE és de {cape.m:.0f} J/kg."),
                ("Analista", f"És un bon valor, suficient per a tempestes fortes, possiblement amb calamarsa. A més, el LFC ({lfc_h:.0f} m) és prou baix per permetre que la convecció arrenqui."),
                ("Usuari", "I s'organitzaran? Com és el cisallament?"),
            ])
            shear_analysis_message = ""
            if shear_0_6 < 10:
                shear_analysis_message = f"Aquí ve el matís. El cisallament del vent ({shear_0_6:.1f} m/s) és feble. Les tempestes seran probablement desorganitzades i de cicle de vida curt, típic de les multicèl·lules."
            elif shear_0_6 < 18:
                shear_analysis_message = f"Aquí ve el matís. El cisallament del vent ({shear_0_6:.1f} m/s) és moderat. Això permetrà que les tempestes s'organitzin en sistemes multicel·lulars més duradors, però no és suficient per a la rotació sostinguda d'una supercèl·lula."
            else:
                 shear_analysis_message = f"El cisallament ({shear_0_6:.1f} m/s) és fort. Encara que no arriba als llindars de supercèl·lula clàssica, hi ha una organització considerable. Les tempestes seran robustes."
            
            chat_log.append(("Analista", shear_analysis_message))
        elif cloud_type == "Castellanus":
            chat_log.extend([
                ("Analista", "Aquest és un cas particular. Tenim energia en altura, però la superfície està desconnectada."),
                ("Usuari", f"Què vols dir? El CAPE és de {cape.m:.0f} J/kg."),
                ("Analista", f"Sí, però fixa't en el CIN: és molt fort, de {cin.m:.0f} J/kg. Això impedeix que la convecció comenci des del terra. No obstant, hi ha una capa inestable a nivells mitjans."),
                ("Analista", "Això pot generar núvols de tipus Altocumulus Castellanus, que són com petites torretes que creixen des d'una base elevada i poden donar lloc a xàfecs sobtats i ratxes de vent.")
            ])
        elif cloud_type == "Cumulus Fractus":
             chat_log.extend([
                ("Analista", "El que veiem aquí són condicions residuals."),
                ("Usuari", "Què vol dir això?"),
                ("Analista", "Hi ha una mica d'humitat i inestabilitat, però és molt poca i desorganitzada. No hi ha prou força per crear núvols ben definits."),
                ("Analista", "Això només permetrà la formació de Cumulus Fractus, que són trossos de núvols esquinçats, sense un desenvolupament clar. No tenen cap mena de risc associat.")
            ])
        else: # Cel Serè
            chat_log.extend([
                ("Analista", "El perfil atmosfèric és molt estable."),
                ("Usuari", "Llavors, no veurem cap núvol?"),
                ("Analista", f"És molt poc probable. Amb un CAPE de només {cape.m:.0f} J/kg, no hi ha pràcticament gens d'energia per al creixement vertical. Tindrem un dia de cel serè o amb alguns núvols alts sense importància.")
            ])

    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type):
    """Genera anàlisi conversacional per al mode laboratori, amb més diàleg."""
    cape, cin, _, lcl_h, _, lfc_h, _, _, _ = calculate_thermo_parameters(p, t, td)
    shear_0_6, _, _, _ = calculate_storm_parameters(p, ws, wd)
    chat_log = []
    
    chat_log.append(("Analista", "Molt bé, anem a analitzar el perfil que has creat. Ho farem com si fóssim un equip, pas a pas. Comencem?"))

    if cape.m < 50:
        chat_log.extend([
            ("Usuari", "Tenim potencial per a tempestes?"),
            ("Analista", f"Ara mateix no. L'energia disponible, el CAPE, és de només {cape.m:.0f} J/kg. L'atmosfera està molt estable.")
        ])
    else:
        chat_log.extend([
            ("Usuari", "Què estic creant amb aquesta energia?"),
        ])
        cloud_mention = f"Això és un escenari típic per a la formació de {cloud_type}."
        if cloud_type == "Cel Serè":
             cloud_mention = "Encara que hi ha energia, la tapadera és tan forta que probably no veuríem cap núvol significatiu."
        chat_log.append(("Analista", f"Has generat un CAPE de {cape.m:.0f} J/kg. {cloud_mention}"))

        chat_log.append(("Usuari", "I la 'tapadera' (CIN)? Com afecta?"))
        if cin.m < -100:
            chat_log.append(("Analista", f"Molt forta. Amb un CIN de {cin.m:.0f} J/kg, l'atmosfera està blindada. És com tenir una tapa d'olla a pressió. La convecció des de superfície és gairebé impossible, necessitaria un forçament extern massiu."))
        elif cin.m < -50:
            chat_log.append(("Analista", f"És considerable, amb {cin.m:.0f} J/kg. Les tempestes de superfície són poc probables, però obre la porta a la convecció de base elevada (Castellanus)."))
        elif cin.m < -25:
            chat_log.append(("Analista", f"És moderada ({cin.m:.0f} J/kg). Permet que l'energia s'acumuli a sota abans de disparar-se, un escenari clàssic per a tempestes fortes."))
        else:
             chat_log.append(("Analista", f"És feble ({cin.m:.0f} J/kg). La convecció té gairebé via lliure per iniciar-se."))
        
        if cin.m > -100 and cape.m > 800:
            chat_log.append(("Usuari", "He modificat el vent. Com afecta?"))
            if shear_0_6 > 18:
                chat_log.append(("Analista", f"El cisallament és fort ({shear_0_6:.1f} m/s). Aquest és l'ingredient clau que pot fer que les tempestes rotin, organitzant-les en supercèl·lules."))
            elif shear_0_6 > 10:
                chat_log.append(("Analista", f"El cisallament és moderat ({shear_0_6:.1f} m/s). Ajuda a organitzar les tempestes en sistemes multicel·lulars i a fer-les més duradores."))
            else:
                chat_log.append(("Analista", f"El cisallament és feble ({shear_0_6:.1f} m/s). Si es formen tempestes, probablement seran més desorganitzades i de vida més curta."))

    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Genera l'anàlisi del xat per a un pas específic d'un tutorial."""
    chat_log = []
    if scenario == 'aiguaneu':
        if step == 0:
            chat_log.append(("Analista", "Benvingut! Anem a analitzar un perfil clàssic d'aiguaneu. L'objectiu és entendre per què no neva tot i fer fred."))
            chat_log.append(("Usuari", "Perfecte. Què és el primer que he de mirar?"))
            chat_log.append(("Analista", "Observa la 'fàbrica de neu' a les capes altes. Com pots veure, per sobre de 700 hPa fa prou fred per formar flocs de neu. Fins aquí, tot correcte."))
        elif step == 1:
            chat_log.append(("Analista", "Molt bé. Ara ve la part clau. Fixa't en la capa al voltant de 850 hPa. La temperatura puja per sobre dels 0°C."))
            chat_log.append(("Usuari", "Això és la 'capa càlida', oi? Què provoca?"))
            chat_log.append(("Analista", "Exacte. Aquesta capa càlida actua com un 'bufador'. Quan els flocs de neu que cauen la travessen, es fonen i es converteixen en gotes de pluja."))
        elif step == 2:
            chat_log.append(("Analista", "Ja gairebé ho tenim. Ara tenim gotes de pluja caient cap a la superfície. Però mira la temperatura a prop del terra..."))
            chat_log.append(("Usuari", "Torna a estar per sota de 0°C!"))
            chat_log.append(("Analista", "Precisament! Aquestes gotes de pluja es tornen a congelar just abans d'arribar a terra, convertint-se en petites boletes de gel. Això és l'aiguaneu (sleet)."))
        elif step == 3:
            chat_log.append(("Analista", "Has analitzat el perfil a la perfecció. Has vist que per tenir neu, no n'hi ha prou amb fred a la superfície, tota la columna d'aire ha de ser coherent."))
            chat_log.append(("Usuari", "Entès. Llavors, com ho podria convertir en una nevada?"))
            chat_log.append(("Analista", "Aquest és el repte! Ara, quan finalitzis el tutorial, ves al Mode Lliure i utilitza l'eina '❄️ Refredar Capa Mitjana'. Veuràs com elimines la capa càlida i el perfil es converteix en una nevada perfecta."))
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem el tutorial de supercèl·lula. El primer pas és sempre crear energia. Necessitem un dia càlid d'estiu. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "Correcte! Molta calor. Ara, afegim el combustible: la humitat. A l'anàlisi final veuràs com augmenta el valor de CAPE quan les línies de temperatura i punt de rosada s'acosten."))
        elif step == 2: chat_log.append(("Analista", "Fantàstic! Has afegit cisallament. Aquest és l'ingredient secret que fa que les tempestes rotin. Ara tenim energia, humitat i rotació: la recepta perfecta!"))
        elif step == 3: chat_log.append(("Analista", "Missió complerta! Has creat un perfil amb molta energia (CAPE alt), humitat i cisallament. A l'anàlisi final, fixa't en com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."))

    return chat_log, None
    
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    if sfc_temp.m < 7.0: # Llindar de temperatura per anàlisi hivernal
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
        else: # T_sfc > 0.5 i < 7.0
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades. Extremi les precaucions.", "dodgerblue"
            else:
                 return "AMBIENT FRED I HUMIT", "Condicions de fred. La precipitació seria en forma de pluja o neu molt humida.", "steelblue"

    if cape.m >= 1200:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)

        if cin.m <= -100:
            return "CONVECCIÓ FORTAMENT INHIBIDA", f"Potencial energètic (CAPE {cape.m:.0f} J/kg) bloquejat per una 'tapadera' molt forta (CIN {cin.m:.0f} J/kg).", "darkslategray"
        
        if -100 < cin.m <= -50:
            return "POSSIBLE CONVECCIÓ DE MITJÀ NIVELL", f"La convecció des de superfície és difícil (CIN {cin.m:.0f} J/kg). Risc de nuclis elevats.", "slategray"

        title = "AVÍS PER TEMPESTES SEVERES"
        color = "darkorange"
        message = f"CAPE: {cape.m:.0f} J/kg. "

        if srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18:
            title, color = "AVÍS PER TORNADO", "darkred"
            message += f"Alt risc de tornados (SRH 0-1km: {srh_0_1:.0f}, LCL: {lcl_h:.0f}m)."
        elif srh_0_3 > 250 and shear_0_6 > 18:
            title, color = "AVÍS PER TEMPS SEVER", "purple"
            message += f"Supercèl·lules probables. Risc de calamarsa gran i/o murs de núvols (SRH 0-3km: {srh_0_3:.0f})."
        elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150):
            title, color = "AVÍS PER VENTS FORTS", "saddlebrown"
            message += "Risc de ratxes de vent lineals severes (downbursts/shelf cloud)."
        elif cape.m > 2500:
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
                if pwat_layer.m > 25:
                    return "AVÍS PER PLUGES INTENSES", "(Activa el forçament) Risc de pluges persistents i fortes.", "darkblue"
                elif pwat_layer.m > 15:
                    return "AVÍS PER PLUJA MODERADA", "Cel cobert amb pluja contínua i moderada.", "steelblue"
                else:
                    return "PREVISIÓ DE PLUJA FEBLE", "(Activa el forçament) S'esperen plugims o ruixats febles.", "cadetblue"
    except Exception:
        pass

    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

def determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p):
    """
    Determina una llista de possibles tipus de núvols basant-se en les condicions del sondeig.
    Aquesta lògica es basa en la taula proporcionada per l'usuari i les seves especificacions.
    """
    potential_clouds = []
    
    try:
        # Assegurar que hi ha prou punts de dades
        if len(p) < 2:
            return ["Dades insuficients"]
            
        heights = mpcalc.pressure_to_height_std(p).to('m')
        rh = mpcalc.relative_humidity_from_dewpoint(t, td)
        
        # Funció per interpolar la temperatura a un nivell de pressió donat
        t_interp_func = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")

        # NÚVOLS ALTS (Gènere: Cirrus)
        mask_high = (heights.m > 6000) & (heights.m < 18000)
        if np.any(mask_high) and np.sum(mask_high) > 1:
            rh_high = rh[mask_high]
            if np.mean(rh_high) > 0.85:
                potential_clouds.append("Cirrostratus")
            elif np.mean(rh_high) > 0.80:
                potential_clouds.append("Cirrocumulus")
            elif np.mean(rh_high) > 0.75:
                potential_clouds.append("Cirrus")

        # NÚVOLS MITJANS (Gènere: Alto)
        mask_mid = (heights.m > 2000) & (heights.m < 7000)
        if np.any(mask_mid) and np.sum(mask_mid) > 1:
            # Altostratus (més humit, sense inestabilitat)
            mask_as = (heights.m > 2000) & (heights.m < 6000)
            if np.any(mask_as) and np.sum(mask_as) > 1 and np.mean(rh[mask_as]) > 0.90 and cape.m < 50:
                 potential_clouds.append("Altostratus")
            # Altocumulus (menys humit, una mica d'inestabilitat permesa)
            elif (0.70 < np.mean(rh[mask_mid]) < 0.90) and cape.m <= 100:
                potential_clouds.append("Altocumulus")

        # NÚVOLS BAIXOS (Gènere: Stratus)
        mask_low = (heights.m < 2500)
        if np.any(mask_low) and np.sum(mask_low) > 1:
            rh_low = rh[mask_low]
            # Stratus (molt baix, molt humit, estable)
            mask_st = (heights.m < 1500)
            if np.any(mask_st) and np.sum(mask_st) > 1 and np.mean(rh[mask_st]) > 0.95 and cape.m < 10:
                potential_clouds.append("Stratus (Boira ascendent)")
            # Stratocumulus (una mica més alt, humit, convecció limitada)
            elif np.mean(rh_low) > 0.85 and cape.m <= 50:
                 potential_clouds.append("Stratocumulus")

        # NÚVOLS DE PRECIPITACIÓ
        mask_nimbostratus = (heights.m > 500) & (heights.m < 5000)
        if np.any(mask_nimbostratus) and np.sum(mask_nimbostratus) > 1 and np.mean(rh[mask_nimbostratus]) > 0.95 and cape.m <= 100:
            potential_clouds.append("Nimbostratus")
            
        # NÚVOLS DE DESENVOLUPAMENT VERTICAL (Convectius)
        has_convective_potential = cape.m > 100 and lcl_h is not None and lcl_h > 0

        if has_convective_potential:
            # Cas 1: LFC baix (< 3000m) -> Convecció de base superficial
            if lfc_h is not None and lfc_h < 3000:
                # Cumulus (requereix LFC > LCL)
                if (100 < cape.m < 2500) and (lfc_h > lcl_h):
                    potential_clouds.append("Cumulus (Humilis, Mediocris o Congestus)")

                # Cumulonimbus (requereix CAPE > 1000 i cim de gel)
                if cape.m > 1000:
                    is_iced_top = False
                    if el_p is not None and not np.isnan(el_p.m):
                        try:
                            t_at_el = t_interp_func(el_p.m)
                            if t_at_el < 0:
                                is_iced_top = True
                        except:
                            pass # Ignorar si la interpolació falla
                    if is_iced_top:
                        potential_clouds.append("Cumulonimbus")
            
            # Cas 2: LFC alt (>= 3000m) o no definit -> Convecció de base elevada
            else: # (lfc_h is None or lfc_h >= 3000 or lfc_h == np.inf)
                # Comprovar si hi ha humitat a nivells mitjans per formar Castellanus
                mask_mid_castellanus = (heights.m > 2000) & (heights.m < 7000)
                if np.any(mask_mid_castellanus) and np.sum(mask_mid_castellanus) > 1:
                    if np.mean(rh[mask_mid_castellanus]) > 0.60: # Llindar d'humitat per Castellanus
                         potential_clouds.append("Altocumulus Castellanus")

        # Neteja de duplicats o sub-tipus redundants
        final_clouds = []
        has_cb = "Cumulonimbus" in potential_clouds
        has_cu = "Cumulus (Humilis, Mediocris o Congestus)" in potential_clouds
        has_ns = "Nimbostratus" in potential_clouds
        has_castellanus = "Altocumulus Castellanus" in potential_clouds
        
        for cloud in potential_clouds:
            if cloud == "Cumulus (Humilis, Mediocris o Congestus)" and has_cb:
                continue # No afegir Cumulus si ja hi ha Cumulonimbus
            if cloud == "Stratocumulus" and (has_cb or has_cu or has_ns):
                continue # No afegir Stratocumulus si hi ha núvols convectius o de pluja més significatius
            if cloud == "Altocumulus" and has_castellanus:
                 continue # Prioritzar Castellanus sobre Altocumulus si es compleixen les condicions
            if cloud not in final_clouds:
                final_clouds.append(cloud)

        if not final_clouds:
            return ["Cel Serè o Núvols residuals"]
            
        return sorted(list(set(final_clouds))) # Retorna llista única i ordenada
        
    except Exception as e:
        return [f"No s'ha pogut determinar la nuvolositat. Error: {e}"]

# =========================================================================
# === 3. FUNCIONS DE DIBUIX ===============================================
# =========================================================================
def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if cape.m <= 0 or not lcl_p:
        return None, None
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
                        if rh[idx] < 0.7: 
                            p_top = p_levels[idx]
                            break
                cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
            except:
                cloud_top_km = cloud_base_km
    return (cloud_base_km, cloud_top_km) if cloud_base_km is not None and cloud_top_km is not None and cloud_top_km > cloud_base_km else (None, None)
def _get_cloud_color(y, base, top, b_min=0.6, b_max=0.95):
    if top <= base: return (b_min,) * 3
    return (np.clip(b_min + (b_max-b_min)*((y-base)/(top-base))**0.7,0,1),)*3
def _draw_cumulonimbus(ax, base_km, top_km):
    updraft_center_x, num_points = 0, 20
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    if len(tower_indices) == 0: tower_indices = np.arange(len(altitudes))
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    widths += np.random.uniform(-0.05, 0.05, len(tower_indices))
    r_pts = [(updraft_center_x + widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    l_pts = [(updraft_center_x - widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    main_poly_pts = [(l_pts[0][0], l_pts[0][1])] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d8d8d8', lw=0, zorder=10))
    for _ in range(120):
        idx = random.randint(1, len(tower_alts) - 1)
        y = tower_alts[idx] + random.uniform(-0.3, 0.3)
        max_x_at_y = np.interp(y, tower_alts, widths, left=widths[0], right=widths[-1])
        x = updraft_center_x + random.uniform(-max_x_at_y, max_x_at_y)
        size = random.uniform(0.2, 0.6) * (1 + (y - base_km) / (top_km - base_km))
        brightness = np.clip(0.85 + 0.15 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
        ax.add_patch(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.1, 0.35), lw=0, zorder=11))
    anvil_altitudes = np.linspace(anvil_base_alt, top_km, 10)
    anvil_spread = 1.5 + random.uniform(-0.2, 0.2)
    for _ in range(80):
        y = random.uniform(anvil_base_alt, top_km)
        height_factor = 1 + (y - anvil_base_alt) / (top_km - anvil_base_alt)
        x = updraft_center_x + random.uniform(-anvil_spread * height_factor, anvil_spread * height_factor)
        width = random.uniform(0.5, 1.2) * height_factor
        height = random.uniform(0.05, 0.15)
        color = tuple([random.uniform(0.95, 1.0)]*3)
        ax.add_patch(Ellipse((x, y), width, height, facecolor=color, alpha=random.uniform(0.1, 0.3), lw=0, zorder=12))
def _draw_cumulus_mediocris(ax, base_km, top_km):
    center_x = 0
    num_particles = 250
    cloud_height = top_km - base_km
    altitudes = np.linspace(base_km, top_km, 20)
    base_width = 0.4 * (1 + 0.8 * np.sin(np.pi * (altitudes - base_km) / (cloud_height + 0.01)))
    noise = np.random.uniform(-0.1, 0.1, len(altitudes))
    widths = base_width + noise
    widths[0] = max(widths[0], 0.3)
    r_pts = [(center_x + widths[i], altitudes[i]) for i in range(len(altitudes))]
    l_pts = [(center_x - widths[i], altitudes[i]) for i in range(len(altitudes))]
    main_poly_pts = [l_pts[0]] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d0d0d0', lw=0, zorder=10))
    patches = []
    for _ in range(num_particles):
        y_progress = random.betavariate(2, 2)
        y = base_km + y_progress * cloud_height
        max_x_at_y = np.interp(y, altitudes, widths)
        x = center_x + random.uniform(-max_x_at_y, max_x_at_y) * 0.95
        size = random.uniform(0.15, 0.5) * (1 + y_progress * 0.5)
        min_bright, max_bright = 0.8, 1.0
        brightness = min_bright + (max_bright - min_bright) * (y_progress ** 0.7)
        color = (brightness, brightness, brightness)
        alpha = random.uniform(0.15, 0.45)
        patch = Circle((x, y), size, facecolor=color, alpha=alpha, lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=11))
def _draw_cumulus_castellanus(ax, base_km, top_km):
    base_thickness = min(0.8, (top_km - base_km) * 0.25)
    patches_base = []
    for _ in range(120):
        x = random.uniform(-1.7, 1.7)
        y = base_km + (random.random() ** 2) * base_thickness
        b = random.uniform(0.8, 0.9)
        patch = Ellipse((x, y), width=random.uniform(0.7, 1.6), height=random.uniform(0.1, 0.25), facecolor=(b, b, b), alpha=random.uniform(0.1, 0.3), lw=0)
        patches_base.append(patch)
    ax.add_collection(PatchCollection(patches_base, match_original=True, zorder=8))
    num_turrets = random.randint(3, 5)
    turret_base_y = base_km + base_thickness * 0.5
    for i in range(num_turrets):
        turret_center_x = random.uniform(-1.3, 1.3)
        turret_top_y = turret_base_y + random.uniform(0.5, 0.95) * (top_km - turret_base_y)
        turret_height = turret_top_y - turret_base_y
        max_width = random.uniform(0.25, 0.4)
        patches_turret = []
        for _ in range(random.randint(60, 90)):
            y = turret_base_y + (random.random() ** 0.8) * turret_height
            normalized_y_in_turret = (y - turret_base_y) / turret_height
            current_width = max_width * np.sin(np.pi * normalized_y_in_turret)
            x = turret_center_x + random.uniform(-current_width * 0.9, current_width * 0.9)
            size = random.uniform(0.1, 0.3) * (1 + normalized_y_in_turret * 0.5)
            brightness = 0.75 + (0.98 - 0.75) * (normalized_y_in_turret ** 0.8)
            patch = Circle((x, y), size, facecolor=(brightness, brightness, brightness), alpha=random.uniform(0.2, 0.5), lw=0)
            patches_turret.append(patch)
        ax.add_collection(PatchCollection(patches_turret, match_original=True, zorder=9 + i))
def _draw_nimbostratus(ax, base_km, top_km, cloud_type):
    color, alpha = '#a9a9a9', 0.9
    ax.add_patch(Rectangle((-1.7, base_km), 3.4, top_km - base_km, facecolor=color, lw=0, zorder=8, alpha=alpha))
    patches = []
    for _ in range(150):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.6, 0.75)
        patch = Ellipse((x, y), width=random.uniform(0.8, 1.5), height=random.uniform(0.1, 0.3), facecolor=(b, b, b), alpha=random.uniform(0.2, 0.4), lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))
def _draw_cumulus_fractus(ax, base_km, thickness):
    patches=[Ellipse((random.gauss(0,0.5),random.uniform(base_km,base_km+thickness)), random.uniform(0.2,0.4), random.uniform(0.3,0.7)*random.uniform(0.2,0.4), angle=random.uniform(-25,25), facecolor=_get_cloud_color(random.uniform(base_km,base_km+thickness),base_km,base_km+thickness,b_min=0.6,b_max=0.8), alpha=0.5,lw=0) for _ in range(150)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))
def _draw_stratiform_cotton_clouds(ax, base_km, top_km):
    patches = []
    for _ in range(200):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.88, 0.98)
        patch = Ellipse((x, y), random.uniform(0.4, 0.9), random.uniform(0.15, 0.3), facecolor=(b, b, b), alpha=random.uniform(0.3, 0.6), lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))
def _draw_clear_sky(ax):
    patches = [Ellipse((random.uniform(-1.5,1.5), random.uniform(10,14)), random.uniform(0.5,1.0), random.uniform(0.1,0.2), facecolor='white', alpha=random.uniform(0.05,0.1), lw=0) for _ in range(15)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=5))
def _draw_precipitation(ax, precip_base_km, ground_km, p_type, center_x=0.0, sub_cloud_rh=0.4):
    if p_type == 'virga':
        alpha = np.clip(sub_cloud_rh * 0.6, 0.15, 0.55)
        fall_percentage = sub_cloud_rh / 0.5
        fall_distance = (precip_base_km - ground_km) * fall_percentage
        end_y = precip_base_km - fall_distance
        if sub_cloud_rh < 0.5: end_y = max(end_y, ground_km + 0.3)
        else: end_y = ground_km
        top_width = random.uniform(0.6, 0.9)
        bottom_width = top_width * 0.5
        points = [(center_x - top_width / 2, precip_base_km), (center_x + top_width / 2, precip_base_km), (center_x + bottom_width / 2, end_y), (center_x - bottom_width / 2, end_y)]
        ax.add_patch(Polygon(points, facecolor='cornflowerblue', alpha=alpha, lw=0, zorder=7))
    elif p_type in ['rain', 'sleet']:
        width = 1.6
        ax.add_patch(Rectangle((center_x - width / 2, ground_km), width, precip_base_km - ground_km, facecolor='cornflowerblue', alpha=0.35, lw=0, zorder=5))
    elif p_type == 'hail':
        ax.scatter(center_x+np.random.normal(0,0.3,150),np.random.uniform(ground_km,precip_base_km,150), s=np.random.uniform(5,40,150),c='white',alpha=0.8,marker='o',edgecolor='gray',linewidth=0.5,zorder=8)
    elif p_type == 'snow':
        ax.scatter(center_x+np.random.normal(0,0.5,300),np.random.uniform(ground_km,precip_base_km,300), s=np.random.uniform(20,70,300),c='white',alpha=np.random.uniform(0.4,0.9,300),marker='*',zorder=8)
def _draw_saturation_layers(ax, p_levels, t_profile, td_profile):
    try:
        saturated_indices = np.where(t_profile.m-td_profile.m <= 1.5)[0]
        if not len(saturated_indices): return
        i=0
        while i < len(saturated_indices):
            start_idx, j = saturated_indices[i], i
            while j+1 < len(saturated_indices) and saturated_indices[j+1]==saturated_indices[j]+1: j+=1
            end_idx = saturated_indices[j]
            h_bottom = mpcalc.pressure_to_height_std(p_levels[start_idx]).to('km').m
            h_top = mpcalc.pressure_to_height_std(p_levels[end_idx]).to('km').m
            if h_top - h_bottom < 0.05: i=j+1; continue
            patches=[]
            for _ in range(int(100+300*(h_top-h_bottom))):
                y, x = random.uniform(h_bottom,h_top), random.uniform(-1.5,1.5)
                brightness = random.uniform(0.65,0.85)
                patches.append(Ellipse((x,y),random.uniform(0.3,0.8),random.uniform(0.05,0.1)*(1+h_top-h_bottom), facecolor=(brightness,)*3,alpha=random.uniform(0.1,0.5),lw=0))
            ax.add_collection(PatchCollection(patches, match_original=True, zorder=7))
            i=j+1
    except Exception: pass
def _draw_base_feature(ax, f_type, base_x_left, base_x_right, base_y, ground_y):
    z, center_x, width = 12, (base_x_left + base_x_right) / 2, base_x_right - base_x_left
    if f_type == 'lowering':
        ax.add_patch(Polygon([(base_x_left, base_y), (base_x_right, base_y), (base_x_right * 0.9 + center_x * 0.1, base_y - 0.2), (base_x_left * 0.9 + center_x * 0.1, base_y - 0.2)], facecolor='dimgray', edgecolor='gray', zorder=z))
    elif f_type == 'wall_cloud':
        top_l, top_r = center_x - (width * 0.75 / 2), center_x + (width * 0.75 / 2)
        bot_l, bot_r = center_x - (width * 0.55 / 2), center_x + (width * 0.55 / 2)
        ax.add_patch(Polygon([(top_l, base_y), (top_r, base_y), (bot_r, base_y - 0.35), (bot_l, base_y - 0.35)], facecolor='#383838', edgecolor='#202020', lw=0.5, zorder=z))
    elif f_type == 'funnel':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, max(base_y - 0.8, ground_y + 0.5))], facecolor='darkgray', alpha=0.8, zorder=z))
    elif f_type == 'tornado':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, ground_y)], facecolor='#505050', zorder=z))
        ax.add_patch(Ellipse((center_x, ground_y + 0.05), width=0.7, height=0.25, facecolor='#654321', alpha=0.7, zorder=z + 1))
def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100)
    ax.set_xlim(-50, 45)

    td_profile = np.minimum(t_profile, td_profile)

    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
        skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
        parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
        skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
        wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
        skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='Tª Bombolla Humida')
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
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set(ylim=(0,16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud (km)"); ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5); ax.set_facecolor('#6495ED')
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//' if ground_color=='#228B22' else ''))
    
    if not convergence_active:
        _draw_saturation_layers(ax, p_levels, t_profile, td_profile)
    if base_km is not None and top_km is not None and (top_km - base_km > 0.1):
        if "Nimbostratus" in cloud_type or "Hivernal" in cloud_type:
            _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif "Altostratus" in cloud_type:
            _draw_stratiform_cotton_clouds(ax, base_km, top_km)
        elif "Cirrus" in cloud_type:
            _draw_clear_sky(ax) # Dibuixa cirrus fins
        elif "Supercèl·lula" in cloud_type or "Cumulonimbus" in cloud_type:
            _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus":
            _draw_cumulus_castellanus(ax, base_km, top_km)
        elif cloud_type in ["Cumulus Mediocris", "Cumulus Congestus", "Cumulus Humilis"]:
            _draw_cumulus_mediocris(ax, base_km, top_km)
        elif cloud_type == "Cumulus Fractus":
            cloud_thickness = top_km - base_km
            _draw_cumulus_fractus(ax, base_km, cloud_thickness)
    elif not np.any((t_profile.m - td_profile.m) <= 1.5):
        _draw_clear_sky(ax)
    if precipitation_type and base_km is not None:
        is_castellanus = (cloud_type == "Castellanus")
        precip_base_km = lfc_h / 1000.0 if is_castellanus and lfc_h > 0 else base_km
        sub_cloud_rh_mean = 0.4
        try:
            p_base_precip = mpcalc.height_to_pressure_std(precip_base_km * units.kilometer)
            p_ground = p_levels[0]
            sub_cloud_mask = (p_levels >= p_base_precip) & (p_levels <= p_ground)
            if np.any(sub_cloud_mask):
                rh_profile = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile)
                sub_cloud_rh_mean = np.mean(rh_profile[sub_cloud_mask]).magnitude
        except Exception:
            sub_cloud_rh_mean = 0.4
        _draw_precipitation(ax, precip_base_km, ground_height_km, precipitation_type, sub_cloud_rh=sub_cloud_rh_mean)
    plt.tight_layout()
    return fig


def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active):
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_shear = fig.add_subplot(gs[0, 1], sharey=ax)
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10); ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitud (km)", xticks=[]); ax.grid(True, linestyle='--', alpha=0.3)
    ax_shear.set(xlim=(-1, 1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    if not base_km or not top_km or cape.m < 5 or not convergence_active:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva\n(Activa el forçament per simular-la)", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='white', bbox=dict(facecolor='darkblue', alpha=0.7))
        ax_shear.axis('off'); return fig
        
    visual_base_km = max(base_km, ground_height_km + 0.5)
    try:
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        h_km = mpcalc.pressure_to_height_std(p_levels).to('km').m
        unique_h, idx = np.unique(h_km, return_index=True)
        if len(unique_h) < 2: return fig
        f_u, f_v = interp1d(unique_h, u.m[idx], bounds_error=False, fill_value='extrapolate'), interp1d(unique_h, v.m[idx], bounds_error=False, fill_value='extrapolate')
        barb_heights = np.arange(0, min(20, h_km.max()), 1)
        ax_shear.barbs(np.zeros_like(barb_heights), barb_heights, (f_u(barb_heights) * units('m/s')).to('knots').m, (f_v(barb_heights) * units('m/s')).to('knots').m, length=7, pivot='middle', color='k')
        altitudes = np.linspace(visual_base_km, top_km, num=50)
        u_at_alts = f_u(altitudes)
        horizontal_offsets = u_at_alts * 0.02
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        shear_factor = np.clip(shear_0_6 / 35, 0.4, 2.5)
        updraft_widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (altitudes - visual_base_km) / (top_km - visual_base_km + 0.01))) * shear_factor
        anvil_extension = np.zeros_like(altitudes)
        if (top_km - visual_base_km) > 4.0:
            anvil_base_alt = top_km * 0.80
            anvil_indices = np.where(altitudes >= anvil_base_alt)[0]
            if len(anvil_indices) > 0:
                u_anvil_top = f_u(top_km)
                wind_direction = np.sign(u_anvil_top) if u_anvil_top != 0 else 1
                max_stretch = abs(u_anvil_top) * 0.06
                growth_factor = (altitudes[anvil_indices] - anvil_base_alt) / (top_km - anvil_base_alt)
                anvil_extension[anvil_indices] = max_stretch * wind_direction * growth_factor**1.5
        r_pts = [(updraft_widths[i] + horizontal_offsets[i] + anvil_extension[i], altitudes[i]) for i in range(len(altitudes))]
        l_pts = [(-updraft_widths[i] + horizontal_offsets[i], altitudes[i]) for i in range(len(altitudes))]
        ax.add_patch(Polygon(r_pts + l_pts[::-1], facecolor='white', edgecolor='lightgray', alpha=0.95, zorder=10))
        _, _, lcl_p, lcl_h, _, _, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
        feature = None
        if top_km - base_km > 4.0 and cape.m > 500:
            if (srh_0_1 >= 150 and lcl_h <= 1000 and shear_0_6 > 15): feature = 'tornado'
            elif (srh_0_1 > 100 and lcl_h < 1200 and shear_0_6 > 12): feature = 'funnel'
            elif srh_0_3 > 150 and shear_0_6 > 18 and cape.m > 1000: feature = 'wall_cloud'
            elif s_0_1 > 8 and lcl_h < 1500: feature = 'lowering'
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_height_km)
    except Exception as e: pass
    plt.tight_layout()
    return fig

def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray'); ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.grid(True, linestyle=':', alpha=0.3, color='white')
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p_levels).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_layer = mpcalc.relative_humidity_from_dewpoint(t_profile[layer_mask], td_profile[layer_mask])
            pwat_layer = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask]).to('mm')
            rh_mean_layer = np.mean(rh_layer)
            if rh_mean_layer > 0.85 and cape.magnitude < 350:
                x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
                max_dbz = np.clip(15 + pwat_layer.m, 15, 45)
                noise = gaussian_filter(np.random.randn(100, 100), sigma=8) * (max_dbz * 0.2)
                Z = max_dbz + noise
                Z = np.clip(Z, 0, 50)
                radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900']
                radar_levels = [0, 15, 20, 25, 30, 35, 45]
                radar_cmap = ListedColormap(radar_colors)
                radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
                ax.contourf(x, y, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
                return fig
    except Exception:
        pass
    if cape.m < 100:
        ax.text(0, 0, "Sense precipitació significativa", ha='center', va='center', color='white', fontsize=9)
        return fig
    shear_0_6, *_ = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    _, _, lcl_p, _, lfc_p, _, el_p, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    mean_u, mean_v = (0,0) * units('m/s')
    if lfc_p and el_p:
        p_mask = (p_levels >= el_p) & (p_levels <= lfc_p)
        if np.sum(p_mask) > 1:
            u, v = mpcalc.wind_components(wind_speed[p_mask], wind_dir[p_mask])
            mean_u, mean_v = np.mean(u), np.mean(v)
    max_dbz = np.clip(20 + (cape.m / 3000) * 55, 20, 75)
    elongation = np.clip(1 + (shear_0_6 / 20), 1, 2.5)
    angle_rad = np.arctan2(mean_u.m, mean_v.m)
    x, y = np.linspace(-50, 50, 150), np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    x_rot, y_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad), -xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    sigma_x, sigma_y = 15, 15 / elongation
    Z = max_dbz * np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
    Z += gaussian_filter(np.random.randn(150, 150), sigma=6) * (max_dbz * 0.1); Z = np.clip(Z, 0, 75)
    radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
    radar_cmap = ListedColormap(radar_colors)
    radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    return fig
def create_hodograph_figure(p, ws, wd, t, td):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=40.)
    h.add_grid(increment=10, ls='--', color='gray')
    ax.set_xlabel('kt')
    ax.set_ylabel('kt')
    
    try:
        p_hodo = p.to('hPa')
        ws_hodo = ws.to('kt')
        wd_hodo = wd.to('deg')
        
        u, v = mpcalc.wind_components(ws_hodo, wd_hodo)
        heights = mpcalc.pressure_to_height_std(p_hodo).to('km')
        
        h_interp = np.arange(0, min(12, heights.m.max()), 0.1) * units.km
        u_interp = np.interp(h_interp.m, heights.m, u.m) * units.kt
        v_interp = np.interp(h_interp.m, heights.m, v.m) * units.kt

        levels = [0, 1, 3, 5, 8, 10]
        colors = ['green', 'orange', 'red', 'purple', 'darkviolet']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, cmap.N)
        
        for i in range(len(h_interp) - 1):
            ax.plot(u_interp[i:i+2].m, v_interp[i:i+2].m, color=cmap(norm(h_interp[i].m)), linewidth=2)
        
        with integrator_lock:
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p_hodo, u, v, heights)
        
        ax.arrow(0, 0, rm[0].m, rm[1].m, color='black', width=0.5, head_width=2, length_includes_head=True, label="Moviment Tempesta (MD)")
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, pad=0.08)
        cbar.set_label('Altitud (km)')
        
    except Exception as e:
        ax.text(0.5, 0.5, "Dades de vent insuficients\nper generar hodògraf.", 
                ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    return fig

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
        st.markdown("""<div class="mode-card"><h3>🛰️Temps real</h3><p>Visualitza els sondejos atmosfèrics més recents basats en dades de models. Navega entre les diferents execucions horàries disponibles.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode temps real", use_container_width=True):
            st.session_state.app_mode = 'live'
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
        "Activar Forçament (Convergència)",
        key='convergence_active',
        help="Simula l'efecte d'un mecanisme de tret (p.ex. convergència o orografia). Si està activat, els núvols creixeran fins al seu topall teòric (EL) si hi ha CAPE, ignorant la inhibició (CIN). Si no, només es formaran en capes ja saturades o si la convecció pot vèncer el CIN per si sola."
    )
    convergence_active = st.session_state.get('convergence_active', False)

    # Càlcul de paràmetres
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
    convection_possible_from_surface = (cin.m > -100 and lfc_h < 3000)

    # Lògica de determinació del tipus de núvol principal
    if sfc_temp.m < 7.0:
        cloud_type = "Hivernal"
    elif cape.m > 1500 and srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18 and convection_possible_from_surface:
        cloud_type = "Supercèl·lula (Tornàdica)"
    elif cape.m > 1500 and srh_0_1 > 120 and lcl_h < 1200 and shear_0_6 > 18 and convection_possible_from_surface:
        cloud_type = "Supercèl·lula (Tuba/Funnel)"
    elif cape.m > 1800 and srh_0_3 > 250 and shear_0_6 > 18 and convection_possible_from_surface:
        cloud_type = "Supercèl·lula (Mur de núvols)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150 and convection_possible_from_surface:
        cloud_type = "Supercèl·lula"
    elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150):
        cloud_type = "Cumulonimbus (Shelf Cloud)"
    elif cape.m > 1200 and s_0_1 > 8 and convection_possible_from_surface:
        cloud_type = "Cumulonimbus (Base Rugosa)"
    elif cape.m >= 1200 and convection_possible_from_surface:
        cloud_type = "Cumulonimbus (Multicèl·lula)"
    elif cape.m > 500 and cin.m < -75:
        cloud_type = "Castellanus"
    elif cape.m >= 800 and convection_possible_from_surface:
        cloud_type = "Cumulus Congestus"
    elif rh_0_4 > 0.85 and cape.m < 250 and pwat_0_4.m > 15:
        cloud_type = "Nimbostratus"
    elif cape.m >= 300 and convection_possible_from_surface:
        cloud_type = "Cumulus Mediocris"
    elif cape.m > 50 and convection_possible_from_surface:
        try:
            p_lcl_val = lcl_p.m if lcl_p else p[0].m - 100; p_cap_level = p_lcl_val - 50
            t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value='extrapolate')
            gradient = (t_interp(p_cap_level) - t_interp(p_lcl_val)) / (p_cap_level - p_lcl_val)
            cloud_type = "Cumulus Humilis" if gradient > 0 else "Cumulus Mediocris"
        except: cloud_type = "Cumulus Humilis"
    elif np.any(p.m < 400) and np.mean(mpcalc.relative_humidity_from_dewpoint(t[p.m < 400], td[p.m < 400])) > 0.7 and cape.m < 50:
        cloud_type = "Cirrus"
    elif np.any((p.m < 650) & (p.m > 400)) and np.mean(mpcalc.relative_humidity_from_dewpoint(t[(p.m < 650) & (p.m > 400)], td[(p.m < 650) & (p.m > 400)])) > 0.85 and cape.m < 100:
        cloud_type = "Altostratus / Altocumulus"
    elif cape.m > 5 and convection_possible_from_surface:
        cloud_type = "Cumulus Fractus"
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

    # Crida a la nova funció per obtenir la llista de núvols
    potential_clouds = determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Assistent d'Anàlisi", "📊 Paràmetres Detallats", "📈 Hodògraf", "☁️ Visualització de Núvols", "📋 Tipus de Núvols", "📡 Simulació Radar"])
    
    with tab1:
        # ===== CSS DEL XAT RESTAURAT =====
        css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>"""
        html_chat = "<div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower()
            html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)

        # ===== DICCIONARI D'IMATGES ACTUALITZAT =====
        image_triggers = {
            "tornado": ("tornado.jpg", "Un tornado format sota una supercèl·lula."),
            "tornàdica": ("tornado.jpg", "Un tornado format sota una supercèl·lula."),
            "tuba": ("funnel.jpg", "Una tuba (funnel cloud) baixant de la base del núvol."),
            "mur de núvols": ("wallcloud.jpg", "Un mur de núvols (wall cloud) ben definit."),
            "shelf cloud": ("shelfcloud.jpg", "Un espectacular núvol de prestatge (shelf cloud)."),
            "base rugosa": ("scud.jpg", "Base rugosa amb fragments de núvols (scud)."),
            "supercèl·lula": ("supercell.jpg", "Una supercèl·lula organitzada."),
            "castellanus": ("castellanus.jpg", "Això és un Altocumulus Castellanus."),
            "fractus": ("fractus.jpg", "Això és un Cumulus Fractus."),
            "cumulonimbus": ("cumulonimbus.jpg", "Això és un Cumulonimbus."),
            "congestus": ("congestus.jpg", "Això és un Cumulus Congestus."),
            "mediocris": ("mediocris.jpg", "Això és un Cumulus Mediocris."),
            "humilis": ("humilis.jpg", "Això és un Cumulus Humilis."),
            "cirrus": ("cirrus.jpg", "Aquests són núvols Cirrus."),
            "altostratus": ("altostratus.jpg", "Aquest és un cel cobert per Altostratus."),
            "aiguaneu": ("sleet.jpg", "Precipitació en forma d'aiguaneu (sleet)."),
            "neu": ("snow.jpg", "Una nevada cobrint el paisatge.")
        }
        images_to_show = set() 
        full_chat_text = " ".join([msg for _, msg in chat_log]).lower()
        full_chat_text += " " + cloud_type.lower() # Afegeix el títol del nùvol per a la cerca
        for keyword, (filename, caption) in image_triggers.items():
            if keyword in full_chat_text:
                images_to_show.add((filename, caption))

        if images_to_show:
            st.markdown("---")
            for filename, caption in sorted(list(images_to_show)):
                image_base64 = get_image_as_base64(filename)
                if image_base64:
                    st.markdown(f"<div style='margin-top: 15px; text-align: center;'><img src='{image_base64}' style='max-width: 80%; border-radius: 10px;'><p style='font-style: italic; color: grey;'>{caption}</p></div>", unsafe_allow_html=True)
                else:
                    st.warning(f"S'ha mencionat '{keyword}', però no s'ha trobat el fitxer '{filename}'.", icon="🖼️")
    
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_h:.0f} m"); param_cols[1].metric("LFC", f"{lfc_h:.0f} m" if lfc_h != np.inf else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); param_cols[3].metric("Shear 0-6km", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1km", f"{srh_0_1:.1f} m²/s²"); param_cols[1].metric("SRH 0-3km", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm")
        rh_display = "N/A"
        try:
            rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("RH Mitja 0-4km", rh_display)
    with tab3:
        st.subheader("Hodògraf del Perfil de Vents")
        fig_hodo = create_hodograph_figure(p, ws, wd, t, td)
        st.pyplot(fig_hodo, use_container_width=True)
    with tab4:
        precipitation_type_visual = precipitation_type
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type_visual, lfc_h, cape, base_km, top_km, cloud_type)
            st.pyplot(fig_clouds, use_container_width=True)
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(p, t, td, ws, wd, convergence_active)
            st.pyplot(fig_structure, use_container_width=True)
    with tab5:
        st.subheader("Llista de Gèneres de Núvols Probables")
        st.markdown("Aquesta llista es genera automàticament analitzant les capes d'humitat, inestabilitat i temperatura del sondeig. Múltiples tipus de núvols poden coexistir a diferents altituds.")
    
        if potential_clouds:
            for cloud in potential_clouds:
                st.markdown(f"- **{cloud}**")
        else:
            st.info("Segons l'anàlisi, no s'espera formació de núvols significatius.")
    
        st.markdown("---")
        st.caption("Aquesta anàlisi es basa en la interpretació automàtica d'un únic perfil vertical i no té en compte factors sinòptics a gran escala com els fronts o la advecció, que són crucials per a un pronòstic complet.")

    with tab6:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)

def show_province_selection_screen():
    # Estableix un fons genèric fosc per a tota la pàgina primer
    set_main_background()
    
    # Dibuixa l'escena de la ciutat i la muntanya
    fig_scape = create_city_mountain_scape()
    st.pyplot(fig_scape, use_container_width=True)

    st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>Selecciona una Província</h2>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    
    with col:
        def select_barcelona():
            st.session_state.province_selected = 'barcelona'
        st.button("Barcelona", on_click=select_barcelona, use_container_width=True, type="primary")
        
        # S'ha ajustat l'estil per millorar la llegibilitat sobre el fons
        st.markdown(
            """
            <div style="text-align: center; margin-top: 25px; padding: 15px; background-color: rgba(0, 0, 0, 0.5); border-radius: 10px;">
                <p style="color: #cccccc; font-weight: bold; margin-bottom: 5px;">Pròximament...</p>
                <p style="color: #a0a0a0; margin: 0;">Tarragona • Lleida • Girona</p>
                <p style="color: #a0a0a0; margin-top: 5px;">i més!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_countdown_timer():
    madrid_tz = ZoneInfo("Europe/Madrid")
    now = datetime.now(madrid_tz)
    run_times_spec = [dt_time(0, 0), dt_time(5, 0), dt_time(12, 0)]
    today = now.date()
    possible_runs = [datetime.combine(today, t, tzinfo=madrid_tz) for t in run_times_spec]
    
    next_run_time = None
    for run_dt in possible_runs:
        if now < run_dt:
            next_run_time = run_dt
            break
            
    if next_run_time is None:
        tomorrow = today + timedelta(days=1)
        next_run_time = datetime.combine(tomorrow, run_times_spec[0], tzinfo=madrid_tz)

    target_timestamp_ms = int(next_run_time.timestamp() * 1000)

    st.markdown("---")
    countdown_html = f"""
    <div style="text-align: center;">
        <span style="font-size: 0.9em;">Pròxima actualització ({next_run_time.strftime('%H:%Mh')}):</span>
        <p id="countdown-timer" style="font-size: 1.6em; font-weight: bold; color: #FFC300; margin:0; line-height:1.2;"></p>
    </div>

    <script>
    function updateCountdown() {{
        const target_ms = {target_timestamp_ms};
        const now_ms = new Date().getTime();
        const timeLeft_ms = target_ms - now_ms;

        if (timeLeft_ms < 0) {{
            document.getElementById("countdown-timer").innerHTML = "Actualitzant...";
            return;
        }}

        let hours = Math.floor((timeLeft_ms % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        let minutes = Math.floor((timeLeft_ms % (1000 * 60 * 60)) / (1000 * 60));
        let seconds = Math.floor((timeLeft_ms % (1000 * 60)) / 1000);

        hours = hours < 10 ? '0' + hours : hours;
        minutes = minutes < 10 ? '0' + minutes : minutes;
        seconds = seconds < 10 ? '0' + seconds : seconds;

        document.getElementById("countdown-timer").innerHTML = hours + ":" + minutes + ":" + seconds;
    }}
    updateCountdown();
    setInterval(updateCountdown, 1000);
    </script>
    """
    st.markdown(countdown_html, unsafe_allow_html=True)
    st.markdown("---")


def run_live_mode():
    if st.session_state.get('province_selected') == 'barcelona':
        st.title("BARCELONA")
        
        with st.sidebar:
            st.header("Controls")
            
            def back_to_selection():
                st.session_state.province_selected = None

            st.button("⬅️ Tornar a la selecció", use_container_width=True, on_click=back_to_selection)
            display_countdown_timer()
            st.subheader("Selecciona una hora")

        if 'live_initialized' not in st.session_state:
            base_files = [f"{h:02d}h.txt" for h in range(24)]
            st.session_state.existing_files = sorted([f for f in base_files if os.path.exists(f)])

            if not st.session_state.existing_files:
                st.error("No s'ha trobat cap arxiu de sondeig. Assegura't que els arxius (p.ex. 09h.txt) existeixen.")
                return

            madrid_tz = ZoneInfo("Europe/Madrid")
            now = datetime.now(madrid_tz)
            current_hour_file = f"{now.hour:02d}h.txt"
            
            st.session_state.current_hour = now.hour

            initial_file = current_hour_file if current_hour_file in st.session_state.existing_files else st.session_state.existing_files[-1]
            st.session_state.selected_file = initial_file

            st.session_state.live_initialized = True
            st.session_state.convergence_active = False
            st.rerun()

        content_placeholder = st.empty()

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
            try:
                current_index = st.session_state.existing_files.index(st.session_state.selected_file)
            except ValueError:
                current_index = 0

            selected_file = st.radio(
                "Hores disponibles:",
                st.session_state.existing_files,
                index=current_index,
                format_func=format_time_for_display,
                key='time_selector'
            )

            if selected_file != st.session_state.selected_file:
                st.session_state.selected_file = selected_file
                st.rerun()
                
        with content_placeholder.container():
            show_loading_animation(message="Carregant Skew-T")
            time.sleep(0.1) 

        try:
            soundings = parse_all_soundings(st.session_state.selected_file)
            content_placeholder.empty()

            if soundings:
                data = soundings[0]
                show_full_analysis_view(
                    p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                    ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                    obs_time=data.get('observation_time', 'Hora no disponible'), 
                    is_sandbox_mode=False
                )
            else:
                st.error(f"No s'han pogut carregar dades de {st.session_state.selected_file}")
        
        except FileNotFoundError:
            content_placeholder.empty()
            st.error(f"L'arxiu '{st.session_state.selected_file}' no existeix.")
            if st.session_state.existing_files:
                st.session_state.selected_file = st.session_state.existing_files[0]
                st.rerun()

    else:
        # La pantalla de selecció de província ara té el seu propi fons
        with st.sidebar:
            st.header("Controls")
            if st.button("⬅️ Tornar a l'inici", use_container_width=True):
                st.session_state.app_mode = 'welcome'
                if 'province_selected' in st.session_state:
                    del st.session_state.province_selected
                st.rerun()
        show_province_selection_screen()

# =================================================================================
# === LABORATORI-TUTORIAL =========================================================
# =================================================================================

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
        st.markdown("""<div class="mode-card"><h4>💧 Tutorial: Aiguaneu</h4><p>Analitza una situació d'aiguaneu a BCN per exemple , identifica la capa càlida culpable i aprèn com transformar la precipitació en neu.</p></div>""", unsafe_allow_html=True)
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
