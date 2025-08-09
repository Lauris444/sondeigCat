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

# Le verrou reste crucial pour éviter les erreurs de concurrence.
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FONCTIONS DE STYLE ET PRÉSENTATION ===================================
# =============================================================================

def show_loading_animation(message="Chargement"):
    """Affiche une animation de chargement personnalisée avec HTML et CSS."""
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
# === 1. FONCTIONS DE CHARGEMENT ET TRAITEMENT DES DONNÉES ===================
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
    # Ajout de mots-clés en français pour l'extraction de l'heure
    time_keywords = ['observation', 'heure', 'time', 'locale', 'run', 'z', 'date']
    for line in block_lines:
        line_strip = line.strip()
        # Ignorer l'heure entre parenthèses et le mot "locale"
        if '(' in line_strip and ')' in line_strip:
            line_strip = re.sub(r'\([^)]*\)', '', line_strip).strip()
        if 'locale' in line_strip.lower():
            continue
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
            st.warning(f"Avertissement : Erreur lors du traitement de la ligne '{line_strip}'. Erreur : {e}")
            continue
    if not p_list or len(p_list) < 2: return None
    observation_time = "\n".join(time_lines) if time_lines else "Heure non disponible"
    sorted_indices = np.argsort(p_list)[::-1]
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
        # Tenter de lire avec l'encodage 'latin-1' qui est plus courant pour les textes français avec accents
        with open(filepath, 'r', encoding='latin-1') as f: lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{filepath}' est introuvable. Assurez-vous qu'il existe dans le même répertoire.")
        return []
    except Exception as e:
        st.error(f"Erreur à la lecture du fichier '{filepath}' : {e}")
        return []
        
    for line in lines:
        if 'Pression' in line and (line.strip().startswith('Niveau') or line.strip().startswith('# Niveau')):
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
# === 2. FONCTIONS DE CALCUL ET D'ANALYSE =====================================
# =========================================================================

def calculate_thermo_parameters(p_levels, t_profile, td_profile):
    with integrator_lock:
        try:
            p, t, td = p_levels, t_profile, td_profile
            valid_indices = ~np.isnan(p.magnitude) & ~np.isnan(t.magnitude) & ~np.isnan(td.magnitude)
            if np.sum(valid_indices) < 2: raise ValueError("Pas assez de données valides.")
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
    """Génère l'analyse conversationnelle pour le mode 'Live'."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, _, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    chat_log = []

    if t_profile[0].m < 7.0:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'rain'
        chat_log.append(("Système", "Lancement de l'analyse du profil hivernal (T < 7°C)."))
        chat_log.append(("Analyste", f"D'accord, nous avons une température en surface de {t_profile[0].m:.1f}°C. Cela change les règles du jeu. Nous ne cherchons plus des orages, mais nous analysons le potentiel de neige."))
        chat_log.append(("Utilisateur", "Parfait. Quels sont les facteurs décisifs pour voir de la neige ?"))
        p_array, t_array = p_levels.m, t_profile.m
        warm_layer_mask = (p_array < p_array[0]) & (p_array > 700) & (t_array > 0.5)
        warm_layer_present = np.any(warm_layer_mask)
        if not warm_layer_present:
            chat_log.append(("Analyste", "Bonnes nouvelles. J'ai examiné toute la colonne d'air et il semble qu'elle reste en dessous ou très proche de 0°C sur tout le parcours."))
            if t_profile[0].m > 1.5:
                chat_log.append(("Utilisateur", f"Alors, la température à la surface ({t_profile[0].m:.1f}°C) n'est-elle pas trop élevée ?"))
                chat_log.append(("Analyste", f"C'est une bonne observation. Une température de {t_profile[0].m:.1f}°C peut faire fondre les flocons juste en arrivant ou donner une neige très humide."))
                precipitation_type = 'rain'
            else:
                chat_log.append(("Utilisateur", "Cela signifie que s'il y a des précipitations, ce sera sous forme de neige ?"))
                chat_log.append(("Analyste", "Exactement. C'est un 'profil de chute de neige'. S'il y a des précipitations, ce sera sous forme de neige."))
                precipitation_type = 'snow'
        else:
            max_temp_in_layer = np.max(t_array[warm_layer_mask])
            chat_log.append(("Analyste", f"Alerte ! J'ai détecté une 'couche chaude' en altitude. La température monte jusqu'à {max_temp_in_layer:.1f}°C."))
            chat_log.append(("Utilisateur", "Et qu'est-ce que cela signifie exactement ? Adieu la neige ?"))
            if t_profile[0].m <= 0.5:
                chat_log.append(("Analyste", "Cette couche fait fondre les flocons, mais comme le sol est en dessous de zéro, les gouttes regèlent. Le résultat le plus probable est du **grésil (sleet)** ou la dangereuse **pluie verglaçante**."))
                precipitation_type = 'sleet'
            else:
                 chat_log.append(("Analyste", "Exactement. La couche chaude fait fondre la neige et, comme la surface est positive, elle arrivera sous forme de pluie froide. C'est le scénario typique de 'il pleut et il fait froid'."))
                 precipitation_type = 'rain'
    else:
        if "Tornadique" in cloud_type or "Tuba" in cloud_type or "Mur" in cloud_type: precipitation_type = 'hail'
        elif cape.m > 500: precipitation_type = 'rain'
        elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
        
        chat_log = [("Système", f"Lancement de l'analyse conversationnelle pour le scénario de {cloud_type}.")]

        if "Tornadique" in cloud_type:
            chat_log.extend([
                ("Analyste", "ALERTE MAXIMALE. Le profil n'est pas seulement supercellulaire, il présente des caractéristiques tornadiques classiques."),
                ("Utilisateur", "Qu'est-ce qui le rend si dangereux ?"),
                ("Analyste", f"Nous avons trois ingrédients clés : une base de nuage très basse ({lcl_h:.0f} m), une rotation à bas niveaux extrêmement forte (SRH 0-1km : {srh_0_1:.0f} m²/s²), et une forte instabilité. La rotation a de fortes chances d'atteindre le sol."),
            ])
        elif "Tuba/Entonnoir" in cloud_type:
            chat_log.extend([
                ("Analyste", "Très grande prudence. C'est un profil de supercellule avec un fort potentiel de développement d'entonnoirs."),
                ("Utilisateur", "Qu'est-ce qui indique ce potentiel ?"),
                ("Analyste", f"La combinaison d'une hélicité significative à bas niveaux (SRH 0-1km : {srh_0_1:.0f} m²/s²) et d'une base de nuage relativement basse ({lcl_h:.0f} m). La rotation peut facilement descendre et se condenser."),
            ])
        elif "Nuage-mur" in cloud_type:
            chat_log.extend([
                ("Analyste", "C'est un profil de supercellule classique et très organisé."),
                ("Utilisateur", "Qu'est-ce qui le rend spécial ?"),
                ("Analyste", f"La rotation aux niveaux moyens est très intense (SRH 0-3km : {srh_0_3:.0f} m²/s²). Cela peut provoquer un abaissement localisé de la base, formant un nuage-mur, qui est la région principale d'où naissent les tornades."),
            ])
        elif "Arcus" in cloud_type:
             chat_log.extend([
                ("Analyste", "Attention. Ce profil est dangereux, mais pour une raison autre que la rotation."),
                ("Utilisateur", "Ce n'est pas une supercellule ?"),
                ("Analyste", f"Pas exactement. Malgré la grande énergie (CAPE : {cape.m:.0f} J/kg), la clé ici est le potentiel pour un courant descendant (downdraft) très violent. Le principal danger réside dans les vents linéaires destructeurs (rafale descendante ou downburst)."),
            ])
        elif "Base Rugueuse" in cloud_type:
            chat_log.extend([
                ("Analyste", "Nous avons un orage fort en cours."),
                ("Utilisateur", "Quel est le détail le plus significatif ?"),
                ("Analyste", f"L'énergie est élevée (CAPE : {cape.m:.0f} J/kg) et il y a une forte entrée d'air humide. Cela provoque la condensation d'humidité sous la base principale, créant une base rugueuse avec des fragments (scud). C'est un signe que l'orage s'alimente fortement."),
            ])
        elif "Supercellule" in cloud_type:
            chat_log.extend([
                ("Analyste", "C'est un profil de manuel pour du temps violent."),
                ("Utilisateur", f"Je vois beaucoup d'énergie (CAPE : {cape.m:.0f} J/kg) et de cisaillement ({shear_0_6:.1f} m/s)."),
                ("Analyste", "Exactement. Cette combinaison crée un moteur qui permet à un orage de s'organiser et de tourner. Le pronostic doit être prudent en raison du risque de grosse grêle et de vents forts."),
            ])
        elif cloud_type == "Nimbostratus":
            chat_log.extend([
                ("Analyste", "Ce profil est très différent. Ici, l'histoire n'est pas celle d'une instabilité explosive."),
                ("Utilisateur", f"C'est vrai, la CAPE est presque inexistante, seulement {cape.m:.0f} J/kg."),
                ("Analyste", f"Exactement. Le protagoniste ici est l'humidité. Nous avons une couche d'air très épaisse et saturée. C'est un scénario typique de pluie stratiforme, associée aux systèmes frontaux. L'intensité dépendra de l'eau précipitable ({pwat_0_4.m:.1f} mm)."),
            ])
        elif cloud_type == "Cirrus":
            chat_log.extend([
                ("Analyste", "Intéressant. Le profil est très sec dans les couches basses et moyennes."),
                ("Utilisateur", "Donc il n'y aura pas de nuages ?"),
                ("Analyste", "À haute altitude, au-dessus de 6-7 km, il y a une fine couche d'humidité. Cela crée les conditions idéales pour les Cirrus, des nuages élevés de cristaux de glace qui ne produisent pas de précipitations."),
            ])
        elif cloud_type == "Altostratus / Altocumulus":
            chat_log.extend([
                ("Analyste", "Nous avons un cas de nébulosité à moyens niveaux."),
                ("Utilisateur", f"Pourquoi ? Il n'y a presque pas de CAPE ({cape.m:.0f} J/kg)."),
                ("Analyste", "Correct, la convection depuis la surface est inhibée. Cependant, observez la couche d'humidité marquée entre 3 et 6 km. Cela formera une couche de nuages moyens, comme des Altostratus ou des Altocumulus."),
            ])
        elif cloud_type == "Cumulus Humilis":
            chat_log.extend([
                ("Analyste", "Nous observons un scénario de temps stable."),
                ("Utilisateur", f"Mais il y a un peu de CAPE, {cape.m:.0f} J/kg."),
                ("Analyste", "Oui, il y a un peu d'énergie, mais très peu et avec un fort 'couvercle' juste au-dessus qui empêche toute croissance. C'est un profil typique pour les 'nuages de beau temps'."),
            ])
        elif cloud_type == "Cumulus Mediocris":
            chat_log.extend([
                ("Analyste", "C'est un profil intéressant pour un après-midi d'été."),
                ("Utilisateur", f"Nous avons {cape.m:.0f} J/kg de CAPE. Est-ce suffisant pour des orages ?"),
                ("Analyste", "C'est une énergie modérée avec un faible cisaillement. Cela permet une certaine croissance verticale, mais pas explosive. Cela favorise la formation de Cumulus Mediocris, qui donnent rarement plus que quelques gouttes."),
            ])
        elif cloud_type == "Cumulus Congestus":
            chat_log.extend([
                ("Analyste", "Attention, ici nous commençons à voir un potentiel pour des phénomènes plus actifs."),
                ("Utilisateur", f"La CAPE est déjà plus considérable, {cape.m:.0f} J/kg."),
                ("Analyste", "Exactement. Nous avons assez d'énergie pour un développement vertical important. Ce sont l'étape préalable au Cumulonimbus et peuvent déjà laisser des averses localement intenses."),
            ])
        elif cloud_type == "Cumulonimbus (Multicellulaire)":
            chat_log.extend([
                ("Analyste", "Bien, nous avons un scénario avec un potentiel d'orages."),
                ("Utilisateur", f"La CAPE est de {cape.m:.0f} J/kg."),
                ("Analyste", f"C'est une bonne valeur, suffisante pour des orages forts. De plus, le LFC ({lfc_h:.0f} m) est assez bas pour permettre à la convection de démarrer."),
                ("Utilisateur", "Et s'organiseront-ils ? Comment est le cisaillement ?"),
            ])
            if shear_0_6 < 10: shear_analysis_message = f"Le cisaillement ({shear_0_6:.1f} m/s) est faible. Les orages seront probablement désorganisés et à cycle de vie court."
            elif shear_0_6 < 18: shear_analysis_message = f"Le cisaillement ({shear_0_6:.1f} m/s) est modéré. Il permettra des systèmes multicellulaires plus durables."
            else: shear_analysis_message = f"Le cisaillement ({shear_0_6:.1f} m/s) est fort. Il y a une organisation considérable et les orages seront robustes."
            chat_log.append(("Analyste", shear_analysis_message))
        elif cloud_type == "Castellanus":
            chat_log.extend([
                ("Analyste", "C'est un cas particulier. Nous avons de l'énergie en altitude, mais la surface est déconnectée."),
                ("Utilisateur", "Que veux-tu dire ?"),
                ("Analyste", f"La CIN est très forte ({cin.m:.0f} J/kg), ce qui empêche la convection de commencer depuis le sol. Cependant, il y a une couche instable aux niveaux moyens qui peut générer des Altocumulus Castellanus."),
            ])
        elif cloud_type == "Cumulus Fractus":
             chat_log.extend([
                ("Analyste", "Ce que nous voyons ici, ce sont des conditions résiduelles."),
                ("Utilisateur", "Qu'est-ce que cela veut dire ?"),
                ("Analyste", "Il y a un peu d'humidité et d'instabilité, mais c'est très peu et désorganisé. Cela ne permettra que la formation de morceaux de nuages déchiquetés, sans aucun risque associé."),
            ])
        else:
            chat_log.extend([
                ("Analyste", "Le profil atmosphérique est très stable."),
                ("Utilisateur", "Alors, nous ne verrons aucun nuage ?"),
                ("Analyste", f"C'est très peu probable. Avec une CAPE de seulement {cape.m:.0f} J/kg, il n'y a pratiquement aucune énergie pour la croissance verticale."),
            ])
    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type):
    """Génère une analyse conversationnelle pour le mode laboratoire."""
    cape, cin, _, lcl_h, _, lfc_h, _, _, _ = calculate_thermo_parameters(p, t, td)
    shear_0_6, _, _, _ = calculate_storm_parameters(p, ws, wd)
    chat_log = []
    
    chat_log.append(("Analyste", "Très bien, analysons le profil que vous avez créé. On commence ?"))

    if cape.m < 50:
        chat_log.extend([
            ("Utilisateur", "Avons-nous un potentiel orageux ?"),
            ("Analyste", f"Pas pour le moment. La CAPE n'est que de {cape.m:.0f} J/kg. L'atmosphère est très stable.")
        ])
    else:
        chat_log.extend([("Utilisateur", "Qu'est-ce que je crée avec cette énergie ?")])
        cloud_mention = f"C'est un scénario typique pour la formation de {cloud_type}."
        if cloud_type == "Ciel Dégagé":
             cloud_mention = "Bien qu'il y ait de l'énergie, le couvercle est si fort que nous ne verrions probablement aucun nuage significatif."
        chat_log.append(("Analyste", f"Vous avez généré une CAPE de {cape.m:.0f} J/kg. {cloud_mention}"))

        chat_log.append(("Utilisateur", "Et le 'couvercle' (CIN) ? Comment affecte-t-il ?"))
        if cin.m < -100:
            chat_log.append(("Analyste", f"Très fort ({cin.m:.0f} J/kg). La convection depuis la surface est presque impossible."))
        elif cin.m < -50:
            chat_log.append(("Analyste", f"Il est considérable ({cin.m:.0f} J/kg). Il ouvre la porte à la convection de base élevée (Castellanus)."))
        elif cin.m < -25:
            chat_log.append(("Analyste", f"Il est modéré ({cin.m:.0f} J/kg). Il permet à l'énergie de s'accumuler, un scénario classique pour de forts orages."))
        else:
             chat_log.append(("Analyste", f"Il est faible ({cin.m:.0f} J/kg). La convection a presque le champ libre."))
        
        if cin.m > -100 and cape.m > 800:
            chat_log.append(("Utilisateur", "J'ai modifié le vent. Comment cela affecte-t-il ?"))
            if shear_0_6 > 18:
                chat_log.append(("Analyste", f"Le cisaillement est fort ({shear_0_6:.1f} m/s). C'est l'ingrédient clé pour organiser les orages en supercellules."))
            elif shear_0_6 > 10:
                chat_log.append(("Analyste", f"Le cisaillement est modéré ({shear_0_6:.1f} m/s). Il aide à organiser les orages en systèmes multicellulaires."))
            else:
                chat_log.append(("Analyste", f"Le cisaillement est faible ({shear_0_6:.1f} m/s). Les orages seront probablement plus désorganisés."))
    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Génère l'analyse du chat pour une étape spécifique d'un tutoriel."""
    chat_log = []
    if scenario == 'grésil':
        if step == 0: chat_log.extend([("Analyste", "Bienvenue ! Analysons un profil classique de grésil."), ("Utilisateur", "Parfait. Que dois-je regarder en premier ?"), ("Analyste", "Observez l' 'usine à neige' dans les couches supérieures. Au-dessus de 700 hPa, il fait assez froid pour former des flocons de neige.")])
        elif step == 1: chat_log.extend([("Analyste", "Très bien. Voici la partie cruciale. Regardez la couche autour de 850 hPa. La température dépasse 0°C."), ("Utilisateur", "C'est la 'couche chaude', n'est-ce pas ? Qu'est-ce que ça provoque ?"), ("Analyste", "Exactement. Cette couche agit comme un 'chalumeau' et fait fondre les flocons, les transformant en gouttes de pluie.")])
        elif step == 2: chat_log.extend([("Analyste", "On y est presque. Maintenant, nous avons des gouttes de pluie qui tombent vers la surface. Mais regardez la température près du sol..."), ("Utilisateur", "Elle est de nouveau en dessous de 0°C !"), ("Analyste", "Précisément ! Ces gouttes regèlent juste avant d'atteindre le sol. C'est ça, le grésil (sleet).")])
        elif step == 3: chat_log.extend([("Analyste", "Vous avez parfaitement analysé le profil."), ("Utilisateur", "Compris. Alors, comment pourrais-je le transformer en chute de neige ?"), ("Analyste", "C'est le défi ! Maintenant, lorsque vous terminerez le tutoriel, allez en Mode Libre et utilisez l'outil '❄️ Refroidir Couche Moyenne'. Vous verrez comment le profil se transforme en une parfaite chute de neige.")])
    elif scenario == 'supercellule':
        if step == 0: chat_log.append(("Analyste", "Commençons le tutoriel de la supercellule. La première étape est toujours de créer de l'énergie. Nous avons besoin d'une chaude journée d'été. Réchauffons la surface !"))
        elif step == 1: chat_log.append(("Analyste", "Correct ! Maintenant, ajoutons le carburant : l'humidité. Vous verrez la valeur de la CAPE augmenter lorsque les lignes de température et de point de rosée se rapprochent."))
        elif step == 2: chat_log.append(("Analyste", "Fantastique ! Vous avez ajouté du cisaillement. C'est l'ingrédient secret qui fait tourner les orages. Nous avons maintenant la recette parfaite !"))
        elif step == 3: chat_log.append(("Analyste", "Mission accomplie ! Vous avez créé un profil avec beaucoup d'énergie (CAPE), d'humidité et de cisaillement. Remarquez comment les paramètres de cisaillement (Shear) et d'hélicité (SRH) ont augmenté."))
    return chat_log, None
    
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    if sfc_temp.m < 7.0: # Analyse hivernale
        if sfc_temp.m <= 0.5:
            try:
                p_arr, t_arr = p_levels.m, t_profile.m
                warm_layer_mask = (p_arr < 950) & (p_arr > 600) & (t_arr > 0.5)
                if np.any(warm_layer_mask):
                    return "GRÉSIL OU PLUIE VERGLAÇANTE", "Une couche chaude en altitude peut faire fondre la neige. Risque de grésil ou de pluie verglaçante.", "mediumorchid"
                else:
                    return "AVIS DE NEIGE", "Profil atmosphérique favorable aux chutes de neige à basse altitude.", "navy"
            except:
                return "AVIS DE NEIGE", "Chutes de neige prévues à basse altitude. Prudence sur les routes.", "navy"
        else:
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVIS DE PLUIE VERGLAÇANTE", "Risque de pluie verglaçante ou de verglas. Prudence extrême.", "dodgerblue"
            else:
                 return "AMBIANCE FROIDE ET HUMIDE", "Conditions froides. Les précipitations seraient sous forme de pluie ou de neige très humide.", "steelblue"

    if cape.m >= 1200:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)

        if cin.m <= -100: return "AVIS, CONVECTION FORTEMENT INHIBÉE", f"Potentiel énergétique (CAPE {cape.m:.0f} J/kg) bloqué par un 'couvercle' très fort (CIN {cin.m:.0f} J/kg).", "darkslategray"
        if -100 < cin.m <= -50: return "AVIS, CONVECTION POSSIBLE EN ALTITUDE", f"La convection depuis la surface est difficile (CIN {cin.m:.0f} J/kg). Risque de cellules en altitude.", "slategray"

        title, color, message = "AVIS D'ORAGES VIOLENTS", "darkorange", f"CAPE: {cape.m:.0f} J/kg. "

        if srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18:
            title, color = "AVIS DE TORNADE", "darkred"; message += f"Risque élevé de tornades (SRH 0-1km: {srh_0_1:.0f}, LCL: {lcl_h:.0f}m)."
        elif srh_0_3 > 250 and shear_0_6 > 18:
            title, color = "AVIS DE TEMPS VIOLENT", "purple"; message += f"Supercellules probables. Risque de grosse grêle et/ou de nuages-murs (SRH 0-3km: {srh_0_3:.0f})."
        elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150):
            title, color = "AVIS DE VENTS FORTS", "saddlebrown"; message += "Risque de rafales de vent linéaires violentes (rafales descendantes/arcus)."
        elif cape.m > 2000 and fz_h < 4200:
            title, color = "AVIS D'ORAGES FORTS", "mediumvioletred"; message += f"Conditions favorables à la grosse grêle (Iso 0°C calculée: {int(fz_h)} m)."
        elif cape.m > 3500:
            title, color = "AVIS DE GRÊLE ET ORAGES FORTS", "mediumvioletred"; message += "Conditions favorables à la grosse grêle."
        
        return title, message, color

    try:
        heights_amsl = mpcalc.pressure_to_height_std(p_levels).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_layer = mpcalc.relative_humidity_from_dewpoint(t_profile[layer_mask], td_profile[layer_mask])
            pwat_layer = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask]).to('mm')
            if np.mean(rh_layer) > 0.85 and cape.magnitude < 350:
                if pwat_layer.m > 25: return "AVIS DE PLUIES INTENSES", "(Activez le forçage) Risque de pluies persistantes et fortes.", "darkblue"
                elif pwat_layer.m > 15: return "AVIS DE PLUIE MODÉRÉE", "Ciel couvert avec pluie continue et modérée.", "steelblue"
                else: return "PRÉVISION DE PLUIE FAIBLE", "(Activez le forçage) Des bruines ou averses faibles sont attendues.", "cadetblue"
    except Exception: pass

    return "AUCUN AVIS", "Conditions météorologiques sans risques significatifs.", "green"

def determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p):
    potential_clouds = []
    try:
        if len(p) < 2: return ["Données insuffisantes"]
        heights = mpcalc.pressure_to_height_std(p).to('m')
        rh = mpcalc.relative_humidity_from_dewpoint(t, td)
        t_interp_func = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")

        if np.any((mask_high := (heights.m > 6000) & (heights.m < 18000))) and np.sum(mask_high) > 1:
            rh_high = np.mean(rh[mask_high])
            if rh_high > 0.85: potential_clouds.append("Cirrostratus")
            elif rh_high > 0.80: potential_clouds.append("Cirrocumulus")
            elif rh_high > 0.75: potential_clouds.append("Cirrus")

        if np.any((mask_mid := (heights.m > 2000) & (heights.m < 7000))) and np.sum(mask_mid) > 1:
            if np.any((mask_as := (heights.m > 2000) & (heights.m < 6000))) and np.sum(mask_as) > 1 and np.mean(rh[mask_as]) > 0.90 and cape.m < 50:
                 potential_clouds.append("Altostratus")
            elif (0.70 < np.mean(rh[mask_mid]) < 0.90) and cape.m <= 100:
                potential_clouds.append("Altocumulus")

        if np.any((mask_low := (heights.m < 2500))) and np.sum(mask_low) > 1:
            if np.any((mask_st := (heights.m < 1500))) and np.sum(mask_st) > 1 and np.mean(rh[mask_st]) > 0.95 and cape.m < 10:
                potential_clouds.append("Stratus (Brouillard ascendant)")
            elif np.mean(rh[mask_low]) > 0.85 and cape.m <= 50:
                 potential_clouds.append("Stratocumulus")

        if np.any((mask_ns := (heights.m > 500) & (heights.m < 5000))) and np.sum(mask_ns) > 1 and np.mean(rh[mask_ns]) > 0.95 and cape.m <= 100:
            potential_clouds.append("Nimbostratus")
            
        if cape.m > 100 and lcl_h is not None and lcl_h > 0:
            if lfc_h is not None and lfc_h < 3000:
                if (100 < cape.m < 2500) and (lfc_h > lcl_h): potential_clouds.append("Cumulus (Humilis, Mediocris ou Congestus)")
                if cape.m > 1000:
                    try:
                        if el_p is not None and not np.isnan(el_p.m) and t_interp_func(el_p.m) < 0:
                            potential_clouds.append("Cumulonimbus")
                    except: pass
            else:
                if np.any((mask_cast := (heights.m > 2000) & (heights.m < 7000))) and np.sum(mask_cast) > 1 and np.mean(rh[mask_cast]) > 0.60:
                         potential_clouds.append("Altocumulus Castellanus")

        final_clouds, has_cb, has_cu, has_ns, has_castellanus = [], "Cumulonimbus" in potential_clouds, "Cumulus (Humilis, Mediocris ou Congestus)" in potential_clouds, "Nimbostratus" in potential_clouds, "Altocumulus Castellanus" in potential_clouds
        for cloud in potential_clouds:
            if (cloud == "Cumulus (Humilis, Mediocris ou Congestus)" and has_cb) or \
               (cloud == "Stratocumulus" and (has_cb or has_cu or has_ns)) or \
               (cloud == "Altocumulus" and has_castellanus) or \
               (cloud in final_clouds): continue
            final_clouds.append(cloud)

        return sorted(final_clouds) if final_clouds else ["Ciel Dégagé ou Nuages résiduels"]
    except Exception as e: return [f"Impossible de déterminer la nébulosité. Erreur : {e}"]

# =========================================================================
# === 3. FONCTIONS DE DESSIN ===============================================
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
        skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Température (T)')
        skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Point de Rosée (Td)')
        parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
        skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Parcelle Adiabatique')
        wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
        skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='T° Bulbe Humide')
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
    ax.set_ylabel("Altitude (km)"); ax.set_title("Visualisation du Nuage")
    ax.grid(True, linestyle='dashdot', alpha=0.5); ax.set_facecolor('#6495ED')
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//' if ground_color=='#228B22' else ''))
    
    if not convergence_active:
        _draw_saturation_layers(ax, p_levels, t_profile, td_profile)
    if base_km is not None and top_km is not None and (top_km - base_km > 0.1):
        if "Nimbostratus" in cloud_type or "Hivernal" in cloud_type: _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif "Altostratus" in cloud_type: _draw_stratiform_cotton_clouds(ax, base_km, top_km)
        elif "Cirrus" in cloud_type: _draw_clear_sky(ax)
        elif "Supercellule" in cloud_type or "Cumulonimbus" in cloud_type: _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus": _draw_cumulus_castellanus(ax, base_km, top_km)
        elif cloud_type in ["Cumulus Mediocris", "Cumulus Congestus", "Cumulus Humilis"]: _draw_cumulus_mediocris(ax, base_km, top_km)
        elif cloud_type == "Cumulus Fractus": _draw_cumulus_fractus(ax, base_km, top_km - base_km)
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
        except Exception: pass
        _draw_precipitation(ax, precip_base_km, ground_height_km, precipitation_type, sub_cloud_rh=sub_cloud_rh_mean)
    plt.tight_layout()
    return fig

def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active):
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_shear = fig.add_subplot(gs[0, 1], sharey=ax)
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set_title("Structure Verticale et Cisaillement", fontsize=10); ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitude (km)", xticks=[]); ax.grid(True, linestyle='--', alpha=0.3)
    ax_shear.set(xlim=(-1, 1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    if not base_km or not top_km or cape.m < 5 or not convergence_active:
        ax.text(0.5, 0.5, "Pas de Structure Convective\n(Activez le forçage pour la simuler)", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='white', bbox=dict(facecolor='darkblue', alpha=0.7))
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
    ax.set_facecolor('darkslategray'); ax.set_title("Écho Radar Simulé", fontsize=10)
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
            if np.mean(rh_layer) > 0.85 and cape.magnitude < 350:
                x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
                max_dbz = np.clip(15 + pwat_layer.m, 15, 45)
                noise = gaussian_filter(np.random.randn(100, 100), sigma=8) * (max_dbz * 0.2)
                Z = np.clip(max_dbz + noise, 0, 50)
                radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900']
                radar_levels = [0, 15, 20, 25, 30, 35, 45]
                radar_cmap = ListedColormap(radar_colors)
                radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
                ax.contourf(x, y, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
                return fig
    except Exception: pass
    if cape.m < 100:
        ax.text(0, 0, "Pas de précipitations significatives", ha='center', va='center', color='white', fontsize=9)
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
    Z += gaussian_filter(np.random.randn(150, 150), sigma=6) * (max_dbz * 0.1)
    Z = np.clip(Z, 0, 75)
    radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
    radar_cmap, radar_norm = ListedColormap(radar_colors), BoundaryNorm(radar_levels, len(radar_colors))
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    return fig

def create_hodograph_figure(p, ws, wd, t, td):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=40.)
    h.add_grid(increment=10, ls='--', color='gray')
    ax.set_xlabel('kt'); ax.set_ylabel('kt')
    
    try:
        p_hodo, ws_hodo, wd_hodo = p.to('hPa'), ws.to('kt'), wd.to('deg')
        u, v = mpcalc.wind_components(ws_hodo, wd_hodo)
        heights = mpcalc.pressure_to_height_std(p_hodo).to('km')
        h_interp = np.arange(0, min(12, heights.m.max()), 0.1) * units.km
        u_interp = np.interp(h_interp.m, heights.m, u.m) * units.kt
        v_interp = np.interp(h_interp.m, heights.m, v.m) * units.kt
        levels, colors = [0, 1, 3, 5, 8, 10], ['green', 'orange', 'red', 'purple', 'darkviolet']
        cmap, norm = ListedColormap(colors), BoundaryNorm(levels, len(colors))
        for i in range(len(h_interp) - 1):
            ax.plot(u_interp[i:i+2].m, v_interp[i:i+2].m, color=cmap(norm(h_interp[i].m)), linewidth=2)
        with integrator_lock:
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p_hodo, u, v, heights)
        ax.arrow(0, 0, rm[0].m, rm[1].m, color='black', width=0.5, head_width=2, length_includes_head=True, label="Mouvement Orage (MD)")
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, pad=0.08)
        cbar.set_label('Altitude (km)')
    except Exception as e:
        ax.text(0.5, 0.5, "Données de vent insuffisantes\npour générer l'hodographe.", 
                ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    return fig

# =========================================================================
# === 4. STRUCTURE DE L'APPLICATION =======================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">METEO-FRANCE PRÉSENTE :</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Un outil pour la visualisation et l\'expérimentation avec les profils atmosphériques.</p>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️ Temps réel</h3><p>Visualisez les sondages atmosphériques les plus récents basés sur des données de modèles pour les zones les plus actives du jour.</p></div>""", unsafe_allow_html=True)
        if st.button("Accéder au Mode Temps Réel", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪 Laboratoire</h3><p>Apprenez de manière interactive comment se forment les phénomènes violents en modifiant pas à pas un sondage ou expérimentez librement.</p></div>""", unsafe_allow_html=True)
        if st.button("Accéder au Laboratoire", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False):
    st.markdown(f"#### {obs_time}")
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    
    st.toggle(
        "Activer le Forçage (Convergence)", key='convergence_active',
        help="Simule l'effet d'un mécanisme de déclenchement (p.ex. orographie). Si activé, les nuages croîtront jusqu'à leur sommet théorique (EL) s'il y a de la CAPE, en ignorant l'inhibition (CIN)."
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
    elif cape.m > 1500 and srh_0_1 > 150 and lcl_h < 1000 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercellule (Tornadique)"
    elif cape.m > 1500 and srh_0_1 > 120 and lcl_h < 1200 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercellule (Tuba/Entonnoir)"
    elif cape.m > 1800 and srh_0_3 > 250 and shear_0_6 > 18 and convection_possible_from_surface: cloud_type = "Supercellule (Nuage-mur)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150 and convection_possible_from_surface: cloud_type = "Supercellule"
    elif cape.m > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150): cloud_type = "Cumulonimbus (Arcus)"
    elif cape.m > 1200 and s_0_1 > 8 and convection_possible_from_surface: cloud_type = "Cumulonimbus (Base Rugueuse)"
    elif cape.m >= 1200 and convection_possible_from_surface: cloud_type = "Cumulonimbus (Multicellulaire)"
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
    else: cloud_type = "Ciel Dégagé"

    if cloud_type == "Ciel Dégagé" and base_km and top_km and (top_km - base_km) > 0.05:
        cloud_type = "Cumulus Fractus"

    if "Supercellule" in cloud_type or "Cumulonimbus" in cloud_type or "Congestus" in cloud_type or "Castellanus" in cloud_type:
        if lfc_h and base_km is not None and (lfc_h / 1000.0) > base_km: base_km = lfc_h / 1000.0
    
    st.subheader("Diagramme Skew-T", anchor=False)
    st.pyplot(create_skewt_figure(p, t, td, ws, wd), use_container_width=True)
    st.divider()

    if is_sandbox_mode:
         chat_log, precipitation_type = generate_dynamic_analysis(p, t, td, ws, wd, cloud_type)
    else:
        chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)

    potential_clouds = determine_potential_cloud_types(p, t, td, cape, lcl_h, lfc_h, el_p)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Assistant d'Analyse", "📊 Paramètres Détaillés", "📈 Hodographe", "☁️ Visualisation des Nuages", "📋 Types de Nuages", "📡 Simulation Radar"])
    
    with tab1:
        css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.utilisateur { background-color: #dcf8c6; align-self: flex-end; }.analyste { background-color: #ffffff; }.système { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.utilisateur strong { color: #005C4B; }</style>"""
        html_chat = "<div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower()
            html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'utilisateur' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)

        image_triggers = {"tornade": ("tornado.jpg", "Une tornade formée sous une supercellule."),"tornadique": ("tornado.jpg", "Une tornade formée sous une supercellule."),"tuba": ("funnel.jpg", "Un tuba (funnel cloud) descendant de la base du nuage."),"nuage-mur": ("wallcloud.jpg", "Un nuage-mur (wall cloud) bien défini."),"arcus": ("shelfcloud.jpg", "Un spectaculaire arcus (shelf cloud)."),"base rugueuse": ("scud.jpg", "Base rugueuse avec fragments de nuages (scud)."),"supercellule": ("supercell.jpg", "Une supercellule organisée."),"castellanus": ("castellanus.jpg", "Ceci est un Altocumulus Castellanus."),"fractus": ("fractus.jpg", "Ceci est un Cumulus Fractus."),"cumulonimbus": ("cumulonimbus.jpg", "Ceci est un Cumulonimbus."),"congestus": ("congestus.jpg", "Ceci est un Cumulus Congestus."),"mediocris": ("mediocris.jpg", "Ceci est un Cumulus Mediocris."),"humilis": ("humilis.jpg", "Ceci est un Cumulus Humilis."),"cirrus": ("cirrus.jpg", "Ce sont des nuages Cirrus."),"altostratus": ("altostratus.jpg", "Ceci est un ciel couvert par des Altostratus."),"grésil": ("sleet.jpg", "Précipitation sous forme de grésil (sleet)."),"neige": ("snow.jpg", "Une chute de neige couvrant le paysage.")}
        images_to_show = set() 
        full_chat_text = " ".join([msg for _, msg in chat_log]).lower() + " " + cloud_type.lower()
        for keyword, (filename, caption) in image_triggers.items():
            if keyword in full_chat_text: images_to_show.add((filename, caption))
        if images_to_show:
            st.markdown("---")
            for filename, caption in sorted(list(images_to_show)):
                image_base64 = get_image_as_base64(filename)
                if image_base64: st.markdown(f"<div style='margin-top: 15px; text-align: center;'><img src='{image_base64}' style='max-width: 80%; border-radius: 10px;'><p style='font-style: italic; color: grey;'>{caption}</p></div>", unsafe_allow_html=True)
                else: st.warning(f"Le mot-clé '{keyword}' a été mentionné, mais le fichier '{filename}' est introuvable.", icon="🖼️")
    with tab2:
        st.subheader("Paramètres Thermodynamiques et de Cisaillement")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_h:.0f} m"); param_cols[1].metric("LFC", f"{lfc_h:.0f} m" if lfc_h != np.inf else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); param_cols[3].metric("Cisaillement 0-6km", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1km", f"{srh_0_1:.1f} m²/s²"); param_cols[1].metric("SRH 0-3km", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm")
        rh_display = "N/A"
        try: rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("HR Moy. 0-4km", rh_display)
    with tab3:
        st.subheader("Hodographe du Profil de Vents")
        st.pyplot(create_hodograph_figure(p, ws, wd, t, td), use_container_width=True)
    with tab4:
        st.subheader("Représentations Graphiques du Nuage")
        cloud_cols = st.columns(2)
        with cloud_cols[0]: st.pyplot(create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type), use_container_width=True)
        with cloud_cols[1]: st.pyplot(create_cloud_structure_figure(p, t, td, ws, wd, convergence_active), use_container_width=True)
    with tab5:
        st.subheader("Liste des Genres de Nuages Probables")
        st.markdown("Cette liste est analysée automatiquement à partir des couches d'humidité, d'instabilité et de température du sondage.")
        if potential_clouds:
            for cloud in potential_clouds: st.markdown(f"- **{cloud}**")
        else: st.info("Selon l'analyse, aucune formation de nuages significative n'est attendue.")
        st.markdown("---")
        st.caption("Cette analyse est basée sur un seul profil vertical et ne tient pas compte des facteurs synoptiques à grande échelle.")
    with tab6:
        st.subheader("Simulation de Réflectivité Radar")
        st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

def show_province_selection_screen():
    set_main_background()
    fig_scape = create_city_mountain_scape()
    st.pyplot(fig_scape, use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>Analyse des Zones Météorologiques</h2>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.button("Suivre la zone d'intérêt du jour", on_click=lambda: st.session_state.update(province_selected='suivi_menu'), use_container_width=True, type="primary")

def show_seguiment_selection_screen():
    st.title("Zone d'Intérêt du Jour")
    st.markdown("Sélectionnez la zone que vous souhaitez analyser. Chaque zone représente un profil atmosphérique différent basé sur les prévisions les plus récentes.")
    
    with st.sidebar:
        st.header("Contrôles")
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.province_selected = None
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🔥 Zone la Plus Remarquable</h4><p>Le profil avec le plus grand potentiel pour des phénomènes significatifs.</p></div>""", unsafe_allow_html=True)
        if st.button("Zone Sud (Potentiel Élevé)", use_container_width=True, type="primary"):
            st.session_state.province_selected = 'suivi_remarquable'
            st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>🤔 Zone Intéressante</h4><p>Un profil qui présente quelques caractéristiques d'intérêt.</p></div>""", unsafe_allow_html=True)
        if st.button("Zone Nord (Intéressante)", use_container_width=True):
            st.session_state.province_selected = 'suivi_interessant'
            st.rerun()

def run_single_sounding_mode(mode):
    suivi_map = {
        'suivi_remarquable': {'file': 'sondeig_destacable.txt', 'title': "ZONE LA PLUS REMARQUABLE", 'zone': "Zone Sud (Potentiel Élevé)"},
        'suivi_interessant': {'file': 'sondeig_interessant.txt', 'title': "ZONE INTÉRESSANTE", 'zone': "Zone Nord (Intéressante)"}
    }
    
    config = suivi_map[mode]
    zone = config['zone']
    st.title(f"{config['title']} - {zone.upper()}")
    
    with st.sidebar:
        st.header("Contrôles")
        st.button("⬅️ Retour à la sélection", use_container_width=True, on_click=lambda: st.session_state.update(province_selected='suivi_menu'))

    content_placeholder = st.empty()
    with content_placeholder.container():
        show_loading_animation(message=f"Chargement {config['title']}")
        time.sleep(0.1) 

    try:
        soundings = parse_all_soundings(config['file'])
        content_placeholder.empty()
        if soundings:
            data = soundings[0]
            obs_time = data.get('observation_time', f"Sondage de la {config['title'].lower()}")
            show_full_analysis_view(
                p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                obs_time=obs_time
            )
        else:
            content_placeholder.empty()
            st.error(f"Impossible de charger les données du sondage '{config['file']}'.")
    except FileNotFoundError:
        content_placeholder.empty()
        st.error(f"Le fichier '{config['file']}' n'existe pas.")

def run_live_mode():
    selection = st.session_state.get('province_selected')
    if selection == 'suivi_menu':
        show_seguiment_selection_screen()
    elif selection and selection.startswith('suivi_'):
        run_single_sounding_mode(selection)
    else: 
        with st.sidebar:
            st.header("Contrôles")
            if st.button("⬅️ Retour à l'accueil", use_container_width=True):
                st.session_state.app_mode = 'welcome'
                if 'province_selected' in st.session_state: del st.session_state.province_selected
                st.rerun()
        show_province_selection_screen()

# =================================================================================
# === LABORATOIRE-TUTORIEL ========================================================
# =================================================================================

def get_tutorial_data():
    return {
        'supercellule': [
            {'action_id': 'warm_low', 'title': 'Étape 1 : Réchauffement de surface', 'instruction': "Nous avons besoin d'énergie. Le moyen le plus courant est le réchauffement solaire pendant la journée. Cliquez sur le bouton ci-dessous pour réchauffer les basses couches.", 'button_label': "☀️ Réchauffer Couche Basse", 'explanation': "Cela augmente la température près de la surface, créant une 'bulle' d'air qui voudra monter."},
            {'action_id': 'moisten_low', 'title': 'Étape 2 : Ajoutez du carburant', 'instruction': "Un orage a besoin d'humidité pour se former. Cliquez sur le bouton pour humidifier les basses couches et rapprocher le point de rosée de la température.", 'button_label': "💧 Humidifier Couche Basse", 'explanation': "Cela permet à l'air ascendant de se condenser plus tôt, libérant de la chaleur latente et donnant plus de force à l'orage (augmentant la CAPE)."},
            {'action_id': 'add_shear_low', 'title': "Étape 3 : Ajoutez le moteur de rotation", 'instruction': "L'ingrédient secret d'une supercellule est le cisaillement du vent à bas niveaux. Cliquez sur le bouton pour ajouter un changement de vent avec l'altitude.", 'button_label': "🌪️ Ajouter Cisaillement Basses Couches", 'explanation': "Cela fera tourner le courant ascendant de l'orage, l'organisant et le rendant beaucoup plus puissant et durable."},
            {'action_id': 'conceptual', 'title': 'Étape 4 : Analyse Finale', 'instruction': "Nous avons maintenant de l'énergie, de l'humidité et de la rotation. Vous avez créé un environnement parfait pour la formation de supercellules.", 'button_label': "Compris, finaliser →", 'explanation': "Dans l'analyse finale, remarquez comment les paramètres de cisaillement (Shear) et d'hélicité (SRH) ont augmenté."},
        ],
        'grésil': [
            {'action_id': 'conceptual', 'title': "Étape 1 : L'Usine à Neige", 'instruction': "Nous avons chargé un profil de grésil. Observez dans les couches supérieures (au-dessus de 700 hPa). Les températures sont négatives. C'est là que se forment les flocons de neige.", 'button_label': "Compris, étape 1/3 →", 'explanation': "C'est ici que se forment les flocons de neige initiaux. Pour l'instant, tout est normal."},
            {'action_id': 'conceptual', 'title': "Étape 2 : La Couche Chaude qui fait tout fondre", 'instruction': "Regardez maintenant la couche moyenne (~850 hPa). La température dépasse 0°C. C'est le problème : les flocons fondent et se transforment en pluie.", 'button_label': "Je vois, étape 2/3 →", 'explanation': "Lorsque les flocons de neige tombent à travers cette couche chaude, ils fondent et se transforment en gouttes de pluie."},
            {'action_id': 'conceptual', 'title': "Étape 3 : Regel en Surface", 'instruction': "Enfin, près du sol, la température redevient négative. Les gouttes de pluie regèlent juste avant de toucher le sol.", 'button_label': "Compris, étape 3/3 →", 'explanation': "C'est ce qui produit le grésil (sleet) ou la dangereuse pluie verglaçante."},
            {'action_id': 'conceptual', 'title': 'Conclusion et Défi Final', 'instruction': "Vous avez analysé un profil classique de grésil ! Vous savez maintenant qu'une couche chaude intermédiaire est la coupable.", 'button_label': "Terminer le Tutoriel", 'explanation': "Défi : Maintenant que vous avez terminé, cliquez sur 'Terminer'. Utilisez l'outil '❄️ Refroidir Couche Moyenne' dans la barre latérale et vous verrez comment vous transformez ce profil en une parfaite chute de neige !"},
        ]
    }

def start_tutorial(scenario_name):
    st.session_state.sandbox_mode = 'tutorial'
    st.session_state.tutorial_active = True
    st.session_state.tutorial_scenario = scenario_name
    st.session_state.tutorial_step = 0
    if scenario_name == 'grésil':
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
    
    st.title("🧪 Laboratoire de Sondages - Mode Tutoriel")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown(f"### Tutoriel: {scenario.replace('_', ' ').title()}")
            st.markdown("---")
            if step_index >= len(steps):
                st.success("🎉 Félicitations, vous avez terminé le tutoriel ! 🎉")
                if st.button("Terminer et Voir le Résultat", use_container_width=True, type="primary"):
                    exit_tutorial(); st.rerun()
            else:
                current_step = steps[step_index]
                st.markdown(f"#### {current_step['title']}")
                with st.container(border=True):
                    st.markdown(current_step['instruction'])
                    if st.button(current_step['button_label'], key=f"tut_action_{step_index}", use_container_width=True, type="primary"):
                        if current_step['action_id'] != 'conceptual': apply_profile_modification(current_step['action_id'])
                        st.session_state.tutorial_step += 1
                        st.rerun()
                st.markdown(f"*{current_step['explanation']}*")
        with col2:
            chat_log, _ = generate_tutorial_analysis(scenario, step_index)
            css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; height: 350px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.utilisateur { background-color: #dcf8c6; align-self: flex-end; }.analyste { background-color: #ffffff; }.système { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.utilisateur strong { color: #005C4B; }</style>"""
            html_chat = "<h6>Assistant d'Analyse</h6><div class='chat-container'>"
            for speaker, message in chat_log:
                css_class = speaker.lower()
                html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'utilisateur' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            html_chat += "</div>"
            st.markdown(css_styles + html_chat, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Abandonner le Tutoriel", use_container_width=True):
            exit_tutorial(); st.rerun()

def show_sandbox_selection_screen():
    st.title("🧪 Bienvenue au Laboratoire !")
    st.markdown("Choisissez comment vous voulez commencer. Vous pouvez suivre un tutoriel guidé pour apprendre les concepts clés ou passer directement au mode libre pour expérimenter par vous-même.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🌪️ Tutoriel : Supercellule</h4><p>Apprenez à créer un environnement avec une instabilité explosive et le cisaillement nécessaire pour les orages les plus violents et organisés.</p></div>""", unsafe_allow_html=True)
        if st.button("Commencer le Tutoriel Supercellule", use_container_width=True): 
            start_tutorial('supercellule'); st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>💧 Tutoriel : Grésil</h4><p>Analysez une situation de grésil, identifiez la couche chaude coupable et apprenez comment transformer la précipitation en neige.</p></div>""", unsafe_allow_html=True)
        if st.button("Commencer le Tutoriel Grésil", use_container_width=True): 
            start_tutorial('grésil'); st.rerun()
    with c3:
        st.markdown("""<div class="mode-card"><h4>🛠️ Mode Libre</h4><p>Passez directement à l'action. Vous aurez le contrôle total sur le profil atmosphérique dès le début pour créer vos propres scénarios.</p></div>""", unsafe_allow_html=True)
        if st.button("Aller au Mode Libre", use_container_width=True, type="primary"):
            st.session_state.sandbox_mode = 'free'; st.rerun()
    st.markdown("---")
    if st.button("⬅️ Retour à l'accueil"):
        st.session_state.app_mode = 'welcome'; st.rerun()
        
def run_sandbox_mode():
    if 'sandbox_mode' not in st.session_state:
        st.session_state.sandbox_mode = 'selection'

    if 'sandbox_initialized' not in st.session_state:
        placeholder = st.empty();
        with placeholder.container(): show_loading_animation(); time.sleep(0.5)
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings: 
            st.error("Fichier 'sondeigproves.txt' introuvable. Assurez-vous que le fichier existe.")
            placeholder.empty(); return
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
        st.header("Boîte à Outils")
        if st.button("⬅️ Retour au Menu du Laboratoire", use_container_width=True):
            for key in ['sandbox_mode', 'tutorial_active', 'tutorial_scenario', 'tutorial_step', 'convergence_active']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
        st.markdown("---")
        st.subheader("Modifications Thermodynamiques")
        st.markdown("**Couches Basses (> 850 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Réchauffer", on_click=apply_profile_modification, args=('warm_low',), use_container_width=True); c2.button("❄️ Refroidir", on_click=apply_profile_modification, args=('cool_low',), use_container_width=True); c1.button("💧 Humidifier", on_click=apply_profile_modification, args=('moisten_low',), use_container_width=True); c2.button("💨 Assécher", on_click=apply_profile_modification, args=('dry_low',), use_container_width=True)
        st.markdown("**Couches Moyennes (850-600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Réchauffer", on_click=apply_profile_modification, args=('warm_mid',), use_container_width=True, key='w_mid'); c2.button("❄️ Refroidir", on_click=apply_profile_modification, args=('cool_mid',), use_container_width=True, key='c_mid'); c1.button("💧 Humidifier", on_click=apply_profile_modification, args=('moisten_mid',), use_container_width=True, key='m_mid'); c2.button("💨 Assécher", on_click=apply_profile_modification, args=('dry_mid',), use_container_width=True, key='d_mid')
        st.markdown("**Couches Hautes (< 600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Réchauffer", on_click=apply_profile_modification, args=('warm_high',), use_container_width=True, key='w_h'); c2.button("❄️ Refroidir", on_click=apply_profile_modification, args=('cool_high',), use_container_width=True, key='c_h'); c1.button("💧 Humidifier", on_click=apply_profile_modification, args=('moisten_high',), use_container_width=True, key='m_h'); c2.button("💨 Assécher", on_click=apply_profile_modification, args=('dry_high',), use_container_width=True, key='d_h')
        st.markdown("---"); st.subheader("Outils Globaux et Vent")
        c1, c2 = st.columns(2); c1.button("🔥 Réchauffer Tout", on_click=apply_profile_modification, args=('warm_all',), use_container_width=True); c2.button("🧊 Refroidir Tout", on_click=apply_profile_modification, args=('cool_all',), use_container_width=True)
        c1.button("💦 Humidifier Tout", on_click=apply_profile_modification, args=('moisten_all',), use_container_width=True); c2.button("🌬️ Assécher Tout", on_click=apply_profile_modification, args=('dry_all',), use_container_width=True)
        st.button("Couvercle (Inversion)", on_click=apply_profile_modification, args=('add_inversion',), use_container_width=True)
        st.markdown("**Cisaillement du Vent**")
        c1, c2, c3 = st.columns(3); c1.button("🌪️ Basses", on_click=apply_profile_modification, args=('add_shear_low',), use_container_width=True); c2.button("🌪️ Moyennes", on_click=apply_profile_modification, args=('add_shear_mid',), use_container_width=True); c3.button("🌪️ Hautes", on_click=apply_profile_modification, args=('add_shear_high',), use_container_width=True)
        def reset_wind_profile():
            st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.button("🚫 Réinitialiser Vents", on_click=reset_wind_profile, use_container_width=True)
        st.markdown("---")
        if st.button("🔄 Réinitialiser au Profil Original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_p_levels, st.session_state.sandbox_t_profile, st.session_state.sandbox_td_profile = data['p_levels'].copy(), data['t_initial'].copy(), data['td_initial'].copy()
            reset_wind_profile()
            if st.session_state.get('tutorial_active', False): exit_tutorial()
            if 'convergence_active' in st.session_state: st.session_state.convergence_active = False
            st.rerun()

    if st.session_state.sandbox_mode == 'selection':
        show_sandbox_selection_screen()
    elif st.session_state.sandbox_mode == 'tutorial':
        show_tutorial_interface()
    elif st.session_state.sandbox_mode == 'free':
        st.title("🧪 Laboratoire de Sondages - Mode Libre")
        show_full_analysis_view(p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile, 
                               td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws, 
                               wd=st.session_state.sandbox_wd, obs_time="Sondage d'Essai - Mode Laboratoire",
                               is_sandbox_mode=True)

# =========================================================================
# === POINT D'ENTRÉE DE L'APPLICATION =====================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Analyseur de Sondages")
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'

    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
