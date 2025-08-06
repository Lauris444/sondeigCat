# ====================================================================================
# === TEMPTESTES.CAT - VISOR DE SONDEJOS ATMOSFÈRICS V2.1 ============================
# ====================================================================================
# AQUEST SCRIPT NECESSITA FITXERS ADDICIONALS PER FUNCIONAR CORRECTAMENT:
# 1. Per la pantalla de benvinguda:
#    - storm_background.mp4 (vídeo d'una tempesta)
# 2. Pel mode en viu (si s'utilitza):
#    - Fitxers de dades horàries com 12am.txt, 1pm.txt, etc.
#
# Els escenaris del mode Laboratori ara estan integrats directament al codi.
# ====================================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import random
import os
import re
import threading
import base64
import io
from datetime import datetime, timezone
import pytz

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
integrator_lock = threading.Lock()

# =============================================================================
# === NOU: BASE DE DADES D'ESCENARIS PREDEFINITS ===============================
# =============================================================================

PRESET_SCENARIOS = {
    "Perfil Base (Plana de Vic, Tardor)": {
        'observation_time': "Escenari: Perfil base neutre, tardor a la Plana de Vic.",
        'p_levels': np.array([1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]) * units.hPa,
        't_initial': np.array([12, 10, 8, 4, 0, -5, -12, -21, -30, -45, -55, -60]) * units.degC,
        'td_initial': np.array([8, 7, 5, 1, -5, -15, -22, -30, -40, -55, -65, -70]) * units.degC,
        'wind_dir_deg': np.array([270, 280, 290, 300, 310, 320, 320, 310, 300, 290, 280, 270]) * units.degrees,
        'wind_speed_kmh': np.array([10, 15, 20, 25, 30, 40, 55, 70, 90, 120, 140, 150]) * units.kph
    },
    "Nevada a Barcelona": {
        'observation_time': "Escenari: Onada de fred siberià amb nevada a cota 0.",
        'p_levels': np.array([1015, 985, 948, 908, 870, 825, 775, 725, 675, 575, 500, 400, 300, 250, 200]) * units.hPa,
        't_initial': np.array([1.1, 0.6, 0.1, -0.5, -1.1, -1.8, -2.5, -4.0, -6.2, -8.9, -15.8, -21.5, -30.2, -42.6, -50.1, -58.5]) * units.degC,
        'td_initial': np.array([0.7, 0.2, -0.2, -0.8, -1.4, -2.1, -2.9, -4.5, -6.8, -9.6, -17.3, -24.5, -35.1, -50.1, -58.4, -64.7]) * units.degC,
        'wind_dir_deg': np.array([55, 45, 40, 35, 30, 25, 10, 295, 305, 311, 315, 320, 335, 345, 348, 350]) * units.degrees,
        'wind_speed_kmh': np.array([15, 20, 25, 28, 30, 34, 39, 47, 56, 65, 76, 90, 100, 125, 135, 140]) * units.kph
    },
    "Supercèl·lula (Oklahoma, EUA)": {
        'observation_time': "Escenari: Clàssic brot de tornados a les planures d'Oklahoma.",
        'p_levels': np.array([980, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150]) * units.hPa,
        't_initial': np.array([30, 28, 25, 21, 16, 10, 2, -7, -18, -33, -45, -58, -65]) * units.degC,
        'td_initial': np.array([24, 23, 22, 20, 15, 9, -1, -9, -22, -38, -50, -62, -70]) * units.degC,
        'wind_dir_deg': np.array([160, 170, 185, 200, 210, 230, 245, 260, 270, 280, 285, 290, 295]) * units.degrees,
        'wind_speed_kmh': np.array([30, 35, 45, 55, 65, 80, 100, 120, 140, 160, 170, 180, 190]) * units.kph
    },
    "Onada de Calor (Sàhara)": {
        'observation_time': "Escenari: Dia extremadament càlid i sec al desert del Sàhara.",
        'p_levels': np.array([1010, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]) * units.hPa,
        't_initial': np.array([48, 44, 40, 36, 31, 22, 13, 2, -10, -25, -38, -52]) * units.degC,
        'td_initial': np.array([-5, -8, -12, -15, -20, -28, -35, -42, -50, -60, -70, -80]) * units.degC,
        'wind_dir_deg': np.array([60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150, 160]) * units.degrees,
        'wind_speed_kmh': np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]) * units.kph
    },
    "Ambient Tropical (Malàisia)": {
        'observation_time': "Escenari: Tarda humida i inestable a la selva de Malàisia.",
        'p_levels': np.array([1010, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150]) * units.hPa,
        't_initial': np.array([32, 29, 26, 23, 20, 13, 6, -2, -11, -24, -35, -48, -63]) * units.degC,
        'td_initial': np.array([28, 26, 24, 21, 18, 11, 4, -4, -13, -27, -38, -51, -66]) * units.degC,
        'wind_dir_deg': np.array([220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100]) * units.degrees,
        'wind_speed_kmh': np.array([5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35]) * units.kph
    },
    "Medicà (Estil Huracà Mediterrani)": {
        'observation_time': "Escenari: Un potent medicà aproximant-se a les Illes Balears.",
        'p_levels': np.array([990, 950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]) * units.hPa,
        't_initial': np.array([19, 17, 14, 11, 8, 3, -4, -12, -22, -35, -46, -58]) * units.degC,
        'td_initial': np.array([18.5, 16.5, 13.5, 10.5, 7.5, 2.5, -4.5, -12.5, -22.5, -35.5, -46.5, -58.5]) * units.degC,
        'wind_dir_deg': np.array([90, 80, 70, 50, 30, 360, 340, 320, 300, 280, 270, 260]) * units.degrees,
        'wind_speed_kmh': np.array([110, 120, 130, 125, 120, 110, 100, 90, 80, 70, 60, 50]) * units.kph
    },
    "Ambient Polar (Antàrtida)": {
        'observation_time': "Escenari: Hivern a l'estació Amundsen-Scott, Pol Sud.",
        'p_levels': np.array([680, 650, 600, 550, 500, 400, 300, 200, 150, 100]) * units.hPa,
        't_initial': np.array([-55, -57, -60, -63, -65, -70, -75, -78, -80, -82]) * units.degC,
        'td_initial': np.array([-60, -62, -65, -68, -70, -75, -80, -83, -85, -87]) * units.degC,
        'wind_dir_deg': np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) * units.degrees,
        'wind_speed_kmh': np.array([40, 45, 50, 55, 60, 65, 70, 75, 80, 85]) * units.kph
    },
    "Cim de l'Everest": {
        'observation_time': "Escenari: Condicions al cim de l'Everest durant la temporada d'escalada.",
        'p_levels': np.array([337, 325, 300, 275, 250, 225, 200, 175, 150]) * units.hPa,
        't_initial': np.array([-25, -27, -31, -35, -40, -45, -51, -57, -63]) * units.degC,
        'td_initial': np.array([-35, -37, -41, -45, -50, -55, -61, -67, -73]) * units.degC,
        'wind_dir_deg': np.array([280, 285, 290, 295, 300, 305, 310, 315, 320]) * units.degrees,
        'wind_speed_kmh': np.array([130, 140, 155, 170, 180, 175, 160, 150, 140]) * units.kph
    }
}


# =========================================================================
# === 0. FUNCIONS AUXILIARS PER MULTIMÈDIA (PANTALLA DE BENVINGUDA) ======
# =========================================================================

def get_file_as_base64(file_path):
    """Codifica un fitxer (vídeo, imatge) a Base64 per incrustar-lo a l'HTML."""
    if not os.path.exists(file_path):
        # Aquesta advertència es mostrarà a la consola on s'executa Streamlit
        print(f"Advertència: El fitxer '{file_path}' no s'ha trobat.")
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

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

# =========================================================================
# === 2. FUNCIONS AUXILIARS PER GESTIÓ DE FITXERS HORARIS (CORREGIT) ======
# =========================================================================

def filename_to_24h_sort_key(filename):
    """Converteix un nom de fitxer com '1pm.txt' a un enter (13) per poder ordenar-lo."""
    match = re.match(r'(\d+)(am|pm)\.txt', filename.lower())
    if not match: return -1  # Retorna -1 per a fitxers amb format incorrecte
    hour, period = int(match.group(1)), match.group(2)
    if period == 'am':
        return 0 if hour == 12 else hour  # 12am (mitjanit) és l'hora 0
    else:  # period == 'pm'
        return 12 if hour == 12 else hour + 12  # 12pm (migdia) és l'hora 12

def hour_24_to_filename(hour):
    """Converteix una hora en format 24h (0-23) a un nom de fitxer com '12am.txt'."""
    if hour == 0: return '12am.txt'
    elif hour < 12: return f'{hour}am.txt'
    elif hour == 12: return '12pm.txt'
    else: return f'{hour - 12}pm.txt'

# =========================================================================
# === 3. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
# =========================================================================

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
        p, ws, wd = p_levels, wind_speed, wind_dir
        u, v = mpcalc.wind_components(ws, wd)
        heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
        valid_mask = ~np.isnan(heights_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2: return 0.0, 0.0, 0.0, 0.0
        p_c, u_c, v_c, h_c = p[valid_mask], u[valid_mask], v[valid_mask], heights_raw[valid_mask]
        _, unique_indices = np.unique(h_c.m, return_index=True)
        if len(unique_indices) < 2: return 0.0, 0.0, 0.0, 0.0
        p_u, u_u, v_u, h_u = p_c[unique_indices], u_c[unique_indices], v_c[unique_indices], h_c[unique_indices]
        h_min, h_max = h_u.m.min(), min(h_u.m.max(), 16000)
        if h_max <= h_min: return 0.0, 0.0, 0.0, 0.0
        h_interp = np.arange(h_min, h_max, 50) * units.meter
        u_i = np.interp(h_interp.m, h_u.m, u_u.m) * units('m/s')
        v_i = np.interp(h_interp.m, h_u.m, v_u.m) * units('m/s')
        u_6, v_6 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        u_1, v_1 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter)[0].m
        return s_0_6, s_0_1, srh_0_1, srh_0_3
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, base_km, top_km, pwat_0_4):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    if fz_h < 1500 or t_profile[0].m < 5:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'sleet'
    elif cape.m > 3000:
        precipitation_type = 'hail'
    elif cape.m > 500:
        precipitation_type = 'rain'
    elif "Nimbostratus" in cloud_type:
        precipitation_type = 'rain'
    elif lfc_p and el_p and (lfc_p.magnitude > el_p.magnitude if lfc_p and el_p else False):
        precipitation_type = 'virga'
    chat_log = [("Tempestes.cat", f"Hola! Detecto una situació compatible amb la formació de núvols de tipus **{cloud_type}**.")]
    if cloud_type == "Hivernal":
        chat_log.extend([("Yo", f"Veig una isoterma 0°C molt baixa, a {fz_h:.0f}m."),("Tempestes.cat", "Exacte. Això, combinat amb la humitat en nivells baixos, és el factor clau."),("Yo", f"La temperatura a la superfície és de {t_profile[0].m:.1f}°C. Què implica?"),])
        if t_profile[0].m <= 0.5:
            chat_log.append(("Tempestes.cat", "Amb temperatures negatives o properes a 0°C a tots els nivells, la precipitació serà neu fins a cotes molt baixes."))
        else:
            chat_log.append(("Tempestes.cat", "Compte. Hi ha una petita capa càlida just sobre la superfície. Això pot provocar que la neu es fongui i es torni a congelar en contacte amb el terra (pluja gelant), un fenomen molt perillós."))
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([("Yo", f"El CAPE és altíssim, {cape.m:.0f} J/kg. Què significa?"),("Tempestes.cat", f"És l'energia disponible per a la tempesta. Un valor tan alt indica un potencial per a corrents ascendents extremament violents, capaços de sostenir calamarsa de gran mida."),("Yo", "I el cisallament del vent? Veig valors elevats."),("Tempestes.cat", f"Correcte. El cisallament de {shear_0_6:.0f} m/s i l'helicitat (SRH) de {srh_0_3:.0f} m²/s² són els ingredients que permetran que la tempesta s'organitzi i roti, formant una supercèl·lula."),("Yo", "Quin és el risc principal?"),("Tempestes.cat", f"Molt alt. Cal esperar calamarsa de gran mida (>4cm), ratxes de vent destructives i, amb un SRH 0-1km de {srh_0_1:.1f}, hi ha un risc significatiu de formació de tornados.")])
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
        chat_log.extend([("Yo", f"El CAPE és de {cape.m:.0f} J/kg. És molt?"),("Tempestes.cat", "És un valor moderat a alt. Indica que hi ha energia suficient per a tempestes fortes, però no explosives."),("Yo", "Per què no s'organitzen com una supercèl·lula?"),("Tempestes.cat", f"El cisallament ({shear_0_6:.0f} m/s) és massa feble. Les tempestes competiran entre elles en lloc de formar una única estructura organitzada. Si són Castellanus, la convecció s'inicia a nivells més alts."),("Yo", "Quins fenòmens podem esperar?"),("Tempestes.cat", "Principalment xàfecs intensos i calamarsa de mida petita a moderada. En el cas dels Castellanus, el principal risc són els esclafits secs (downbursts) si la base està molt elevada.")])
    elif "Nimbostratus" in cloud_type:
        chat_log.extend([("Yo", "Veig molta humitat a capes baixes però gairebé gens d'inestabilitat (CAPE)."),("Tempestes.cat", f"Exacte. No hi ha motor convectiu (CAPE de {cape.m:.0f} J/kg), però l'atmosfera està saturada en una capa molt gruixuda. Això és típic de la pluja estratiforme, associada a fronts."),("Yo", "Com de potent serà la pluja? Depèn de l'aigua precipitable (PWAT), oi?"),])
        if "Intens" in cloud_type:
            chat_log.append(("Tempestes.cat", f"Sí. El PWAT a la capa 0-4 km és de **{pwat_0_4.m:.1f} mm**, un valor molt alt. Això es traduirà en pluges **contínues i abundants**, amb risc d'acumulacions importants."))
        elif "Moderat" in cloud_type:
            chat_log.append(("Tempestes.cat", f"Correcte. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És un valor considerable que alimentarà xàfecs **moderats i persistents**, el que popularment anomenem 'petacs' de pluja."))
        else:
            chat_log.append(("Tempestes.cat", f"Exactament. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És suficient per a **ruixats febles i intermitents** o plugims, però no s'esperen grans quantitats."))
    else:
        chat_log.extend([("Yo", " sembla un dia tranquil, oi?"),("Tempestes.cat", f"Sí. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),("Yo", "Veurem algun núvol?"),("Tempestes.cat", f"Probablement només alguns {cloud_type} senza cap mena de desenvolupament vertical ni risc de precipitació.")])
    return chat_log, precipitation_type

def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    if fz_h < 1500 or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades. Extremi les precaucions.", "dodgerblue"
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
                    return "AVÍS PER PLUGES INTENSES", "Risc de pluges persistents i fortes. Possible acumulació d'aigua.", "darkblue"
                elif pwat_layer.m > 15:
                    return "AVÍS PER PLUJA MODERADA", "Cel cobert amb pluja contínua i moderada. Visibilitat reduïda.", "steelblue"
                else:
                    return "PREVISIÓ DE PLUJA FEBLE", "S'esperen plugims o ruixats febles i intermitents.", "cadetblue"
    except Exception:
        pass
    if cape.m >= 1000:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        if srh_0_1 > 150 and shear_0_6 > 15:
            return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        if lfc_h > 3000:
            return "AVÍS PER TEMPESTES DE BASE ALTA", "Nuclis de base alta. Risc de ratxes de vent fortes i sobtades (downbursts).", "darkorange"
        if cape.m > 2000:
            return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa. Protegiu vehicles.", "purple"
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"

# =========================================================================
# === 4. FUNCIONS DE DIBUIX ===============================================
# =========================================================================

def create_logo_figure():
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    bg_color, cloud_color, senyera_red, senyera_yellow = '#F5F1E9', '#4B2A4B', '#DA121A', '#FCDD09'
    ax.add_patch(Circle((5, 5), 5, facecolor=bg_color))
    cloud_verts = [(2, 5.8), (1.5, 6.8), (2.5, 7.8), (4, 8.3), (6, 8.3), (7.5, 7.8), (8.5, 6.8), (8, 5.8), (7, 5.3), (3, 5.3)]
    ax.add_patch(Polygon(cloud_verts, facecolor=cloud_color, zorder=10))
    ax.text(5, 6.6, 'tempestes.cat', ha='center', va='center', fontsize=3.3, color='white', weight='bold', fontfamily='sans-serif', zorder=20)
    bar_heights, start_x, bar_width, rain_start_y = [0.8, 1.0, 0.9, 0.7, 0.95, 0.85, 0.6, 0.75, 0.5], 3.0, 0.4, 5.3
    for i, h in enumerate(bar_heights):
        x_pos, color, bar_height = start_x + i * bar_width, senyera_red if i % 2 == 0 else senyera_yellow, h * 4.0
        ax.add_patch(Rectangle((x_pos + 0.05, rain_start_y - bar_height - 0.05), bar_width, bar_height, facecolor='black', alpha=0.3, lw=0, zorder=4))
        ax.add_patch(Rectangle((x_pos, rain_start_y - bar_height), bar_width, bar_height, facecolor=color, lw=0, zorder=5))
    return fig

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
    if "Intens" in cloud_type:
        color, alpha = '#808080', 0.95
    elif "Moderat" in cloud_type:
        color, alpha = '#a9a9a9', 0.9
    else:
        color, alpha = '#c0c0c0', 0.85
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

def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    _, _, lcl_p, lcl_h, _, _, _, el_h, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if not lcl_p: return None, None
    cloud_base_km = lcl_h / 1000.0
    if convergence_active:
        cloud_top_km = el_h / 1000.0 if el_h > lcl_h else cloud_base_km
    else:
        try:
            rh = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile)
            indices_above_lcl = np.where(p_levels <= lcl_p)[0]
            p_top = p_levels[-1]
            if len(indices_above_lcl) > 0:
                for idx in indices_above_lcl:
                    if rh[idx] < 0.5: p_top = p_levels[idx]; break
            cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
        except: cloud_top_km = cloud_base_km
    return (cloud_base_km, cloud_top_km) if cloud_base_km and cloud_top_km and cloud_top_km > cloud_base_km else (None, None)

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
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    td_profile = np.minimum(t_profile, td_profile)
    skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
    parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
    skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
    skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='Tª Bombolla Humida')
    skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
    skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
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
    _draw_saturation_layers(ax, p_levels, t_profile, td_profile)

    if base_km is not None and top_km is not None:
        if "Nimbostratus" in cloud_type:
            _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif cloud_type == "Cumulonimbus (Multicèl·lula)" or cloud_type == "Supercèl·lula":
            _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus":
            _draw_cumulus_castellanus(ax, base_km, top_km)
        elif cloud_type == "Cumulus Mediocris":
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
    if not base_km or not top_km or cape.m < 100:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='white', bbox=dict(facecolor='darkblue', alpha=0.7))
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

# =========================================================================
# === 5. FUNCIONS PER A L'ESTRUCTURA DE L'APP =============================
# =========================================================================

def run_display_logic(p, t, td, ws, wd, obs_time):
    logo_fig = create_logo_figure()
    st.markdown(f"#### {obs_time}")
    convergence_active = st.session_state.get('convergence_active', True)
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
    if sfc_temp.m < 5 or fz_h < 1500: cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape.m < 350:
        if pwat_0_4.m > 25: cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4.m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150: cloud_type = "Supercèl·lula"
    elif cape.m > 500:
        cloud_type = "Cumulonimbus (Multicèl·lula)"
        if lfc_h >= 3000: cloud_type = "Castellanus"
    elif base_km and top_km:
        if (top_km - base_km) > 2.0 and lfc_h < 3000: cloud_type = "Cumulus Mediocris"
        elif (top_km - base_km) > 0: cloud_type = "Cumulus Fractus"
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    st.subheader("Diagrama Skew-T", anchor=False)
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    st.divider()
    chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])
    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_buffer = io.BytesIO()
        logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        logo_base64 = base64.b64encode(logo_buffer.getvalue()).decode()
        css_styles = f"""<style>.chat-container {{ background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: Arial, sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }}.message-row {{ display: flex; align-items: flex-end; gap: 10px; }}.message-row-right {{ justify-content: flex-end; }}.message {{ padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }}.yo {{ background-color: #0078D4; color: white; }}.tempestes-cat {{ background-color: #FFFFFF; border: 1px solid #e0e0e0; }}.sistema {{ background-color: #E1F2FB; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }}.message strong {{ display: block; margin-bottom: 3px; font-weight: bold; }}.yo strong {{color: #FFFFFF;}}.tempestes-cat strong {{ color: #075E54; }}.profile-pic {{ width: 40px; height: 40px; border-radius: 50%; object-fit: cover; }}.online-status {{ text-align: center; font-size: 0.9em; color: #666; padding: 5px; }}</style>"""
        html_chat = "<div class='online-status'>Tempestes.cat • en línia</div><div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower().replace('.', '-')
            if speaker == "Tempestes.cat":
                html_chat += f"""<div class="message-row"><img src="data:image/png;base64,{logo_base64}" class="profile-pic"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            elif speaker == "Yo":
                html_chat += f"""<div class="message-row message-row-right"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            else:
                html_chat += f"<div class='message sistema'>{message}</div>"
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A"); param_cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); param_cols[3].metric("Shear 0-6", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1", f"{srh_0_1:.1f} m²/s²"); param_cols[1].metric("SRH 0-3", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm")
        rh_display = "N/A"
        try:
            rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("RH Mitja 0-4km", rh_display)
    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type)
            st.pyplot(fig_clouds, use_container_width=True)
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(p, t, td, ws, wd, convergence_active)
            st.pyplot(fig_structure, use_container_width=True)
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)

# =========================================================================
# === 6. EXECUCIÓ DELS MODES DE L'APP =====================================
# =========================================================================

def show_welcome_screen():
    # =================================================================================
    # === PAS 1: PREPARA EL FITXER DE VÍDEO ==========================================
    # =================================================================================
    video_path = "storm_background.mp4"
    video_base64 = get_file_as_base64(video_path)

    # =================================================================================
    # === PAS 2: DEFINEIX L'HTML I EL CSS PER A L'ESTIL ESPECTACULAR ===================
    # =================================================================================
    
    if not video_base64:
        st.error(f"Error de Càrrega: No s'ha pogut carregar el vídeo de fons '{video_path}'. "
                 "Si us plau, assegura't que el fitxer existeix a la mateixa carpeta que l'script i reinicia.")
        return

    page_bg_img = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@300;400;700&display=swap');
    .stApp {{ background: #000; }}
    #video-background {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; object-fit: cover; z-index: -2; filter: blur(2px) brightness(0.7); }}
    .video-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(0deg, rgba(10, 20, 40, 0.8) 0%, rgba(20, 40, 60, 0.5) 100%); z-index: -1; }}
    .welcome-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; color: white; padding-top: 2rem; padding-bottom: 2rem; }}
    .welcome-container h1 {{ font-family: 'Teko', sans-serif; font-size: 6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 4px; text-shadow: 0 0 15px rgba(255, 255, 255, 0.5), 0 0 25px rgba(0, 150, 255, 0.4); margin-bottom: 0; }}
    .welcome-container p.subtitle {{ font-family: 'Teko', sans-serif; font-size: 1.8rem; font-weight: 300; letter-spacing: 2px; margin-top: -10px; opacity: 0.9; }}
    .mode-card {{ background: rgba(15, 32, 51, 0.5); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 2rem; margin-top: 1rem; margin-bottom: 1rem; text-align: center; backdrop-filter: blur(10px) saturate(120%); -webkit-backdrop-filter: blur(10px) saturate(120%); transition: all 0.3s ease; height: 280px; display: flex; flex-direction: column; justify-content: space-between; }}
    .mode-card:hover {{ transform: translateY(-10px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); border-color: rgba(0, 191, 255, 0.7); }}
    .mode-card h3 {{ font-family: 'Teko', sans-serif; font-size: 2.5rem; margin-bottom: 1rem; color: #00BFFF; }}
    .mode-card p {{ font-size: 1rem; line-height: 1.6; color: rgba(255, 255, 255, 0.85); flex-grow: 1; }}
    .stButton > button {{ width: 100%; }}
    </style>
    <video id="video-background" autoplay loop muted>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <div class="video-overlay"></div>
    """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # =================================================================================
    # === PAS 3: MOSTRA EL CONTINGUT DE LA PANTALLA DE BENVINGUDA =====================
    # =================================================================================
    
    with st.container():
        st.markdown(
            """
            <div class="welcome-container">
                <h1>El Cor de la Tempesta</h1>
                <p class="subtitle">Analitza, prediu i comprèn els fenòmens atmosfèrics</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="mode-card">
                <div>
                    <h3>🛰️ Mode en Viu</h3>
                    <p>
                    Connecta't a les dades més recents. Visualitza els sondejos atmosfèrics del dia,
                    hora a hora, per seguir l'evolució del temps real com un professional.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Entra al Centre de Comandament", use_container_width=True, key="live_button"):
            st.session_state.app_mode = 'live'
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="mode-card">
                <div>
                    <h3>🧪 Laboratori de Tempestes</h3>
                    <p>
                    Converteix-te en l'arquitecte del temps. Modifica paràmetres, carrega escenaris
                    extrems i descobreix com neixen les supercèl·lules, les nevades o les pluges torrencials.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Inicia l'Experiment", use_container_width=True, type="primary", key="sandbox_button"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def run_live_mode():
    st.title("🛰️ Mode en Viu: Sondejos Reals")
    
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Mode Viu)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
        if st.button("Actualitzar Dades Ara"):
            st.cache_data.clear()
            st.rerun()

    @st.cache_data(ttl=300) # Cache de 5 minuts
    def get_sounding_data():
        all_files_in_dir = os.listdir('.')
        valid_sounding_files = [f for f in all_files_in_dir if filename_to_24h_sort_key(f) != -1]
        
        if not valid_sounding_files:
            return [], None, None, None
            
        sorted_files = sorted(valid_sounding_files, key=filename_to_24h_sort_key)
        now_utc = datetime.now(timezone.utc)
        now_madrid = now_utc.astimezone(pytz.timezone('Europe/Madrid'))
        current_hour_filename = hour_24_to_filename(now_utc.hour)
        
        return sorted_files, now_utc, now_madrid, current_hour_filename

    existing_files, utc_time, madrid_time, current_hour_file = get_sounding_data()

    if not existing_files:
        st.error("No s'ha trobat cap arxiu de sondeig vàlid (ex: 1am.txt, 2pm.txt) al directori de l'aplicació.")
        return

    # Lògica d'inicialització i selecció d'índex
    if 'live_initialized' not in st.session_state or st.session_state.get('last_known_file_list') != existing_files:
        st.session_state.last_known_file_list = existing_files
        try:
            initial_index = existing_files.index(current_hour_file)
            st.session_state.info_msg = f"S'ha carregat automàticament el sondeig per a les **{utc_time.hour:02d}:00 UTC**."
        except ValueError:
            available_hours = [int(filename_to_24h_sort_key(f)) for f in existing_files]
            past_or_current_hours = [h for h in available_hours if h <= utc_time.hour]
            if past_or_current_hours:
                closest_hour = max(past_or_current_hours)
                initial_index = available_hours.index(closest_hour)
                st.session_state.info_msg = (f"Sondeig per a les {utc_time.hour:02d}:00 UTC no trobat. "
                                              f"Mostrant el més recent disponible: **{existing_files[initial_index]}**.")
            else:
                initial_index = 0
                st.session_state.info_msg = f"Mostrant el primer sondeig disponible: **{existing_files[initial_index]}**."

        st.session_state.sounding_index = initial_index
        st.session_state.live_initialized = True
    
    with st.sidebar:
        st.info(f"**Hora Local:** {madrid_time.strftime('%H:%M:%S')}\n\n"
                f"**Hora UTC:** {utc_time.strftime('%H:%M:%S')}")
        if 'info_msg' in st.session_state:
            st.success(st.session_state.info_msg)

    def sync_index_from_selectbox():
        st.session_state.sounding_index = existing_files.index(st.session_state.selectbox_widget)
        if 'info_msg' in st.session_state:
            del st.session_state.info_msg
            
    selected_file = st.sidebar.selectbox("Selecciona un sondeig manualment:", 
                                         options=existing_files, 
                                         index=st.session_state.get("sounding_index", 0), 
                                         key='selectbox_widget', 
                                         on_change=sync_index_from_selectbox)
    
    soundings = parse_all_soundings(selected_file)
    if not soundings:
        st.error(f"No s'han pogut carregar les dades del sondeig de '{selected_file}'.")
        return
        
    data = soundings[0]
    run_display_logic(p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                      ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                      obs_time=data.get('observation_time', f"Sondeig de {selected_file}"))

def run_sandbox_mode():
    st.title("🧪 Laboratori de Sondejos")
    
    if 'sandbox_initialized' not in st.session_state:
        # Carrega el perfil base des del nostre nou diccionari
        data = PRESET_SCENARIOS["Perfil Base (Plana de Vic, Tardor)"]
        st.session_state.sandbox_p_levels = data['p_levels'].copy()
        st.session_state.sandbox_t_profile = data['t_initial'].copy()
        st.session_state.sandbox_td_profile = data['td_initial'].copy()
        st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True

    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Laboratori)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
        
        st.subheader("Escenaris Predefinits")
        
        # El selectbox ara utilitza les claus del nostre diccionari de presets
        selected_preset = st.selectbox("Tria un escenari:", list(PRESET_SCENARIOS.keys()))
        
        if st.button("Carregar Escenari", use_container_width=True):
            # Carrega les dades directament del diccionari
            data = PRESET_SCENARIOS[selected_preset]
            st.session_state.sandbox_p_levels = data['p_levels'].copy()
            st.session_state.sandbox_t_profile = data['t_initial'].copy()
            st.session_state.sandbox_td_profile = data['td_initial'].copy()
            st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
            # Actualitza els sliders amb els nous valors del preset
            st.session_state.t_slider = data['t_initial'][0].m
            st.session_state.td_slider = data['td_initial'][0].m
            st.rerun()

        st.markdown("---")
        st.subheader("Modificació Manual (en viu)")
        
        t_profile_mod = st.session_state.sandbox_t_profile.copy()
        td_profile_mod = st.session_state.sandbox_td_profile.copy()
        
        sfc_t_val = st.session_state.get('t_slider', t_profile_mod[0].m)
        sfc_td_val = st.session_state.get('td_slider', td_profile_mod[0].m)

        new_sfc_t = st.slider("🌡️ Temperatura en Superfície (°C)", -40.0, 50.0, sfc_t_val, 0.5, key="t_slider")
        new_sfc_td = st.slider("💧 Punt de Rosada en Superfície (°C)", -40.0, new_sfc_t, sfc_td_val, 0.5, key="td_slider")
        
        t_profile_mod[0] = new_sfc_t * units.degC
        td_profile_mod[0] = new_sfc_td * units.degC
        
        st.markdown("**Consells:**")
        if new_sfc_t - new_sfc_td < 2:
            st.info("Humitat molt alta a la superfície pot generar núvols baixos i boira.")
        if new_sfc_t > 25 and new_sfc_td > 18:
            st.success("Condicions molt favorables per a convecció forta si hi ha inestabilitat.")
        
    run_display_logic(
        p=st.session_state.sandbox_p_levels,
        t=t_profile_mod,
        td=td_profile_mod,
        ws=st.session_state.sandbox_ws,
        wd=st.session_state.sandbox_wd,
        obs_time=PRESET_SCENARIOS[selected_preset]['observation_time'] if 'selected_preset' in locals() else "Sondeig de Prova - Mode Laboratori"
    )

# =========================================================================
# === 7. PUNT D'ENTRADA DE L'APLICACIÓ ====================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Visor de Sondejos", page_icon="⛈️")
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    
    if st.session_state.app_mode == 'welcome':
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
