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
# === 2. FUNCIONS AUXILIARS PER GESTIÓ DE FITXERS HORARIS =================
# =========================================================================

def filename_to_24h_sort_key(filename):
    """Converteix un nom de fitxer com '1pm.txt' a un enter (13) per poder ordenar-lo."""
    match = re.match(r'(\d+)(am|pm)\.txt', filename.lower())
    if not match: return -1
    hour, period = int(match.group(1)), match.group(2)
    if period == 'am': return 0 if hour == 12 else hour
    else: return 12 if hour == 12 else hour + 12

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
        chat_log.extend([("Yo", " sembla un dia tranquil, oi?"),("Tempestes.cat", f"Sí. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),("Yo", "Veurem algun núvol?"),("Tempestes.cat", f"Probablement només alguns {cloud_type} sense cap mena de desenvolupament vertical ni risc de precipitació.")])
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

def create_welcome_graphic():
    """Crea una il·lustració de benvinguda inspirada en la imatge proporcionada."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    # Dibuixar fons amb gradient
    gradient = np.zeros((10, 10, 4))
    gradient[:, :, 0] = np.linspace(0.1, 0.2, 10)  # Vermell
    gradient[:, :, 1] = np.linspace(0.1, 0.25, 10) # Verd
    gradient[:, :, 2] = np.linspace(0.2, 0.4, 10)  # Blau
    gradient[:, :, 3] = 1
    ax.imshow(np.vstack([gradient]*100), extent=[0, 16, 0, 9], aspect='auto', zorder=0)
    
    # Dibuixar núvols foscos a la part inferior
    for _ in range(30):
        x = random.uniform(-2, 18)
        y = random.uniform(0, 2.5)
        size = random.uniform(2, 6)
        brightness = random.uniform(0.05, 0.15)
        ax.add_patch(Circle((x, y), size, facecolor=(brightness, brightness, brightness+0.05), lw=0, alpha=random.uniform(0.3, 0.6), zorder=1))

    # Dibuixar el polígon en perspectiva per al Skew-T
    skew_poly = Polygon([[4, 1.5], [12, 1], [12.5, 8], [4.5, 7.5]], facecolor='white', alpha=0.9, edgecolor='gray', lw=2, zorder=2)
    ax.add_patch(skew_poly)

    # Simular contingut del Skew-T
    # Línies de la graella (isotermes inclinades)
    for t in np.arange(-100, 50, 10):
        x1, y1 = 4.2 + (t+40)*0.06, 1.3
        x2, y2 = 4.7 + (t+40)*0.06, 7.7
        ax.plot([x1, x2], [y1, y2], color='lightgray', lw=0.5, zorder=3)
    # Perfils de Temperatura i Punt de Rosada (simplificats)
    p = np.linspace(1.8, 7.0, 10)
    t = np.array([8.5, 8.2, 7.5, 6.0, 5.0, 6.5, 5.0, 3.0, 1.5, 0.5])
    td = np.array([7.0, 6.0, 4.0, 1.0, -2.0, -1.0, -5.0, -10.0, -12.0, -15.0])
    ax.plot(t, p, color='red', lw=2, zorder=4)
    ax.plot(td, p, color='blue', lw=2, zorder=4)
    # Hodògraf (simplificat)
    ax.add_patch(Circle((5.5, 6.5), 0.7, facecolor='white', edgecolor='black', lw=0.5, zorder=4))
    ax.plot([5.5, 5.3, 5.6], [6.5, 6.8, 6.3], color='green', zorder=5)
    # Barbes de vent (simplificades)
    ax.plot([11.5, 11.5], [1.2, 7.8], color='gray', lw=0.5, zorder=3)
    for h in np.linspace(2, 7.5, 8):
        ax.plot([11.5, 11.8], [h, h+0.1], color='red', lw=1, zorder=4)

    # Títols
    fig.text(0.5, 0.85, "Benvingut al Visor de Sondejos", ha='center', fontsize=40, color='white', weight='bold', alpha=0.9)
    fig.text(0.5, 0.15, "Tempestes.cat", ha='center', fontsize=30, color='white', style='italic', alpha=0.8)
    
    plt.tight_layout()
    return fig

# =========================================================================
# === 5. FUNCIONS D'EXECUCIÓ DELS MODES DE L'APP ==========================
# =========================================================================

def show_welcome_screen():
    # Utilitza la nova funció de dibuix personalitzada
    welcome_fig = create_welcome_graphic()
    st.pyplot(welcome_fig, use_container_width=True)
    
    st.subheader("Tria un mode per començar")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛰️ Mode en Viu")
        st.info("Visualitza els sondejos atmosfèrics basats en dades reals del dia. Navega entre les diferents hores disponibles.")
        if st.button("Accedir al Mode en Viu", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("### 🧪 Laboratori de Sondejos")
        st.info("Experimenta amb un sondeig base o carrega escenaris extrems per entendre com afecten els paràmetres al temps.")
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
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
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings:
            st.error("Error crític: No s'ha trobat 'sondeigproves.txt'. Aquest mode no pot funcionar.")
            return
        data = soundings[0]
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
        preset_files = {
            'Perfil Base': "sondeigproves.txt", 'Neu a BCN': "snow_bcn.txt", 'Supercèl·lula (Oklahoma)': "oklahoma.txt", 
            'Ambient Tropical (Malàisia)': "malaysia.txt", 'Desert (Sahara)': "sahara.txt", 
            'Medicane': "medicaine.txt", 'Cicló Tropical': "cyclone.txt",
            'Antàrtida': "antarctica.txt", 'Everest': "everest.txt"
        }
        selected_preset = st.selectbox("Tria un escenari:", list(preset_files.keys()))
        
        if st.button("Carregar Escenari", use_container_width=True):
            filepath = preset_files[selected_preset]
            soundings = parse_all_soundings(filepath)
            if soundings:
                data = soundings[0]
                st.session_state.sandbox_p_levels = data['p_levels'].copy()
                st.session_state.sandbox_t_profile = data['t_initial'].copy()
                st.session_state.sandbox_td_profile = data['td_initial'].copy()
                st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
                st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
                st.rerun()
            else:
                st.error(f"No s'ha pogut carregar l'escenari '{selected_preset}' des de '{filepath}'.")

        st.markdown("---")
        st.subheader("Modificació Manual (en viu)")
        
        t_profile_mod = st.session_state.sandbox_t_profile.copy()
        td_profile_mod = st.session_state.sandbox_td_profile.copy()
        
        new_sfc_t = st.slider("🌡️ Temperatura en Superfície (°C)", -40.0, 50.0, t_profile_mod[0].m, 0.5, key="t_slider")
        new_sfc_td = st.slider("💧 Punt de Rosada en Superfície (°C)", -40.0, new_sfc_t, td_profile_mod[0].m, 0.5, key="td_slider")
        
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
        obs_time="Sondeig de Prova - Mode Laboratori"
    )

# =========================================================================
# === 7. PUNT D'ENTRADA DE L'APLICACIÓ ====================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Visor de Sondejos", page_icon="⛈️")
    
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
