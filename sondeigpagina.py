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
import time
import threading
import base64
import io

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
integrator_lock = threading.Lock()


# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================
def parse_all_soundings(filepath):
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'.")
        return []

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
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
# =========================================================================

def calculate_low_level_moisture(p_levels, td_profile):
    try:
        p_clean, td_clean = p_levels.to('hPa'), td_profile.to('degC')
        heights_amsl = mpcalc.pressure_to_height_std(p_clean).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) < 2: return units.Quantity(0, 'mm')
        return mpcalc.precipitable_water(p_clean[layer_mask], td_clean[layer_mask]).to('mm')
    except Exception: return units.Quantity(0, 'mm')

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
            p_range, t_range = np.arange(p.m.min(), p.m.max()), t_interp(p_range)
            fz_idx = np.where(t_range < 0)[0]
            fz_lvl = p_range[fz_idx[0]] * units.hPa if fz_idx.size > 0 else np.nan * units.hPa
        except Exception: fz_lvl = np.nan * units.hPa
        if el_p is None and cape.magnitude > 0: el_p = p[-1] 
        lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p else 0
        lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.inf
        el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else lfc_h
        fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
        return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
    except Exception: return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0)

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
        u_i, v_i = np.interp(h_interp.m, h_u.m, u_u.m) * units('m/s'), np.interp(h_interp.m, h_u.m, v_u.m) * units('m/s')
        u_6, v_6 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        u_1, v_1 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
        return s_0_6, s_0_1, srh_0_1, srh_0_3
    except Exception: return 0.0, 0.0, 0.0, 0.0

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, pwat_0_4):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    
    precipitation_type = None
    if fz_h < 1500 or t_profile[0].m < 5:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'sleet'
    elif cape.m > 3000: precipitation_type = 'hail'
    elif cape.m > 500: precipitation_type = 'rain'
    elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'
    elif lfc_p and el_p and el_p < lfc_p: precipitation_type = 'virga'

    full_conversation = []
    initial_summary = f"Hola! Mira el que veig: una situació compatible amb la formació de núvols de tipus **{cloud_type}**."
    full_conversation.append(("Tempestes.cat", initial_summary))

    if cloud_type == "Hivernal":
        full_conversation.extend([
            ("Yo", f"La isoterma 0°C és a {fz_h:.0f}m. Això és molt baix, no?"),
            ("Tempestes.cat", "Exacte. Això, combinat amb la humitat en nivells baixos, és el factor clau per a la neu."),
            ("Yo", f"I amb una temperatura en superfície de {t_profile[0].m:.1f}°C, què podem esperar?"),
        ])
        if t_profile[0].m <= 0.5:
            full_conversation.append(("Tempestes.cat", "Amb temperatures sota zero o properes a 0°C a tots els nivells, la precipitació serà neu fins a cotes molt baixes. Prepara el trineu!"))
        else:
            full_conversation.append(("Tempestes.cat", "Compte! Hi ha una petita capa càlida just sobre la superfície. Això pot provocar pluja gelant, un fenomen molt perillós a les carreteres."))
            
    elif cloud_type == "Supercèl·lula":
        full_conversation.extend([
            ("Yo", f"El CAPE és de {cape.m:.0f} J/kg. Què significa un valor tan alt?"),
            ("Tempestes.cat", "És l'energia de la tempesta. Un valor tan alt indica corrents ascendents extremadament violents, capaços de sostenir calamarsa de gran mida."),
            ("Yo", "I el cisallament del vent? Veig valors elevats."),
            ("Tempestes.cat", f"Correcte. El cisallament de {shear_0_6:.0f} m/s i l'helicitat (SRH) de {srh_0_3:.0f} m²/s² són els ingredients que faran que la tempesta giri, formant una supercèl·lula."),
            ("Yo", "Quin és el risc principal, doncs?"),
            ("Tempestes.cat", f"Molt alt. Cal esperar calamarsa de mida grossa (>4cm), ratxes de vent destructives i, amb un SRH 0-1km de {srh_0_1:.1f}, hi ha un risc significatiu de tornados.")
        ])
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
         full_conversation.extend([
            ("Yo", f"El CAPE és de {cape.m:.0f} J/kg. És molt?"),
            ("Tempestes.cat", "És un valor moderat-alt. Hi ha energia suficient per a tempestes fortes, però no explosives."),
            ("Yo", "Per què no és una supercèl·lula, llavors?"),
            ("Tempestes.cat", f"El cisallament ({shear_0_6:.0f} m/s) és massa feble. Les tempestes competiran entre elles en lloc de formar una única estructura organitzada. Si són Castellanus, la convecció s'inicia des de més amunt."),
            ("Yo", "Quins fenòmens hem d'esperar?"),
            ("Tempestes.cat", "Principalment xàfecs intensos i calamarsa de mida petita a moderada. En el cas dels Castellanus, el risc principal són els esclafits secs (downbursts).")
        ])
    elif "Nimbostratus" in cloud_type:
        full_conversation.extend([
            ("Yo", "Veig molta humitat a capes baixes però gairebé gens d'inestabilitat (CAPE). Què passa?"),
            ("Tempestes.cat", f"Exacte. No hi ha motor convectiu (CAPE de {cape.m:.0f} J/kg), però l'atmosfera està saturada. Això és típic de la pluja estratiforme, associada a fronts."),
            ("Yo", "Llavors, com serà la pluja? Depèn del PWAT, oi?"),
        ])
        if "Intens" in cloud_type:
             full_conversation.append(("Tempestes.cat", f"Sí. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**, un valor molt alt. Això es traduirà en pluges contínues i abundants. Risc d'acumulacions importants."))
        elif "Moderat" in cloud_type:
             full_conversation.append(("Tempestes.cat", f"Correcte. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. Un valor considerable que alimentarà xàfecs moderats i persistents, el que anomenem 'petacs' de pluja."))
        else: # Fluix
             full_conversation.append(("Tempestes.cat", f"Exactament. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. Suficient per a ruixats febles i intermitents, però no s'esperen grans quantitats."))
    else:
        full_conversation.extend([
            ("Yo", "Sembla un dia tranquil, oi?"),
            ("Tempestes.cat", f"Sí. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable. No hi ha risc de temps sever."),
            ("Yo", "Veurem algun núvol?"),
            ("Tempestes.cat", f"Probablement només alguns {cloud_type} sense desenvolupament vertical ni risc de precipitació. Un dia perfecte per passejar!")
        ])
        
    return full_conversation, precipitation_type

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
            if np.mean(rh_layer) > 0.85 and cape.magnitude < 350:
                if pwat_layer.m > 25: return "AVÍS PER PLUGES INTENSES", "Risc de pluges persistents i fortes. Possible acumulació d'aigua.", "darkblue"
                elif pwat_layer.m > 15: return "AVÍS PER PLUJA MODERADA", "Cel cobert amb pluja contínua i moderada. Visibilitat reduïda.", "steelblue"
                else: return "PREVISIÓ DE PLUJA FEBLE", "S'esperen plugims o ruixats febles i intermitents.", "cadetblue"
    except Exception: pass
    if cape.m >= 1000:
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        if srh_0_1 > 150 and shear_0_6 > 15: return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        if lfc_h > 3000: return "AVÍS PER TEMPESTES DE BASE ALTA", "Nuclis de base alta. Risc de ratxes de vent fortes i sobtades (downbursts).", "darkorange"
        if cape.m > 2000: return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa. Protegiu vehicles.", "purple"
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"

# =========================================================================
# === 3. FUNCIONS DE DIBUIX ===============================================
# =========================================================================

def create_logo_figure():
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')
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

def _draw_cumulonimbus(ax, base_km, top_km):
    updraft_center_x, num_points, patches = 0, 20, []
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    if len(tower_indices) == 0: tower_indices = np.arange(len(altitudes))
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    widths += np.random.uniform(-0.05, 0.05, len(tower_indices))
    r_pts = [(updraft_center_x + w, alt) for w, alt in zip(widths, tower_alts)]
    l_pts = [(updraft_center_x - w, alt) for w, alt in zip(widths, tower_alts)]
    ax.add_patch(Polygon([l_pts[0]] + r_pts + l_pts[::-1], facecolor='#d8d8d8', lw=0, zorder=10))
    for _ in range(120):
        idx = random.randint(1, len(tower_alts) - 1)
        y = tower_alts[idx] + random.uniform(-0.3, 0.3)
        max_x = np.interp(y, tower_alts, widths, left=widths[0], right=widths[-1])
        x = updraft_center_x + random.uniform(-max_x, max_x)
        size = random.uniform(0.2, 0.6) * (1 + (y - base_km) / (top_km - base_km))
        brightness = np.clip(0.85 + 0.15 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
        patches.append(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.1, 0.35), lw=0))
    for _ in range(80):
        y = random.uniform(anvil_base_alt, top_km)
        h_factor = 1 + (y - anvil_base_alt) / (top_km - anvil_base_alt)
        x = updraft_center_x + random.uniform(-1.5 * h_factor, 1.5 * h_factor)
        w, h = random.uniform(0.5, 1.2) * h_factor, random.uniform(0.05, 0.15)
        patches.append(Ellipse((x, y), w, h, facecolor=tuple([random.uniform(0.95, 1.0)]*3), alpha=random.uniform(0.1, 0.3), lw=0))
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=11))

def _draw_cumulus_mediocris(ax, base_km, top_km):
    center_x, num_particles, cloud_height = 0, 250, top_km - base_km
    altitudes = np.linspace(base_km, top_km, 20)
    base_width = 0.4 * (1 + 0.8 * np.sin(np.pi * (altitudes - base_km) / (cloud_height + 0.01)))
    widths = base_width + np.random.uniform(-0.1, 0.1, len(altitudes))
    widths[0] = max(widths[0], 0.3)
    r_pts, l_pts = [(center_x + w, alt) for w, alt in zip(widths, altitudes)], [(center_x - w, alt) for w, alt in zip(widths, altitudes)]
    ax.add_patch(Polygon([l_pts[0]] + r_pts + l_pts[::-1], facecolor='#d0d0d0', lw=0, zorder=10))
    patches = []
    for _ in range(num_particles):
        y_progress = random.betavariate(2, 2)
        y = base_km + y_progress * cloud_height
        max_x = np.interp(y, altitudes, widths)
        x = center_x + random.uniform(-max_x, max_x) * 0.95
        size = random.uniform(0.15, 0.5) * (1 + y_progress * 0.5)
        brightness = 0.8 + 0.2 * (y_progress ** 0.7)
        patches.append(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.15, 0.45), lw=0))
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=11))

def _draw_cumulus_castellanus(ax, base_km, top_km):
    base_thickness = min(0.8, (top_km - base_km) * 0.25)
    patches_base, num_turrets = [], random.randint(3, 5)
    for _ in range(120):
        x, y = random.uniform(-1.7, 1.7), base_km + (random.random() ** 2) * base_thickness
        b = random.uniform(0.8, 0.9)
        patches_base.append(Ellipse((x, y), width=random.uniform(0.7, 1.6), height=random.uniform(0.1, 0.25), facecolor=(b, b, b), alpha=random.uniform(0.1, 0.3), lw=0))
    ax.add_collection(PatchCollection(patches_base, match_original=True, zorder=8))
    for i in range(num_turrets):
        turret_center_x, turret_base_y = random.uniform(-1.3, 1.3), base_km + base_thickness * 0.5
        turret_top_y = turret_base_y + random.uniform(0.5, 0.95) * (top_km - turret_base_y)
        turret_height, max_width, patches_turret = turret_top_y - turret_base_y, random.uniform(0.25, 0.4), []
        for _ in range(random.randint(60, 90)):
            y = turret_base_y + (random.random() ** 0.8) * turret_height
            norm_y = (y - turret_base_y) / turret_height
            current_width = max_width * np.sin(np.pi * norm_y)
            x = turret_center_x + random.uniform(-current_width * 0.9, current_width * 0.9)
            size = random.uniform(0.1, 0.3) * (1 + norm_y * 0.5)
            brightness = 0.75 + 0.23 * (norm_y ** 0.8)
            patches_turret.append(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.2, 0.5), lw=0))
        ax.add_collection(PatchCollection(patches_turret, match_original=True, zorder=9 + i))

def _draw_nimbostratus(ax, base_km, top_km, cloud_type):
    if "Intens" in cloud_type: color, alpha = '#808080', 0.95
    elif "Moderat" in cloud_type: color, alpha = '#a9a9a9', 0.9
    else: color, alpha = '#c0c0c0', 0.85
    ax.add_patch(Rectangle((-1.7, base_km), 3.4, top_km - base_km, facecolor=color, lw=0, zorder=8, alpha=alpha))
    patches = [Ellipse((random.uniform(-1.7, 1.7), random.uniform(base_km, top_km)), width=random.uniform(0.8, 1.5), height=random.uniform(0.1, 0.3), facecolor=(b,b,b), alpha=random.uniform(0.2, 0.4), lw=0) for _ in range(150) for b in [random.uniform(0.6, 0.75)]]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))

def _draw_cumulus_fractus(ax, base_km, thickness):
    patches=[Ellipse((random.gauss(0,0.5),random.uniform(base_km,base_km+thickness)), random.uniform(0.2,0.4), random.uniform(0.3,0.7)*random.uniform(0.2,0.4), angle=random.uniform(-25,25), facecolor=tuple([c]*3), alpha=0.5,lw=0) for _ in range(150) for c in [np.clip(0.6 + 0.25*((random.uniform(base_km,base_km+thickness)-base_km)/(thickness+0.01))**0.7,0,1)]]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))

def _draw_clear_sky(ax): ax.add_collection(PatchCollection([Ellipse((random.uniform(-1.5,1.5), random.uniform(10,14)), random.uniform(0.5,1.0), random.uniform(0.1,0.2), facecolor='white', alpha=random.uniform(0.05,0.1), lw=0) for _ in range(15)], match_original=True, zorder=5))

def _draw_precipitation(ax, precip_base_km, ground_km, p_type, center_x=0.0, sub_cloud_rh=0.4):
    if p_type == 'virga':
        end_y = precip_base_km - (precip_base_km - ground_km) * (sub_cloud_rh / 0.5)
        end_y = max(end_y, ground_km + 0.3) if sub_cloud_rh < 0.5 else ground_km
        top_w, bot_w = random.uniform(0.6, 0.9), top_w * 0.5
        ax.add_patch(Polygon([(center_x-top_w/2, precip_base_km), (center_x+top_w/2, precip_base_km), (center_x+bot_w/2, end_y), (center_x-bot_w/2, end_y)], facecolor='cornflowerblue', alpha=np.clip(sub_cloud_rh*0.6,0.15,0.55), lw=0, zorder=7))
    elif p_type in ['rain', 'sleet']: ax.add_patch(Rectangle((center_x-0.8, ground_km), 1.6, precip_base_km - ground_km, facecolor='cornflowerblue', alpha=0.35, lw=0, zorder=5))
    elif p_type == 'hail': ax.scatter(center_x+np.random.normal(0,0.3,150),np.random.uniform(ground_km,precip_base_km,150), s=np.random.uniform(5,40,150),c='white',alpha=0.8,marker='o',edgecolor='gray',linewidth=0.5,zorder=8)
    elif p_type == 'snow': ax.scatter(center_x+np.random.normal(0,0.5,300),np.random.uniform(ground_km,precip_base_km,300), s=np.random.uniform(20,70,300),c='white',alpha=np.random.uniform(0.4,0.9,300),marker='*',zorder=8)

def _draw_saturation_layers(ax, p_levels, t_profile, td_profile):
    try:
        saturated_indices = np.where(t_profile.m - td_profile.m <= 1.5)[0]
        if not len(saturated_indices): return
        i=0
        while i < len(saturated_indices):
            start_idx, j = saturated_indices[i], i
            while j+1 < len(saturated_indices) and saturated_indices[j+1] == saturated_indices[j]+1: j+=1
            end_idx = saturated_indices[j]
            h_bottom = mpcalc.pressure_to_height_std(p_levels[start_idx]).to('km').m
            h_top = mpcalc.pressure_to_height_std(p_levels[end_idx]).to('km').m
            if h_top - h_bottom >= 0.05:
                patches = [Ellipse((random.uniform(-1.5,1.5), y), random.uniform(0.3,0.8), random.uniform(0.05,0.1)*(1+h_top-h_bottom), facecolor=(b,)*3, alpha=random.uniform(0.1,0.5), lw=0) for _ in range(int(100+300*(h_top-h_bottom))) for y,b in [(random.uniform(h_bottom,h_top), random.uniform(0.65,0.85))]]
                ax.add_collection(PatchCollection(patches, match_original=True, zorder=7))
            i = j+1
    except Exception: pass

def _calculate_dynamic_cloud_heights(p, t, td, conv_active):
    _, _, lcl_p, lcl_h, _, _, _, el_h, _ = calculate_thermo_parameters(p, t, td)
    if not lcl_p: return None, None
    base_km = lcl_h / 1000.0
    if conv_active: top_km = el_h / 1000.0 if el_h > lcl_h else base_km
    else:
        try:
            rh = mpcalc.relative_humidity_from_dewpoint(t, td)
            indices = np.where(p <= lcl_p)[0]
            p_top = p[-1]
            if len(indices) > 0:
                for idx in indices:
                    if rh[idx] < 0.5: p_top = p[idx]; break
            top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
        except: top_km = base_km
    return (base_km, top_km) if top_km > base_km else (None, None)

def _draw_base_feature(ax, f_type, base_x_left, base_x_right, base_y, ground_y):
    z, center_x, width = 12, (base_x_left + base_x_right) / 2, base_x_right - base_x_left
    if f_type == 'lowering': ax.add_patch(Polygon([(base_x_left, base_y), (base_x_right, base_y), (base_x_right*0.9+center_x*0.1, base_y-0.2), (base_x_left*0.9+center_x*0.1, base_y-0.2)], fc='dimgray', ec='gray', zorder=z))
    elif f_type == 'wall_cloud': ax.add_patch(Polygon([(center_x-width*0.375, base_y), (center_x+width*0.375, base_y), (center_x+width*0.275, base_y-0.35), (center_x-width*0.275, base_y-0.35)], fc='#383838', ec='#202020', lw=0.5, zorder=z))
    elif f_type == 'funnel': ax.add_patch(Polygon([(center_x-0.2, base_y), (center_x+0.2, base_y), (center_x, max(base_y-0.8, ground_y+0.5))], fc='darkgray', alpha=0.8, zorder=z))
    elif f_type == 'tornado':
        ax.add_patch(Polygon([(center_x-0.2, base_y), (center_x+0.2, base_y), (center_x, ground_y)], fc='#505050', zorder=z))
        ax.add_patch(Ellipse((center_x, ground_y+0.05), width=0.7, height=0.25, fc='#654321', alpha=0.7, zorder=z+1))

def create_skewt_figure(p, t, td, ws, wd):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100); ax.set_xlim(-50, 45)
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange'); skew.plot_moist_adiabats(alpha=0.3, color='green')
    skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    skew.plot(p, t, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p, np.minimum(t, td), 'b', linewidth=2, label='Punt de Rosada (Td)')
    parcel_prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')
    skew.plot(p, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    skew.plot(p, mpcalc.wet_bulb_temperature(p, t, td), color='purple', linewidth=1.5, label='Tª Bombolla Humida')
    skew.shade_cape(p, t, parcel_prof, fc='yellow', alpha=0.3); skew.shade_cin(p, t, parcel_prof, fc='black', alpha=0.3)
    cape, cin, lcl_p, _, lfc_p, _, el_p, _, _ = calculate_thermo_parameters(p, t, td)
    xlims = ax.get_xlim()
    if lcl_p: ax.plot(xlims, [lcl_p.m]*2, 'gray', ls='--', label='LCL')
    if lfc_p: ax.plot(xlims, [lfc_p.m]*2, 'purple', ls='--', label='LFC')
    if el_p: ax.plot(xlims, [el_p.m]*2, 'red', ls='--', label='EL')
    ax.legend(); plt.tight_layout()
    return fig

def create_cloud_drawing_figure(p, t, td, conv_active, precip_type, lfc_h, cape, base_km, top_km, cloud_type):
    fig, ax = plt.subplots(figsize=(5, 8))
    ground_km = mpcalc.pressure_to_height_std(p[0]).to('km').m
    ax.set(ylim=(0,16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0,17,2), ylabel="Altitud (km)", title="Visualització del Núvol")
    ax.grid(True, ls='dashdot', alpha=0.5); ax.set_facecolor('#6495ED')
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    ground_color = 'white' if precip_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_km, color=ground_color, alpha=0.8, zorder=3, hatch='//' if ground_color=='#228B22' else ''))
    _draw_saturation_layers(ax, p, t, td)
    
    if "Nimbostratus" in cloud_type: _draw_nimbostratus(ax, 0.5, 4.0, cloud_type)
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Supercèl·lula"]: _draw_cumulonimbus(ax, max(base_km, ground_km+0.5), top_km)
    elif cloud_type == "Castellanus": _draw_cumulus_castellanus(ax, max(lfc_h/1000.0, ground_km+0.5), top_km)
    elif cloud_type == "Cumulus Mediocris": _draw_cumulus_mediocris(ax, max(base_km, ground_km+0.5), top_km)
    elif cloud_type == "Cumulus Fractus": _draw_cumulus_fractus(ax, max(base_km, ground_km+0.5), top_km-base_km)
    elif not np.any((t.m-td.m)<=1.5): _draw_clear_sky(ax)
    
    if precip_type:
        is_castellanus = (cloud_type == "Castellanus")
        precip_base_km = lfc_h/1000.0 if is_castellanus and lfc_h else base_km
        sub_cloud_rh_mean = 0.4
        try:
            p_base_precip, p_ground = mpcalc.height_to_pressure_std(precip_base_km*units.km), p[0]
            mask = (p >= p_base_precip) & (p <= p_ground)
            if np.any(mask): sub_cloud_rh_mean = np.mean(mpcalc.relative_humidity_from_dewpoint(t[mask], td[mask])).m
        except Exception: pass
        _draw_precipitation(ax, precip_base_km, ground_km, precip_type, sub_cloud_rh=sub_cloud_rh_mean)
        
    plt.tight_layout()
    return fig

def create_cloud_structure_figure(p, t, td, ws, wd, conv_active):
    fig, gs = plt.figure(figsize=(5, 8)), plt.GridSpec(1, 2, width_ratios=(4,1), wspace=0)
    ax, ax_shear = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1], sharey=ax)
    ground_km = mpcalc.pressure_to_height_std(p[0]).to('km').m
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10); ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5,0), 3, ground_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0,20), xlim=(-1.5,1.5), ylabel="Altitud (km)", xticks=[]); ax.grid(True, ls='--', alpha=0.3)
    ax_shear.set(xlim=(-1,1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_shear.spines['left'].set_visible(False); ax_shear.patch.set_alpha(0.0)
    cape, *_ = calculate_thermo_parameters(p, t, td)
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, conv_active)
    if not base_km or not top_km or cape.m < 100:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='white', bbox=dict(fc='darkblue', alpha=0.7))
        ax_shear.axis('off'); return fig
    visual_base_km = max(base_km, ground_km + 0.5)
    try:
        u, v = mpcalc.wind_components(ws, wd)
        h_km = mpcalc.pressure_to_height_std(p).to('km').m
        unique_h, idx = np.unique(h_km, return_index=True)
        if len(unique_h) < 2: return fig
        f_u, f_v = interp1d(unique_h, u.m[idx], 'extrapolate'), interp1d(unique_h, v.m[idx], 'extrapolate')
        barb_h = np.arange(0, min(20, h_km.max()), 1)
        ax_shear.barbs(np.zeros_like(barb_h), barb_h, (f_u(barb_h)*units('m/s')).to('knots').m, (f_v(barb_h)*units('m/s')).to('knots').m, length=7, pivot='middle', color='k')
        alts, u_alts = np.linspace(visual_base_km, top_km, 50), f_u(alts)
        offsets = u_alts * 0.02
        s_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
        shear_factor = np.clip(s_0_6/35, 0.4, 2.5)
        widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (alts - visual_base_km)/(top_km - visual_base_km + 0.01))) * shear_factor
        anvil_ext = np.zeros_like(alts)
        if (top_km - visual_base_km) > 4.0:
            anvil_base = top_km * 0.8
            anvil_idx = np.where(alts >= anvil_base)[0]
            if len(anvil_idx) > 0:
                u_anvil_top = f_u(top_km)
                max_stretch = abs(u_anvil_top) * 0.06 * np.sign(u_anvil_top)
                growth = ((alts[anvil_idx]-anvil_base)/(top_km-anvil_base))**1.5
                anvil_ext[anvil_idx] = max_stretch * growth
        r_pts, l_pts = [(w+off+ext, alt) for w,off,ext,alt in zip(widths,offsets,anvil_ext,alts)], [(-w+off, alt) for w,off,alt in zip(widths,offsets,alts)]
        ax.add_patch(Polygon(r_pts + l_pts[::-1], fc='white', ec='lightgray', alpha=0.95, zorder=10))
        _, _, lcl_p, lcl_h, _, _, _, _, _ = calculate_thermo_parameters(p, t, td)
        feature = None
        if top_km-base_km>4 and cape.m>500:
            if srh_0_1>=150 and lcl_h<=1000 and s_0_6>15: feature='tornado'
            elif srh_0_1>100 and lcl_h<1200 and s_0_6>12: feature='funnel'
            elif srh_0_3>150 and s_0_6>18 and cape.m>1000: feature='wall_cloud'
            elif s_0_1>8 and lcl_h<1500: feature='lowering'
        if feature: _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_km)
    except Exception: pass
    plt.tight_layout()
    return fig

def create_radar_figure(p, t, td, ws, wd):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray'); ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.grid(True, ls=':', alpha=0.3, color='white')
    cape, *_ = calculate_thermo_parameters(p, t, td)
    try:
        heights_agl = (mpcalc.pressure_to_height_std(p).to('m')-mpcalc.pressure_to_height_std(p[0]).to('m')).to('km')
        mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(mask) > 2:
            rh_mean = np.mean(mpcalc.relative_humidity_from_dewpoint(t[mask], td[mask]))
            pwat = mpcalc.precipitable_water(p[mask], td[mask]).to('mm').m
            if rh_mean > 0.85 and cape.m < 350:
                x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
                max_dbz = np.clip(15+pwat, 15, 45)
                Z = np.clip(max_dbz + gaussian_filter(np.random.randn(100,100), sigma=8) * (max_dbz*0.2), 0, 50)
                colors, levels = ['#00a0f0','#0000ff','#00ff00','#008000','#ffff00','#ff9900'], [0,15,20,25,30,35,45]
                ax.contourf(x, y, Z, levels=levels, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)))
                return fig
    except Exception: pass
    if cape.m < 100:
        ax.text(0,0, "Sense precipitació significativa", ha='center', va='center', color='white', fontsize=9)
        return fig
    s_0_6, *_ = calculate_storm_parameters(p, ws, wd)
    _, _, _, _, lfc_p, _, el_p, _, _ = calculate_thermo_parameters(p, t, td)
    mean_u, mean_v = (0,0) * units('m/s')
    if lfc_p and el_p:
        p_mask = (p >= el_p) & (p <= lfc_p)
        if np.sum(p_mask) > 1: mean_u, mean_v = np.mean(mpcalc.wind_components(ws[p_mask], wd[p_mask]))
    max_dbz = np.clip(20 + (cape.m/3000)*55, 20, 75)
    elongation = np.clip(1 + (s_0_6/20), 1, 2.5)
    angle = np.arctan2(mean_u.m, mean_v.m)
    x, y = np.linspace(-50, 50, 150), np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    x_r, y_r = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
    Z = max_dbz * np.exp(-((x_r**2/(2*15**2)) + (y_r**2/(2*(15/elongation)**2))))
    Z = np.clip(Z + gaussian_filter(np.random.randn(150,150), sigma=6)*(max_dbz*0.1), 0, 75)
    colors = ['#00a0f0','#0000ff','#00ff00','#008000','#ffff00','#ff9900','#ff0000','#c80000','#ff00ff','#960096']
    levels = [0,15,20,25,30,35,40,45,50,55,75]
    ax.contourf(xx, yy, Z, levels=levels, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)))
    return fig

# =========================================================================
# === 5. LÒGICA DE L'APLICACIÓ STREAMLIT =================================
# =========================================================================

# --- Funcions de callback per a la navegació ---
def increment_index():
    if st.session_state.sounding_index < len(st.session_state.existing_files) - 1: st.session_state.sounding_index += 1
    st.session_state.chat_open = False # Resetejar el xat en canviar de sondeig
    st.session_state.chat_progress = 0

def decrement_index():
    if st.session_state.sounding_index > 0: st.session_state.sounding_index -= 1
    st.session_state.chat_open = False
    st.session_state.chat_progress = 0

def sync_index_from_selectbox():
    st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget)
    st.session_state.chat_open = False
    st.session_state.chat_progress = 0

def load_sounding_data_from_index():
    st.session_state.selected_file = st.session_state.existing_files[st.session_state.sounding_index]
    soundings = parse_all_soundings(st.session_state.selected_file)
    if not soundings:
        st.error(f"No s'han pogut carregar dades de {st.session_state.selected_file}")
        st.session_state.sounding_index = st.session_state.loaded_sounding_index
        return
    st.session_state.original_data = soundings[0]
    reset_working_profiles()
    st.session_state.loaded_sounding_index = st.session_state.sounding_index

def reset_working_profiles():
    data = st.session_state.original_data
    st.session_state.p_levels, st.session_state.t_profile, st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir, st.session_state.observation_time = data['p_levels'].copy(), data['t_initial'].copy(), data['td_initial'].copy(), data['wind_speed_kmh'].to('m/s'), data['wind_dir_deg'].copy(), data.get('observation_time', 'Hora no disponible')

def get_whatsapp_button_html(unread=False):
    # Icona de WhatsApp en Base64 per evitar dependències externes
    whatsapp_icon_b64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0OCIgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTcuOSAyYS45LjkgMCAwIDEgLjg5LjUgMS42IDEuNiAwIDAgMSAuMTUgMS4xMyA4LjcgOC43IDAgMCAxIC0yLjcgNS41MiA5LjQgOS40IDAgMCAxLTUuNDcgMi43QTEuNiAxLjYgMCAwIDEgOC44IDEzYy0uNDUtLjA3LS44LS4xMy0xLjE1LS4yMWwtMi40LS44NWMtLjI0LS4wOC0uNDQtLjE3LS42LS4yNmExIDEgMCAwIDEgLS42LTEuMDdsLjEtLjExYy4xMy0uMjIuMjktLjQzLjQ4LS42MmwzLjY4LTMuNjlhMS40IDEuNCAwIDAgMSAyIDBsMy42NyAzLjY4YTEuNCAxLjQgMCAwIDEgMCAyTDEyLjggMTNsLS4zNy4zNmEzIDEgMCAwIDEtLjkuNDQgNC41IDQuNSAwIDAgMCAuNDUgMS44NmMyLjE1IDEgNC4xMy40IDUuMy0xLjNhNCA0IDAgMCAwIC45LTMuMWwxLjQtMi4xMkEuOS45IDAgMCAxIDE3LjkgMnoiPjwvcGF0aD48L3N2Zz4="
    
    notification_html = """
    <span style="position: absolute; top: -5px; right: -5px; background-color: red; color: white; border-radius: 50%; padding: 2px 7px; font-size: 12px; font-weight: bold; border: 2px solid white;">1</span>
    """ if unread else ""

    return f"""
    <div style="position: relative; display: inline-block; cursor: pointer;">
        <div style="background-color: #25D366; border-radius: 50%; padding: 10px; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <img src="{whatsapp_icon_b64}" width="32" height="32">
        </div>
        {notification_html}
    </div>
    """

def display_whatsapp_chat(full_conversation, logo_b64):
    # CSS per al xat
    st.markdown(f"""
    <style>
        .chat-container {{ background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARMAAAARCAYAAAA/I2f7AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAZiS0dEAAAAAAAA+UO7fwAAAAlwSFlzAAAASAAAAEgARslrPgAAACVpVFh0ZGF0ZTpjcmVhdGUAMjAyMy0wNC0yNFQxMzo0MzozNCswMDowMEe0sVMAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjMtMDQtMjRUMTM6NDM6MzQrMDA6MDBCOfWfAAABFklEQVR42u3UsQmAYBRF0SkEryA4ipOTycnjCiIjn4h6aNTVtP2n/9AkwJ8M4y0DAAAAAAAAAAAAAAAAAADgLdZudnys5/bT68G15nZns2VzP69L1+sDAAAAAAAAAAAAAAAAAADwt+rW3M8L12s3AzaPNxvz+QLr1QMAAAAAAAAAAAAAAAAAAPD3urW33e3msfP1ms3e+VrN5urZAAAAAAAAAAAAAAAAAAD4u/rE2u3ms9dudv31fC3ndvP1qunPAQAAAAAAAAAAAAAAAAB479Y6vj0fH1vN6/f1Sz/Xn8/XrNcDAAAAAAAAAAAAAAAAAADwt+q9mdvP+/f1a7e7Aa/Z/Pq9/QIAAAC+h6jYdBS8LtuEAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDIxLTA5LTA5VDExOjA5OjU3KzAyOjAwGzsrQwAAACV0RVh0ZGF0ZTptb2RpZnkAMjIxLTA5LTA5VDExOjA5OjU3KzAyOjAw0/vJxwAAABl0RVh0U29mdHdhcmUAQWRvYmUgSW1hZ2VSZWFkeXHJZTwAAAAASUVORK5CYII='); background-color: #E5DDD5; padding: 15px; border-radius: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; max-height: 500px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; border: 1px solid #ccc; }}
        .message-row {{ display: flex; align-items: flex-end; gap: 10px; max-width: 85%; }}
        .message-row-right {{ justify-content: flex-end; margin-left: auto; }}
        .message {{ padding: 6px 12px 8px; border-radius: 8px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }}
        .yo {{ background-color: #DCF8C6; }}
        .tempestes-cat {{ background-color: #FFFFFF; }}
        .message .time {{ font-size: 0.7em; color: #999; float: right; margin-left: 10px; margin-top: 5px; }}
        .message .ticks {{ font-size: 0.8em; color: #999; }}
        .message.sent .ticks {{ color: #4FC3F7; }}
        .profile-pic {{ width: 40px; height: 40px; border-radius: 50%; object-fit: cover; }}
        .typing-indicator {{ display: flex; align-items: center; background: #fff; border-radius: 8px; padding: 8px 12px; margin-right: auto; box-shadow: 0 1px 1px rgba(0,0,0,0.1);}}
        .typing-indicator span {{ height: 8px; width: 8px; background-color: #ccc; border-radius: 50%; display: inline-block; margin: 0 2px; animation: bounce 1.3s infinite; }}
        .typing-indicator span:nth-of-type(2) {{ animation-delay: 0.15s; }}
        .typing-indicator span:nth-of-type(3) {{ animation-delay: 0.3s; }}
        @keyframes bounce {{ 0%, 80%, 100% {{ transform: scale(0); }} 40% {{ transform: scale(1.0); }} }}
    </style>
    """, unsafe_allow_html=True)

    # --- Lògica d'estat per al xat ---
    if 'chat_progress' not in st.session_state: st.session_state.chat_progress = 0
    if 'is_typing' not in st.session_state: st.session_state.is_typing = False
    
    def handle_send_message():
        st.session_state.chat_progress += 1 # Revela el missatge de l'usuari
        st.session_state.is_typing = True

    # --- Dibuixa el xat ---
    with st.container():
        st.markdown(f"**Tempestes.cat** - _en línia_")
        chat_placeholder = st.container()
        
        if st.session_state.is_typing:
            time.sleep(1.5) # Simula el temps de resposta
            st.session_state.is_typing = False
            st.session_state.chat_progress += 1 # Revela la resposta del bot
            st.rerun()

        with chat_placeholder:
            with st.container(border=True): # Simula la finestra del xat
                visible_log = full_conversation[:st.session_state.chat_progress + 1]
                
                with st.container(height=350):
                    for i, (speaker, message) in enumerate(visible_log):
                        is_user = (speaker == "Yo")
                        align_class = "message-row-right" if is_user else ""
                        msg_class = "yo" if is_user else "tempestes-cat"
                        
                        # El missatge de l'usuari es marca com a "llegit" si ja hi ha una resposta
                        is_sent = is_user and i < st.session_state.chat_progress
                        sent_class = "sent" if is_sent else ""

                        if is_user:
                            st.markdown(f"""
                            <div class="message-row {align_class}">
                                <div class="message {msg_class} {sent_class}">
                                    {message}
                                    <span class="time">12:34 PM <span class="ticks">✔✔</span></span>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="message-row {align_class}">
                                <img src="data:image/png;base64,{logo_b64}" class="profile-pic">
                                <div class="message {msg_class}">
                                    {message}
                                    <span class="time">12:34 PM</span>
                                </div>
                            </div>""", unsafe_allow_html=True)
                    
                    if st.session_state.is_typing:
                        st.markdown("""
                        <div class="message-row">
                            <img src="data:image/png;base64,{logo_b64}" class="profile-pic">
                            <div class="typing-indicator"><span></span><span></span><span></span></div>
                        </div>""", unsafe_allow_html=True)
                
                st.divider()

                # --- Àrea d'entrada de l'usuari ---
                next_user_message_index = st.session_state.chat_progress + 1
                if next_user_message_index < len(full_conversation):
                    next_message_text = full_conversation[next_user_message_index][1]
                    cols = st.columns([8,2])
                    with cols[0]:
                        st.text_input("Missatge:", value=next_message_text, disabled=True, label_visibility="collapsed")
                    with cols[1]:
                        st.button("Enviar", on_click=handle_send_message, use_container_width=True, type="primary")
                else:
                    st.success("Conversa finalitzada.")

def main():
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")

    if 'initialized' not in st.session_state:
        base_files = [f"{h}{p}.txt" for h in range(1, 13) for p in ['am', 'pm']]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files: st.stop()
        st.session_state.sounding_index, st.session_state.loaded_sounding_index = 0, -1
        st.session_state.convergence_active, st.session_state.initialized = True, True
        st.session_state.chat_open = False
        st.session_state.chat_progress = 0

    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        load_sounding_data_from_index()

    logo_fig = create_logo_figure()
    logo_buffer = io.BytesIO()
    logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    logo_b64 = base64.b64encode(logo_buffer.getvalue()).decode()
    
    with st.sidebar:
        st.image(logo_buffer)
        st.title("Controls")
        st.selectbox("Selecciona una hora:", options=st.session_state.existing_files, index=st.session_state.sounding_index, key='selectbox_widget', on_change=sync_index_from_selectbox)
        st.toggle("Activar convergència", value=st.session_state.convergence_active, key='convergence_active')
        if st.button("🔄 Reiniciar Perfils"): reset_working_profiles(); st.success("Perfils reiniciats.")
        with st.expander("🔬 Modificació Avançada"):
            sfc_temp_val = st.session_state.t_profile[0].m
            new_sfc_temp = st.slider("Tª Superfície (°C)", sfc_temp_val-20, sfc_temp_val+20, sfc_temp_val, 0.5)
            if new_sfc_temp != sfc_temp_val: st.session_state.t_profile[0] = new_sfc_temp * units.degC

    st.title("Visor de Sondejos Atmosfèrics")
    time_parts = st.session_state.observation_time.split('\n')
    cleaned_time_str = next((p.strip() for p in time_parts if 'local' in p.lower()), time_parts[0].strip() if time_parts else "")
    st.markdown(f"#### {cleaned_time_str}")

    p, t, td, ws, wd = st.session_state.p_levels, st.session_state.t_profile, st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f'<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>', unsafe_allow_html=True)

    sub_cols = st.columns([2, 8, 2])
    with sub_cols[0]: st.button('← Anterior', on_click=decrement_index, disabled=(st.session_state.sounding_index==0), use_container_width=True)
    with sub_cols[1]: st.subheader("Diagrama Skew-T", anchor=False)
    with sub_cols[2]: st.button('Següent →', on_click=increment_index, disabled=(st.session_state.sounding_index>=len(st.session_state.existing_files)-1), use_container_width=True)
    
    st.pyplot(create_skewt_figure(p, t, td, ws, wd), use_container_width=True)
    st.divider()

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, st.session_state.convergence_active)
    
    cloud_type, pwat_0_4, rh_0_4 = "Cel Serè", units.Quantity(0, 'mm'), 0.0
    try:
        heights_agl = (mpcalc.pressure_to_height_std(p).to('m') - mpcalc.pressure_to_height_std(p[0]).to('m')).to('km')
        mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(mask) > 2:
            rh_0_4 = np.mean(mpcalc.relative_humidity_from_dewpoint(t[mask], td[mask]))
            pwat_0_4 = mpcalc.precipitable_water(p[mask], td[mask]).to('mm')
    except Exception: pass
    
    sfc_temp = t[0].m
    if sfc_temp < 5 or fz_h < 1500: cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape.m < 350:
        if pwat_0_4.m > 25: cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4.m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150: cloud_type = "Supercèl·lula"
    elif cape.m > 500: cloud_type = "Cumulonimbus (Multicèl·lula)" if lfc_h < 3000 else "Castellanus"
    elif base_km and top_km:
        thickness = top_km - base_km
        if thickness > 2.0 and lfc_h < 3000: cloud_type = "Cumulus Mediocris"
        elif thickness > 0: cloud_type = "Cumulus Fractus"

    full_conversation, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, pwat_0_4)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])

    with tab1:
        if 'chat_open' not in st.session_state: st.session_state.chat_open = False
        
        if not st.session_state.chat_open:
            cols = st.columns([1,2,1])
            with cols[1]:
                st.markdown('<div style="text-align: center;">Fes clic per obrir el xat d\'anàlisi:</div>', unsafe_allow_html=True)
                if st.button("Obrir Anàlisi de Tempestes.cat", key="open_chat_btn", help="Fes clic per veure la conversa"):
                    st.session_state.chat_open = True
                    st.rerun()
                # Aquesta part és per si es vol usar la icona, però el botó és més clar
                # st.markdown(f'<div style="text-align: center;">{get_whatsapp_button_html(unread=True)}</div>', unsafe_allow_html=True)

        else:
            display_whatsapp_chat(full_conversation, logo_b64)

    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        cols = st.columns(4)
        cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A"); cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); cols[3].metric("Shear 0-6", f"{shear_0_6:.1f} m/s")
        cols[0].metric("SRH 0-1", f"{srh_0_1:.1f} m²/s²"); cols[1].metric("SRH 0-3", f"{srh_0_3:.1f} m²/s²")
        cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm"); cols[3].metric("RH Mitja 0-4km", f"{rh_0_4*100:.0f}%")

    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(create_cloud_drawing_figure(p, t, td, st.session_state.convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type), use_container_width=True)
        with cols[1]:
            st.pyplot(create_cloud_structure_figure(p, t, td, ws, wd, st.session_state.convergence_active), use_container_width=True)
            
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

if __name__ == '__main__':
    main()
