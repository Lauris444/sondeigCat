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
from datetime import datetime
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
        translated_line = re.sub(r'\(.*?\)|locale', '', line, flags=re.IGNORECASE).strip()
        for fr, ca in days_fr_to_ca.items(): translated_line = translated_line.replace(fr, ca)
        for fr, ca in months_fr_to_ca.items(): translated_line = translated_line.replace(fr, ca)
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
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
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
    chat_log = [("Tempestes.cat", f"Hola! He analitzat el sondeig i detecto una situació compatible amb la formació de núvols de tipus {cloud_type}.")]
    if cloud_type == "Hivernal":
        chat_log.extend([("Jo", f"La isoterma de 0°C està molt baixa, a uns {fz_h:.0f} metres."),("Tempestes.cat", "Exacte, aquest és el factor clau. Combinat amb la humitat present, afavoreix precipitacions hivernals."),("Jo", f"La temperatura a la superfície és de {t_profile[0].m:.1f}°C. Què podem esperar?"),])
        if t_profile[0].m <= 0.5:
            chat_log.append(("Tempestes.cat", "Amb temperatures negatives o properes a zero a tots els nivells, la precipitació serà de neu fins a les cotes més baixes."))
        else:
            chat_log.append(("Tempestes.cat", "Atenció. Hi ha una petita capa càlida just sobre la superfície. La neu podria fondre's en travessar-la i tornar-se a congelar en contacte amb el terra (pluja gelant) o arribar com aguanieve. És un fenomen perillós."))
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([("Jo", f"Veig uns valors d'inestabilitat i cisallament molt alts."),("Tempestes.cat", f"Correcte. Tenim un CAPE de {cape.m:.0f} J/kg, que és el combustible de la tempesta. A més, el cisallament de {shear_0_6:.1f} m/s i sobretot l'helicitat (SRH) de {srh_0_1:.1f} m²/s² a nivells baixos són ideals per a la rotació."), ("Jo", f"I el CIN de {cin.m:.0f} J/kg? Actua com a fre?"), ("Tempestes.cat", "Exactament. Aquest CIN actua com una 'tapadera' que impedeix que es formin tempestes dèbils. Si la convecció aconsegueix trencar aquesta tapadora, el desenvolupament pot ser explosiu, donant lloc a la supercèl·lula."), ("Jo", "Quin és el risc principal en aquest cas?"), ("Tempestes.cat", "El risc és molt alt. Cal esperar calamarsa gran o molt gran, ratxes de vent destructives i, amb aquests valors d'SRH, hi ha un risc significatiu de tornados.")])
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
        chat_log.extend([("Jo", f"Veig un CAPE de {cape.m:.0f} J/kg. És un valor considerable."),("Tempestes.cat", "Sí, indica energia suficient per a tempestes fortes, però no tan organitzades com una supercèl·lula."),("Jo", "I per què no s'organitzen més?"),("Tempestes.cat", f"La clau és el cisallament del vent, de només {shear_0_6:.1f} m/s. És massa feble per induir una rotació sostinguda. Les tempestes competiran entre elles en lloc de formar una única estructura dominant."),("Jo", "Quins fenòmens hem de vigilar?"),("Tempestes.cat", "Principalment xàfecs intensos que poden deixar calamarsa petita o moderada. Pels Castellanus, si la base del núvol és molt alta, el risc principal són els esclafits secs (downbursts).")])
    elif "Nimbostratus" in cloud_type:
        chat_log.extend([("Jo", "Aquí veig molta humitat però gairebé no hi ha inestabilitat."),("Tempestes.cat", f"Exacte. No hi ha un motor convectiu (CAPE de només {cape.m:.0f} J/kg), però l'atmosfera està saturada en una capa molt gruixuda. Això és característic de la pluja estratiforme, sovint associada a sistemes frontals."),("Jo", "La intensitat de la pluja depèn de l'aigua precipitable (PWAT), oi?"),])
        if "Intens" in cloud_type:
            chat_log.append(("Tempestes.cat", f"Sí. El PWAT en els primers 4 km és de {pwat_0_4.m:.1f} mm, un valor molt alt. Això es traduirà en pluges contínues i abundants, amb risc d'acumulacions importants."))
        elif "Moderat" in cloud_type:
            chat_log.append(("Tempestes.cat", f"Correcte. El PWAT en els primers 4 km és de {pwat_0_4.m:.1f} mm. És un valor considerable que alimentarà pluges moderades i persistents."))
        else:
            chat_log.append(("Tempestes.cat", f"Exactament. El PWAT és de {pwat_0_4.m:.1f} mm. És suficient per a ruixats febles i intermitents o plugims, però no s'esperen grans quantitats."))
    else:
        chat_log.extend([("Jo", "Sembla un dia bastant tranquil, oi?"),("Tempestes.cat", f"Sí, totalment. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),("Jo", "Veurem algun núvol?"),("Tempestes.cat", f"Probablement només alguns núvols de tipus {cloud_type} sense desenvolupament vertical ni risc de precipitació.")])
    return chat_log, precipitation_type

def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    if fz_h < 1500 or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            p_low = p_levels[p_levels.magnitude > (p_levels.magnitude[0] - 300)]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVÍS PER PLUJA GEBRADORA / AGUANIEVE", "Risc de pluja gelant o aguanieve. Extremi les precaucions.", "dodgerblue"
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
# === 3. FUNCIONS DE DIBUIX (SENCERES) ====================================
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
        idx = random.randint(1, len(tower_alts) - 1) if len(tower_alts) > 1 else 0
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
        color = 'mediumpurple' if p_type == 'sleet' else 'cornflowerblue'
        width = 1.6
        ax.add_patch(Rectangle((center_x - width / 2, ground_km), width, precip_base_km - ground_km, facecolor=color, alpha=0.45, lw=0, zorder=5))
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
    center_x, width = (base_x_left + base_x_right) / 2, base_x_right - base_x_left
    if f_type == 'wall_cloud':
        top_l, top_r = center_x - (width * 0.75 / 2), center_x + (width * 0.75 / 2)
        bot_l, bot_r = center_x - (width * 0.55 / 2), center_x + (width * 0.55 / 2)
        ax.add_patch(Polygon([(top_l, base_y), (top_r, base_y), (bot_r, base_y - 0.35), (bot_l, base_y - 0.35)], facecolor='#383838', edgecolor='#202020', lw=0.5, zorder=12))
    elif f_type == 'funnel':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, max(base_y - 0.8, ground_y + 0.5))], facecolor='darkgray', alpha=0.8, zorder=12))
    elif f_type == 'tornado':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, ground_y)], facecolor='#505050', zorder=12))
        ax.add_patch(Ellipse((center_x, ground_y + 0.05), width=0.7, height=0.25, facecolor='#654321', alpha=0.7, zorder=13))
    elif f_type == 'shelf_cloud':
        shelf_pts = [(base_x_left - 0.3, base_y), (base_x_right + 0.3, base_y), (base_x_right, base_y - 0.2), (base_x_left, base_y - 0.2)]
        ax.add_patch(Polygon(shelf_pts, facecolor='darkgray', edgecolor='gray', lw=0.5, zorder=12))
    elif f_type == 'base_rugosa':
        patches = []
        for _ in range(40):
            x = center_x + random.uniform(-width/2, width/2)
            y = base_y - random.uniform(0.05, 0.25)
            size = random.uniform(0.1, 0.3)
            patches.append(Circle((x,y), size, facecolor='gray', alpha=random.uniform(0.3, 0.6), lw=0))
        ax.add_collection(PatchCollection(patches, match_original=True, zorder=12))

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

def create_hodograph_figure(p_levels, wind_speed, wind_dir):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    u, v = mpcalc.wind_components(wind_speed.to('knots'), wind_dir)
    h.plot(u, v, color='blue', linewidth=2)
    
    heights = mpcalc.pressure_to_height_std(p_levels)
    max_h = heights.m.max()
    altitudes_km = np.array([0, 1, 3, 6, 9]) * units.km
    
    valid_altitudes_km = altitudes_km[altitudes_km.to('m').m <= max_h]
    
    if len(valid_altitudes_km) > 0:
        p_interp_func = interp1d(heights.m, p_levels.m, bounds_error=False, fill_value="extrapolate")
        p_points_mag = p_interp_func(valid_altitudes_km.to('m').m)

        interp_ws_knots = np.interp(p_points_mag, p_levels.m[::-1], wind_speed.to('knots').m[::-1])
        interp_wd_deg = np.interp(p_points_mag, p_levels.m[::-1], wind_dir.m[::-1])

        u_points, v_points = mpcalc.wind_components(interp_ws_knots * units.knots, interp_wd_deg * units.degrees)

        for i, (u_pt, v_pt, alt) in enumerate(zip(u_points, v_points, valid_altitudes_km)):
            ax.scatter(u_pt.m, v_pt.m, color='orange', s=50, zorder=10)
            ax.text(u_pt.m + 4, v_pt.m + 4, f'{alt.m:.0f}', fontsize=10, weight='bold', ha='center', va='center', zorder=10)

    ax.set_title("Hodògraf (nusos / km)", fontsize=10)
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
        if "Nimbostratus" in cloud_type: _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif cloud_type == "Cumulonimbus (Multicèl·lula)" or cloud_type == "Supercèl·lula": _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus": _draw_cumulus_castellanus(ax, base_km, top_km)
        elif cloud_type == "Cumulus Mediocris": _draw_cumulus_mediocris(ax, base_km, top_km)
        elif cloud_type == "Cumulus Fractus": _draw_cumulus_fractus(ax, base_km, top_km - base_km)
    elif not np.any((t_profile.m - td_profile.m) <= 1.5):
        _draw_clear_sky(ax)
    if precipitation_type and base_km is not None:
        precip_base_km = lfc_h / 1000.0 if cloud_type == "Castellanus" and lfc_h > 0 else base_km
        sub_cloud_rh_mean = 0.4
        try:
            p_base_precip = mpcalc.height_to_pressure_std(precip_base_km * units.kilometer)
            sub_cloud_mask = (p_levels >= p_base_precip) & (p_levels <= p_levels[0])
            if np.any(sub_cloud_mask):
                sub_cloud_rh_mean = np.mean(mpcalc.relative_humidity_from_dewpoint(t_profile[sub_cloud_mask], td_profile[sub_cloud_mask])).magnitude
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
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10); ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitud (km)", xticks=[]); ax.grid(True, linestyle='--', alpha=0.3)
    ax_shear.set(xlim=(-1, 1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    feature_label = "Base plana"
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
            if (srh_0_1 >= 150 and shear_0_6 > 20 and lcl_h <= 1000): feature = 'tornado'; feature_label = "Tornado"
            elif (srh_0_1 > 100 and shear_0_6 > 18 and lcl_h < 1200): feature = 'funnel'; feature_label = "Fibló (Funnel Cloud)"
            elif srh_0_3 > 250 and shear_0_6 > 20: feature = 'wall_cloud'; feature_label = "Núvol Mur (Wall Cloud)"
            elif shear_0_6 > 25: feature = 'shelf_cloud'; feature_label = "Núvol Prestatge (Shelf Cloud)"
            elif shear_0_6 > 15: feature = 'base_rugosa'; feature_label = "Base Rugosa"
        
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_height_km)
            
    except Exception as e: pass
    
    ax.text(0.5, 0.02, feature_label, ha='center', va='bottom', fontsize=12, color='white', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
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
# === 4. NOVES FUNCIONS PER A L'ESTRUCTURA DE L'APP ======================
# =========================================================================
def create_welcome_figure():
    """Dibuixa una escena de tempesta amb una supercèl·lula, llamps i un tornado."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0c0a1a')  # Cel nocturn fosc
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    # Fons amb gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 100, 0, 70], aspect='auto', cmap='Blues', alpha=0.5, origin='lower')

    # Terra
    ax.add_patch(Rectangle((0, 0), 100, 10, facecolor='#1a1a1a', zorder=1))

    # Cos principal del núvol (supercèl·lula)
    cloud_color_dark = '#2c3e50'
    cloud_color_mid = '#34495e'
    cloud_color_light = '#566573'

    # Capes de la supercèl·lula
    ax.add_patch(Ellipse((50, 45), 80, 25, facecolor=cloud_color_dark, alpha=0.8, zorder=5))
    ax.add_patch(Ellipse((50, 42), 70, 20, facecolor=cloud_color_mid, alpha=0.9, zorder=6))
    ax.add_patch(Ellipse((50, 40), 60, 15, facecolor=cloud_color_light, alpha=1, zorder=7))

    # Textura del núvol amb cercles
    patches = []
    for _ in range(300):
        x = random.gauss(50, 20)
        y = random.gauss(42, 5)
        size = random.uniform(2, 8)
        brightness = random.uniform(0.3, 0.6)
        alpha = random.uniform(0.1, 0.4)
        patch = Circle((x, y), size, facecolor=(brightness, brightness, brightness), alpha=alpha, lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=8))
    
    # Tornado
    tornado_base_y = 38
    tornado_points = [
        (48, tornado_base_y), (52, tornado_base_y),
        (51, 15), (53, 10), (49, 5), (50, 10)
    ]
    ax.add_patch(Polygon(tornado_points, facecolor='#202020', alpha=0.7, zorder=9))
    ax.add_patch(Ellipse((51, 10), 20, 4, facecolor='#383838', alpha=0.5, zorder=2)) # Pols

    # Llamps
    def draw_lightning(start_x, start_y, end_y):
        x = [start_x]
        y = [start_y]
        current_y = start_y
        while current_y > end_y:
            next_y = current_y - random.uniform(1, 5)
            next_x = x[-1] + random.uniform(-4, 4)
            y.append(next_y)
            x.append(next_x)
            current_y = next_y
        ax.plot(x, y, color='pink', linewidth=2, alpha=0.8, zorder=10)
        ax.plot(x, y, color='white', linewidth=0.5, alpha=0.9, zorder=11)

    draw_lightning(30, 45, 15)
    draw_lightning(70, 45, 20)

    plt.tight_layout(pad=0)
    return fig

def show_welcome_screen():
    # 1. Generar la figura del dibuix
    welcome_fig = create_welcome_figure()
    
    # 2. Convertir la figura a una imatge en format base64
    buf = io.BytesIO()
    welcome_fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(welcome_fig) # Tancar la figura per alliberar memòria

    # 3. Utilitzar la imatge base64 com a fons amb CSS
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .welcome-container {{
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(5px);
    }}
    .welcome-container h1, .welcome-container h3, .welcome-container p {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
        st.title("Benvingut al Visor de Sondejos de Tempestes.cat")
        st.subheader("Tria un mode per començar")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🛰️ Mode en Viu")
            st.markdown("<p>Visualitza els sondejos atmosfèrics basats en dades reals i l'hora actual d'Espanya. Navega entre les diferents hores disponibles.</p>", unsafe_allow_html=True)
            if st.button("Accedir al Mode en Viu", use_container_width=True):
                st.session_state.app_mode = 'live'
                st.rerun()
        with col2:
            st.markdown("### 🧪 Laboratori de Sondejos")
            st.markdown("<p>Experimenta amb un sondeig de proves. Modifica paràmetres com la temperatura i la humitat o carrega escenaris predefinits per entendre com afecten el temps.</p>", unsafe_allow_html=True)
            if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
                st.session_state.app_mode = 'sandbox'
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def apply_preset(preset_name):
    original_data = st.session_state.sandbox_original_data
    p_levels_hpa = st.session_state.sandbox_p_levels.magnitude
    t_new = original_data['t_initial'].to('degC').magnitude.copy()
    td_new = original_data['td_initial'].to('degC').magnitude.copy()
    ws_new = original_data['wind_speed_kmh'].to('m/s').magnitude.copy()
    wd_new = original_data['wind_dir_deg'].magnitude.copy()
    if preset_name == 'neu':
        sfc_temp_orig = t_new[0]
        temp_shift = -10.0 - sfc_temp_orig
        t_new += temp_shift
        td_new = t_new - np.random.uniform(0.5, 1.5, len(td_new))
    elif preset_name == 'aguanieve':
        sfc_temp_orig = t_new[0]
        temp_shift = -1.0 - sfc_temp_orig
        t_new += temp_shift
        warm_layer_mask = (p_levels_hpa > 700) & (p_levels_hpa < 850)
        t_new[warm_layer_mask] += 6
        td_new = t_new - np.random.uniform(0.5, 2, len(td_new))
    elif preset_name == 'calor':
        t_new += 15
        td_new = t_new - np.random.uniform(15, 25, len(td_new))
    elif preset_name == 'supercel':
        t_new[0] = 28.0; td_new[0] = 22.0
        inversion_mask = (p_levels_hpa > 800) & (p_levels_hpa < 900)
        t_new[inversion_mask] += 3
        p_profile_points = np.array([1000, 925, 850, 700, 500, 300])
        ws_profile_points_ms = np.array([10, 15, 20, 25, 35, 50])
        wd_profile_points_deg = np.array([140, 160, 180, 210, 240, 270])
        ws_new = np.interp(p_levels_hpa, p_profile_points[::-1], ws_profile_points_ms[::-1])
        wd_new = np.interp(p_levels_hpa, p_profile_points[::-1], wd_profile_points_deg[::-1])
    elif preset_name == 'pluja':
        td_new = t_new - np.random.uniform(1, 3, len(td_new))
    td_new = np.minimum(t_new, td_new)
    st.session_state.sandbox_t_profile = t_new * units.degC
    st.session_state.sandbox_td_profile = td_new * units.degC
    st.session_state.sandbox_ws = ws_new * units('m/s')
    st.session_state.sandbox_wd = wd_new * units.degrees

def run_display_logic(p, t, td, ws, wd, obs_time):
    cleaned_obs_time = obs_time.split('\n')[0]
    st.markdown(f"#### {cleaned_obs_time}")
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
    
    col_hodo, col_skew = st.columns([2, 5])
    with col_hodo:
        st.subheader("Hodògraf", anchor=False)
        fig_hodo = create_hodograph_figure(p, ws, wd)
        st.pyplot(fig_hodo, use_container_width=True)
    with col_skew:
        st.subheader("Diagrama Skew-T", anchor=False)
        fig_skewt = create_skewt_figure(p, t, td, ws, wd)
        st.pyplot(fig_skewt, use_container_width=True)
        
    st.divider()
    chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])
    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_fig = create_logo_figure()
        logo_buffer = io.BytesIO()
        logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        logo_base64 = base64.b64encode(logo_buffer.getvalue()).decode()
        css_styles = f"""<style>.chat-container {{ background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: Arial, sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }}.message-row {{ display: flex; align-items: flex-end; gap: 10px; }}.message-row-right {{ justify-content: flex-end; }}.message {{ padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }}.yo {{ background-color: #0078D4; color: white; }}.tempestes-cat {{ background-color: #FFFFFF; border: 1px solid #e0e0e0; }}.sistema {{ background-color: #E1F2FB; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }}.message strong {{ display: block; margin-bottom: 3px; font-weight: bold; }}.yo strong {{color: #FFFFFF;}}.tempestes-cat strong {{ color: #075E54; }}.profile-pic {{ width: 40px; height: 40px; border-radius: 50%; object-fit: cover; }}.online-status {{ text-align: center; font-size: 0.9em; color: #666; padding: 5px; }}</style>"""
        html_chat = "<div class='online-status'>Tempestes.cat • en línia</div><div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower().replace('.', '-')
            if speaker == "Tempestes.cat":
                html_chat += f"""<div class="message-row"><img src="data:image/png;base64,{logo_base64}" class="profile-pic"><div class="message {css_class}"><strong>{speaker}</strong> {message}</div></div>"""
            elif speaker == "Jo":
                html_chat += f"""<div class="message-row message-row-right"><div class="message {css_class}"><strong>{speaker}</strong> {message}</div></div>"""
            else:
                html_chat += f"<div class='message sistema'>{message}</div>"
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE (J/kg)", f"{cape.m:.0f}"); param_cols[1].metric("CIN (J/kg)", f"{cin.m:.0f}")
        param_cols[2].metric("PWAT Total (mm)", f"{pwat_total.m:.1f}"); param_cols[3].metric("Isoterma 0°C (km)", f"{fz_h/1000:.2f}")
        param_cols[0].metric("LCL (hPa)", f"{lcl_p.m:.0f}" if lcl_p else "N/A"); param_cols[1].metric("LFC (hPa)", f"{lfc_p.m:.0f}" if lfc_p else "N/A")
        param_cols[2].metric("EL (hPa)", f"{el_p.m:.0f}" if el_p else "N/A"); param_cols[3].metric("Cisallament 0-1km (m/s)", f"{s_0_1:.1f}")
        param_cols[0].metric("Cisallament 0-6km (m/s)", f"{shear_0_6:.1f}"); param_cols[1].metric("SRH 0-1km (m²/s²)", f"{srh_0_1:.1f}")
        param_cols[2].metric("SRH 0-3km (m²/s²)", f"{srh_0_3:.1f}");
        rh_display = "N/A"
        try: rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("RH Mitja 0-4km (%)", rh_display)
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

def run_live_mode():
    st.title("🛰️ Mode en Viu: Sondejos Reals")
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Mode Viu)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    if 'live_initialized' not in st.session_state:
        base_files = ['12am.txt'] + [f'{i}am.txt' for i in range(1, 12)] + ['12pm.txt'] + [f'{i}pm.txt' for i in range(1, 12)]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files:
            st.error("No s'ha trobat cap arxiu de sondeig per al mode en viu."); return
        madrid_tz = pytz.timezone('Europe/Madrid')
        now = datetime.now(madrid_tz)
        hour_12 = now.hour % 12 if now.hour % 12 != 0 else 12
        am_pm = 'am' if now.hour < 12 else 'pm'
        current_hour_file = f"{hour_12}{am_pm}.txt"
        initial_index = 0
        if current_hour_file in st.session_state.existing_files:
            initial_index = st.session_state.existing_files.index(current_hour_file)
        st.session_state.sounding_index = initial_index
        st.session_state.loaded_sounding_index = -1
        st.session_state.live_initialized = True
    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        selected_file = st.session_state.existing_files[st.session_state.sounding_index]
        soundings = parse_all_soundings(selected_file)
        if soundings:
            st.session_state.live_data = soundings[0]
            st.session_state.loaded_sounding_index = st.session_state.sounding_index
        else:
            st.error(f"No s'han pogut carregar dades de {selected_file}"); st.session_state.sounding_index = st.session_state.loaded_sounding_index; return
    with st.sidebar:
        def sync_index_from_selectbox():
            st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget)
        st.selectbox("Selecciona una hora:", options=st.session_state.existing_files, index=st.session_state.sounding_index, key='selectbox_widget', on_change=sync_index_from_selectbox)
    main_cols = st.columns([1, 10, 1])
    with main_cols[0]:
        if st.button('←', use_container_width=True, disabled=(st.session_state.sounding_index == 0)):
            st.session_state.sounding_index -= 1; st.rerun()
    with main_cols[2]:
        if st.button('→', use_container_width=True, disabled=(st.session_state.sounding_index >= len(st.session_state.existing_files) - 1)):
            st.session_state.sounding_index += 1; st.rerun()
    data = st.session_state.live_data
    run_display_logic(p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], obs_time=data.get('observation_time', 'Hora no disponible'))

def run_sandbox_mode():
    st.title("🧪 Laboratori de Sondejos")
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Laboratori)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings:
            st.error("No s'ha trobat o no s'ha pogut llegir 'sondeigproves.txt'. Aquest mode no pot funcionar."); return
        st.session_state.sandbox_original_data = soundings[0]
        st.session_state.sandbox_p_levels = st.session_state.sandbox_original_data['p_levels'].copy()
        st.session_state.sandbox_t_profile = st.session_state.sandbox_original_data['t_initial'].copy()
        st.session_state.sandbox_td_profile = st.session_state.sandbox_original_data['td_initial'].copy()
        st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True
    with st.sidebar:
        if st.button("🔄 Reiniciar al perfil original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_t_profile = data['t_initial'].copy()
            st.session_state.sandbox_td_profile = data['td_initial'].copy()
            st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
            st.rerun()
        st.markdown("---")
        st.subheader("Modificació Manual")
        sfc_t = st.session_state.sandbox_t_profile[0].magnitude
        new_sfc_t = st.slider("🌡️ Temperatura en Superfície (°C)", -20.0, 50.0, sfc_t, 0.5)
        sfc_td = st.session_state.sandbox_td_profile[0].magnitude
        new_sfc_td = st.slider("💧 Punt de Rosada en Superfície (°C)", -20.0, new_sfc_t, sfc_td, 0.5)
        st.session_state.sandbox_t_profile[0] = new_sfc_t * units.degC
        st.session_state.sandbox_td_profile[0] = new_sfc_td * units.degC
        st.markdown("---")
        st.subheader("Escenaris Predefinits")
        if st.button("❄️ Nevada Severa (-10°C)", use_container_width=True): apply_preset('neu'); st.rerun()
        if st.button("💧 Aguanieve (Capa càlida)", use_container_width=True): apply_preset('aguanieve'); st.rerun()
        if st.button("☀️ Calor Extrema", use_container_width=True): apply_preset('calor'); st.rerun()
        if st.button("🌪️ Supercèl·lula Clàssica", use_container_width=True): apply_preset('supercel'); st.rerun()
        if st.button("🌧️ Pluja Estratiforme", use_container_width=True): apply_preset('pluja'); st.rerun()
    run_display_logic(p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile, td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws, wd=st.session_state.sandbox_wd, obs_time="Sondeig de Prova - Mode Laboratori")

# =========================================================================
# === 6. PUNT D'ENTRADA DE L'APLICACIÓ ====================================
# =========================================================================
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()


