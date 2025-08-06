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
        months_fr_to_ca = {'janvier': 'de gener', 'février': 'de febrer', 'mars': 'de març', 'avril': 'd\'abril', 'mai': 'de maig', 'juin': 'de juny', 'juillet': 'de juliol', 'août': 'd\'agost', 'septembre': 'de setembre', 'octobre': 'd\'octubre', 'november': 'de novembre', 'décembre': 'de desembre'}
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
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
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
    elif lfc_p and el_p and el_p < lfc_p:
         precipitation_type = 'virga'

    chat_log = [("Tempestes.cat", f"Hola! Detecto una situació compatible amb la formació de núvols de tipus **{cloud_type}**.")]
    
    if cloud_type == "Hivernal":
        chat_log.extend([
            ("Yo", f"Veig una isoterma 0°C molt baixa, a {fz_h:.0f}m."),
            ("Tempestes.cat", "Exacte. Això, combinat amb la humitat en nivells baixos, és el factor clau."),
            ("Yo", f"La temperatura a la superfície és de {t_profile[0].m:.1f}°C. Què implica?"),
        ])
        if t_profile[0].m <= 0.5:
            chat_log.append(("Tempestes.cat", "Amb temperatures negatives o properes a 0°C a tots els nivells, la precipitació serà neu fins a cotes molt baixes."))
        else:
            chat_log.append(("Tempestes.cat", "Compte. Hi ha una petita capa càlida just sobre la superfície. Això pot provocar que la neu es fongui i es torni a congelar en contacte amb el terra (pluja gelant), un fenomen molt perillós."))
            
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([
            ("Yo", f"El CAPE és altíssim, {cape.m:.0f} J/kg. Què significa?"),
            ("Tempestes.cat", f"És l'energia disponible per a la tempesta. Un valor tan alt indica un potencial per a corrents ascendents extremadament violents, capaços de sostenir calamarsa de gran mida."),
            ("Yo", "I el cisallament del vent? Veig valors elevats."),
            ("Tempestes.cat", f"Correcte. El cisallament de {shear_0_6:.0f} m/s i l'helicitat (SRH) de {srh_0_3:.0f} m²/s² són els ingredients que permetran que la tempesta s'organitzi i roti, formant una supercèl·lula."),
            ("Yo", "Quin és el risc principal?"),
            ("Tempestes.cat", "Molt alt. Cal esperar calamarsa de gran mida (>4cm), ratxes de vent destructives i, amb un SRH 0-1km de {srh_0_1:.1f}, hi ha un risc significatiu de formació de tornados.")
        ])
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
         chat_log.extend([
            ("Yo", f"El CAPE és de {cape.m:.0f} J/kg. És molt?"),
            ("Tempestes.cat", "És un valor moderat a alt. Indica que hi ha energia suficient per a tempestes fortes, però no explosives."),
            ("Yo", "Per què no s'organitzen com una supercèl·lula?"),
            ("Tempestes.cat", f"El cisallament ({shear_0_6:.0f} m/s) és massa feble. Les tempestes competiran entre elles en lloc de formar una única estructura organitzada. Si són Castellanus, la convecció s'inicia a nivells més alts."),
            ("Yo", "Quins fenòmens podem esperar?"),
            ("Tempestes.cat", "Principalment xàfecs intensos i calamarsa de mida petita a moderada. En el cas dels Castellanus, el principal risc són els esclafits secs (downbursts) si la base està molt elevada.")
        ])
    elif "Nimbostratus" in cloud_type:
        chat_log.extend([
            ("Yo", "Veig molta humitat a capes baixes però gairebé gens d'inestabilitat (CAPE)."),
            ("Tempestes.cat", f"Exacte. No hi ha motor convectiu (CAPE de {cape.m:.0f} J/kg), però l'atmosfera està saturada en una capa molt gruixuda. Això és típic de la pluja estratiforme, associada a fronts."),
            ("Yo", "Com de potent serà la pluja? Depèn de l'aigua precipitable (PWAT), oi?"),
        ])
        if "Intens" in cloud_type:
             chat_log.append(("Tempestes.cat", f"Sí. El PWAT a la capa 0-4 km és de **{pwat_0_4.m:.1f} mm**, un valor molt alt. Això es traduirà en pluges **contínues i abundants**, amb risc d'acumulacions importants."))
        elif "Moderat" in cloud_type:
             chat_log.append(("Tempestes.cat", f"Correcte. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És un valor considerable que alimentarà xàfecs **moderats i persistents**, el que popularment anomenem 'petacs' de pluja."))
        else: # Fluix
             chat_log.append(("Tempestes.cat", f"Exactament. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És suficient per a **ruixats febles i intermitents** o plugims, però no s'esperen grans quantitats."))
    else:
        chat_log.extend([
            ("Yo", " sembla un dia tranquil, oi?"),
            ("Tempestes.cat", f"Sí. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),
            ("Yo", "Veurem algun núvol?"),
            ("Tempestes.cat", f"Probablement només alguns {cloud_type} sense cap mena de desenvolupament vertical ni risc de precipitació.")
        ])
        
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
# === 3. FUNCIONS DE DIBUIX ===============================================
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

def _draw_cumulus_mediocris(ax, base_km, top_km):
    center_x, num_particles = 0, 250
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

# ... (La resta de funcions de dibuix van aquí. Per brevetat, les ometo,
# però estarien presents al teu script final) ...
def _draw_cumulonimbus(ax, base_km, top_km): pass
def _draw_cumulus_castellanus(ax, base_km, top_km): pass
def _draw_nimbostratus(ax, base_km, top_km, cloud_type): pass
def _draw_cumulus_fractus(ax, base_km, thickness): pass
def _draw_clear_sky(ax): pass
def _draw_precipitation(ax, precip_base_km, ground_km, p_type, center_x=0.0, sub_cloud_rh=0.4): pass
def _draw_saturation_layers(ax, p_levels, t_profile, td_profile): pass
def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active): return None, None
def _draw_base_feature(ax, f_type, base_x_left, base_x_right, base_y, ground_y): pass
def create_cloud_drawing_figure(p, t, td, conv, precip, lfc, cape, base, top, ctype): return plt.figure()
def create_cloud_structure_figure(p, t, td, ws, wd, conv): return plt.figure()
def create_radar_figure(p, t, td, ws, wd): return plt.figure()


def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100); ax.set_xlim(-50, 45)
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
    skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    td_profile = np.minimum(t_profile, td_profile)
    skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
    parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
    skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
    skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    xlims = ax.get_xlim()
    if lcl_p: ax.plot(xlims, [lcl_p.m, lcl_p.m], 'gray', linestyle='--', label='LCL')
    if lfc_p: ax.plot(xlims, [lfc_p.m, lfc_p.m], 'purple', linestyle='--', label='LFC')
    if el_p: ax.plot(xlims, [el_p.m, el_p.m], 'red', linestyle='--', label='EL')
    ax.legend(); plt.tight_layout()
    return fig

# =========================================================================
# === 5. LÒGICA DE L'APLICACIÓ STREAMLIT =================================
# =========================================================================

def load_new_sounding_data():
    st.session_state.selected_file = st.session_state.existing_files[st.session_state.sounding_index]
    soundings = parse_all_soundings(st.session_state.selected_file)
    if soundings:
        st.session_state.original_data = soundings[0]
        reset_working_profiles()
        st.session_state.loaded_sounding_index = st.session_state.sounding_index
    else:
        st.error(f"Error en carregar les dades de {st.session_state.selected_file}")
        st.session_state.sounding_index = st.session_state.loaded_sounding_index # Revertir si falla

def reset_working_profiles():
    data = st.session_state.original_data
    st.session_state.p_levels = data['p_levels'].copy()
    st.session_state.t_profile = data['t_initial'].copy()
    st.session_state.td_profile = data['td_initial'].copy()
    st.session_state.wind_speed = data['wind_speed_kmh'].to('m/s')
    st.session_state.wind_dir = data['wind_dir_deg'].copy()
    st.session_state.observation_time = data.get('observation_time', 'Hora no disponible')

def main():
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")

    if 'initialized' not in st.session_state:
        base_files = ["1am.txt", "2am.txt", "3am.txt", "4am.txt", "5am.txt", "6am.txt", "7am.txt", "8am.txt", "9am.txt", "10am.txt", "11am.txt", "12pm.txt", "1pm.txt", "2pm.txt", "3pm.txt", "4pm.txt", "5pm.txt", "6pm.txt", "7pm.txt", "8pm.txt", "9pm.txt", "10pm.txt", "11pm.txt", "12am.txt"]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files:
            st.error("Error: No s'ha trobat cap arxiu de sondeig.")
            st.stop()
        st.session_state.sounding_index = 0
        st.session_state.loaded_sounding_index = -1 
        st.session_state.convergence_active = True
        st.session_state.initialized = True
    
    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        load_new_sounding_data()

    def increment_index():
        if st.session_state.sounding_index < len(st.session_state.existing_files) - 1:
            st.session_state.sounding_index += 1

    def decrement_index():
        if st.session_state.sounding_index > 0:
            st.session_state.sounding_index -= 1

    def sync_index_from_selectbox():
        st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget)
    
    logo_fig = create_logo_figure()
    
    with st.sidebar:
        st.pyplot(logo_fig)
        st.title("Controls")
        st.selectbox("Selecciona una hora:", 
                     options=st.session_state.existing_files, 
                     index=st.session_state.sounding_index,
                     key='selectbox_widget', 
                     on_change=sync_index_from_selectbox)
        
        st.toggle("Activar convergència", value=st.session_state.convergence_active, key='convergence_active')
        if st.button("🔄 Reiniciar Perfils"):
            reset_working_profiles(); st.success("Perfils reiniciats.")
        with st.expander("🔬 Modificació Avançada"):
            sfc_temp_val = st.session_state.t_profile[0].magnitude
            new_sfc_temp = st.slider("Temperatura en Superfície (°C)", sfc_temp_val - 20, sfc_temp_val + 20, sfc_temp_val, 0.5)
            if new_sfc_temp != sfc_temp_val: 
                st.session_state.t_profile[0] = new_sfc_temp * units.degC

    st.title("Visor de Sondejos Atmosfèrics")

    time_parts = st.session_state.observation_time.split('\n')
    cleaned_time_str = next((p.strip() for p in time_parts if 'local' in p.lower()), (time_parts[0].strip() if time_parts else ""))
    st.markdown(f"#### {cleaned_time_str}")

    p, t, td, ws, wd = st.session_state.p_levels, st.session_state.t_profile, st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)

    sub_cols = st.columns([2, 8, 2])
    with sub_cols[0]: st.button('← Anterior', on_click=decrement_index, disabled=(st.session_state.sounding_index == 0), use_container_width=True)
    with sub_cols[1]: st.subheader("Diagrama Skew-T", anchor=False)
    with sub_cols[2]: st.button('Següent →', on_click=increment_index, disabled=(st.session_state.sounding_index >= len(st.session_state.existing_files) - 1), use_container_width=True)

    st.pyplot(create_skewt_figure(p, t, td, ws, wd), use_container_width=True)
    st.divider()

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, st.session_state.convergence_active)
    
    cloud_type, pwat_0_4, rh_0_4 = "Cel Serè", units.Quantity(0, 'mm'), 0.0
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p).to('m'); heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_0_4 = np.mean(mpcalc.relative_humidity_from_dewpoint(t[layer_mask], td[layer_mask]))
            pwat_0_4 = mpcalc.precipitable_water(p[layer_mask], td[layer_mask]).to('mm')
    except Exception: pass

    sfc_temp = t[0]; 
    if sfc_temp.m < 5 or fz_h < 1500: cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape.m < 350:
        if pwat_0_4.m > 25: cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4.m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150: cloud_type = "Supercèl·lula"
    elif cape.m > 500: cloud_type = "Cumulonimbus (Multicèl·lula)" if lfc_h < 3000 else "Castellanus"
    elif base_km and top_km:
        cloud_thickness = top_km - base_km
        if cloud_thickness > 2.0 and lfc_h < 3000: cloud_type = "Cumulus Mediocris"
        elif cloud_thickness > 0: cloud_type = "Cumulus Fractus"

    chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres", "☁️ Visualització", "📡 Radar"])

    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_buffer = io.BytesIO(); logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        logo_base64 = base64.b64encode(logo_buffer.getvalue()).decode()
        css_styles = """<style>...</style>""" # El CSS per al xat
        html_chat = "<div class='online-status'>Tempestes.cat • en línia</div><div class='chat-container'>"
        for speaker, message in chat_log:
            # ... Lògica per construir el HTML del xat estàtic
            pass
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
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm"); param_cols[3].metric("RH Mitja 0-4km", f"{rh_0_4*100:.0f}%")

    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]: st.pyplot(create_cloud_drawing_figure(p, t, td, st.session_state.convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type), use_container_width=True)
        with cloud_cols[1]: st.pyplot(create_cloud_structure_figure(p, t, td, ws, wd, st.session_state.convergence_active), use_container_width=True)
            
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

if __name__ == '__main__':
    main()
