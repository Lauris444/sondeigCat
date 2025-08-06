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
import pytz  # Para manejo de zonas horarias

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
        
        # Càlcul de parcel·la
        parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
        
        # Càlcul CAPE/CIN amb comprovació d'errors
        with integrator_lock:
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
        
        # Càlcul nivells característics
        lcl_p, lcl_t = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
        lfc_p, lfc_t = mpcalc.lfc(p, t, td, parcel_prof)
        el_p, el_t = mpcalc.el(p, t, td, parcel_prof)
        
        # Càlcul isoterma 0°C
        try:
            valid_p = p[~np.isnan(t)]
            if len(valid_p) < 2: 
                fz_lvl = np.nan * units.hPa
            else:
                t_interp = interp1d(valid_p.m, t[~np.isnan(t)].m, bounds_error=False, fill_value="extrapolate")
                p_range = np.linspace(valid_p.m.min(), valid_p.m.max(), 100)
                t_range = t_interp(p_range) * units.degC
                below_zero = np.where(t_range < 0 * units.degC)[0]
                fz_lvl = p_range[below_zero[0]] * units.hPa if below_zero.size > 0 else np.nan * units.hPa
        except Exception: 
            fz_lvl = np.nan * units.hPa
        
        # Conversió a altituds
        lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m') if lcl_p else None
        lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m') if lfc_p else None
        el_h = mpcalc.pressure_to_height_std(el_p).to('m') if el_p else None
        fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m') if not np.isnan(fz_lvl.m) else None
        
        return (cape, cin, 
                lcl_p, lcl_h, 
                lfc_p, lfc_h, 
                el_p, el_h, 
                fz_h)
                
    except Exception as e:
        st.warning(f"Error en càlculs termodinàmics: {str(e)}")
        return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), 
                None, None, 
                None, None, 
                None, None, 
                None)

def calculate_storm_parameters(p_levels, wind_speed, wind_dir):
    try:
        p, ws, wd = p_levels, wind_speed, wind_dir
        u, v = mpcalc.wind_components(ws, wd)
        heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
        
        # Filtrar valors no vàlids
        valid_mask = ~np.isnan(heights_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        p_c, u_c, v_c, h_c = p[valid_mask], u[valid_mask], v[valid_mask], heights_raw[valid_mask]
        
        # Eliminar duplicats
        _, unique_indices = np.unique(h_c.m, return_index=True)
        if len(unique_indices) < 2: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        p_u, u_u, v_u, h_u = p_c[unique_indices], u_c[unique_indices], v_c[unique_indices], h_c[unique_indices]
        
        # Definir rangs d'interpolació
        h_min, h_max = h_u.m.min(), min(h_u.m.max(), 16000)
        if h_max <= h_min: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        h_interp = np.arange(h_min, h_max, 50) * units.meter
        u_i = np.interp(h_interp.m, h_u.m, u_u.m) * units('m/s')
        v_i = np.interp(h_interp.m, h_u.m, v_u.m) * units('m/s')
        
        # Càlcul de cisallaments
        u_6, v_6 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        
        u_1, v_1 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        
        # Càlcul helicitat
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter)[0].m
        
        # Càlcul BRN
        brn = mpcalc.bulk_richardson_number(p, u_i, v_i, h_interp, depth=6000 * units.meter).m
        
        # Càlcul Energy Helicity Index
        ehi = (cape.m * srh_0_1) / 160000 if cape.m > 0 else 0
        
        return s_0_6, s_0_1, srh_0_1, srh_0_3, brn, ehi
        
    except Exception as e:
        st.warning(f"Error en càlculs de tempesta: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, base_km, top_km, pwat_0_4):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, s_0_1, srh_0_1, srh_0_3, brn, ehi = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    
    precipitation_type = None
    if fz_h and fz_h < 1500 * units.m or t_profile[0].m < 5:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'sleet'
    elif cape and cape.m > 3000:
        precipitation_type = 'hail'
    elif cape and cape.m > 500:
        precipitation_type = 'rain'
    elif "Nimbostratus" in cloud_type:
        precipitation_type = 'rain'
    elif lfc_p and el_p and lfc_p.magnitude > el_p.magnitude:
        precipitation_type = 'virga'
    
    chat_log = [("Tempestes.cat", f"Hola! Detecto una situació compatible amb la formació de núvols de tipus **{cloud_type}**.")]
    
    # Anàlisi per tipus de núvol
    if cloud_type == "Hivernal":
        chat_log.extend([
            ("Yo", f"Veig una isoterma 0°C a {fz_h.m:.0f}m."),
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
            ("Tempestes.cat", f"Molt alt. Cal esperar calamarsa de gran mida (>4cm), ratxes de vent destructives i, amb un SRH 0-1km de {srh_0_1:.1f}, hi ha un risc significatiu de formació de tornados.")
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
        else:
            chat_log.append(("Tempestes.cat", f"Exactament. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És suficient per a **ruixats febles i intermitents** o plugims, però no s'esperen grans quantitats."))
    
    else:
        chat_log.extend([
            ("Yo", "Sembla un dia tranquil, oi?"),
            ("Tempestes.cat", f"Sí. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),
            ("Yo", "Veurem algun núvol?"),
            ("Tempestes.cat", f"Probablement només alguns {cloud_type} sense cap mena de desenvolupament vertical ni risc de precipitació.")
        ])
    
    return chat_log, precipitation_type

def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    # Casos hivernals
    if fz_h and fz_h < 1500 * units.m or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades. Extremi les precaucions.", "dodgerblue"
    
    # Casos convectius
    if cape and cape.m >= 1000:
        shear_0_6, s_0_1, srh_0_1, srh_0_3, brn, ehi = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        
        if srh_0_1 > 150 and shear_0_6 > 15 and ehi > 2.5:
            return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        
        if lfc_h and lfc_h > 3000 * units.m:
            return "AVÍS PER TEMPESTES DE BASE ALTA", "Nuclis de base alta. Risc de ratxes de vent fortes i sobtades (downbursts).", "darkorange"
        
        if cape.m > 2000 and srh_0_1 > 100:
            return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa. Protegiu vehicles.", "purple"
        
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
    
    # Casos estratiformes
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
    
    cloud_verts = [(2, 5.8), (1.5, 6.8), (2.5, 7.8), (4, 8.3), (6, 8.3), 
                   (7.5, 7.8), (8.5, 6.8), (8, 5.8), (7, 5.3), (3, 5.3)]
    ax.add_patch(Polygon(cloud_verts, facecolor=cloud_color, zorder=10))
    
    ax.text(5, 6.6, 'tempestes.cat', ha='center', va='center', 
            fontsize=3.3, color='white', weight='bold', fontfamily='sans-serif', zorder=20)
    
    bar_heights, start_x, bar_width, rain_start_y = [0.8, 1.0, 0.9, 0.7, 0.95, 0.85, 0.6, 0.75, 0.5], 3.0, 0.4, 5.3
    for i, h in enumerate(bar_heights):
        x_pos, color, bar_height = start_x + i * bar_width, senyera_red if i % 2 == 0 else senyera_yellow, h * 4.0
        ax.add_patch(Rectangle((x_pos + 0.05, rain_start_y - bar_height - 0.05), bar_width, bar_height, 
                     facecolor='black', alpha=0.3, lw=0, zorder=4))
        ax.add_patch(Rectangle((x_pos, rain_start_y - bar_height), bar_width, bar_height, 
                     facecolor=color, lw=0, zorder=5))
    
    return fig

def create_hodograph_figure(p_levels, wind_speed, wind_dir):
    fig = plt.figure(figsize=(5, 5))
    hodo = Hodograph(fig, component_range=60)
    
    try:
        # Convert wind speed to m/s
        wspd_ms = wind_speed.to('m/s')
        
        # Calculate wind components
        u, v = mpcalc.wind_components(wspd_ms, wind_dir)
        
        # Filter valid data
        valid_mask = ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2:
            plt.text(0.5, 0.5, "Dades insuficients", ha='center', va='center')
            return fig
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        p_valid = p_levels[valid_mask]
        
        # Define height layers
        layers = [
            (0, 3000, 'Baixos (0-3km)', 'blue'),
            (3000, 6000, 'Mitjos (3-6km)', 'green'),
            (6000, 9000, 'Alts (6-9km)', 'red')
        ]
        
        # Calculate heights
        heights = mpcalc.pressure_to_height_std(p_valid).to('m')
        
        # Plot each layer
        for bottom, top, label, color in layers:
            layer_mask = (heights >= bottom * units.m) & (heights < top * units.m)
            if np.sum(layer_mask) > 1:
                hodo.plot(u_valid[layer_mask], v_valid[layer_mask], color=color, linewidth=2, label=label)
        
        # Add wind barbs
        for i in range(0, len(u_valid), max(1, len(u_valid)//5)):
            hodo.plot_colormapped(u_valid[i], v_valid[i], heights[i])
        
        # Add storm motion vectors
        mean_u = np.mean(u_valid).m
        mean_v = np.mean(v_valid).m
        hodo.plot(mean_u, mean_v, 'ko', markersize=8, label='Mitjana')
        
        # Calculate and plot Bunkers vectors
        if len(u_valid) > 10:
            try:
                rm_u, rm_v = mpcalc.bunkers_storm_motion(p_valid, u_valid, v_valid, heights)
                hodo.plot(rm_u, rm_v, 'ro', markersize=8, label='Moviment Tempesta')
            except Exception:
                pass
        
        hodo.add_grid(increment=10)
        plt.legend(loc='upper right')
        plt.title('Hodògraf i Moviment de Tempesta')
        
    except Exception as e:
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
    
    plt.tight_layout()
    return fig

def _draw_cumulonimbus(ax, base_km, top_km):
    updraft_center_x, num_points = 0, 20
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    
    # Draw main cloud body
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    if len(tower_indices) == 0: 
        tower_indices = np.arange(len(altitudes))
    
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    widths += np.random.uniform(-0.05, 0.05, len(tower_indices))
    
    r_pts = [(updraft_center_x + widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    l_pts = [(updraft_center_x - widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    main_poly_pts = [(l_pts[0][0], l_pts[0][1])] + r_pts + l_pts[::-1]
    
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d8d8d8', lw=0, zorder=10))
    
    # Add texture to main cloud
    for _ in range(120):
        idx = random.randint(1, len(tower_alts) - 1)
        y = tower_alts[idx] + random.uniform(-0.3, 0.3)
        max_x_at_y = np.interp(y, tower_alts, widths, left=widths[0], right=widths[-1])
        x = updraft_center_x + random.uniform(-max_x_at_y, max_x_at_y)
        size = random.uniform(0.2, 0.6) * (1 + (y - base_km) / (top_km - base_km))
        brightness = np.clip(0.85 + 0.15 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
        ax.add_patch(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.1, 0.35), lw=0, zorder=11))
    
    # Draw anvil
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

def _draw_nimbostratus(ax, base_km, top_km, cloud_type):
    if "Intens" in cloud_type:
        color, alpha = '#808080', 0.95
    elif "Moderat" in cloud_type:
        color, alpha = '#a9a9a9', 0.9
    else:
        color, alpha = '#c0c0c0', 0.85
    
    # Draw main cloud layer
    ax.add_patch(Rectangle((-1.7, base_km), 3.4, top_km - base_km, 
                 facecolor=color, lw=0, zorder=8, alpha=alpha))
    
    # Add texture
    patches = []
    for _ in range(150):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.6, 0.75)
        patch = Ellipse((x, y), width=random.uniform(0.8, 1.5), 
                        height=random.uniform(0.1, 0.3), 
                        facecolor=(b, b, b), alpha=random.uniform(0.2, 0.4), lw=0)
        patches.append(patch)
    
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))

def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    
    # Configuració eixos
    ax.set_ylim(1050, 100)
    ax.set_xlim(-50, 45)
    
    # Dibuix adiabàtiques
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    
    # Perfil temperatura i rosada
    td_profile = np.minimum(t_profile, td_profile)
    skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
    
    # Perfil bombolla adiabàtica
    parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
    skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    
    # Temperatura de bombolla humida
    wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
    skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='Tª Bombolla Humida')
    
    # Àrees CAPE/CIN
    skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
    skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    
    # Nivells característics
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    xlims = ax.get_xlim()
    
    if lcl_p: 
        ax.plot(xlims, [lcl_p.m, lcl_p.m], 'gray', linestyle='--', label='LCL')
    if lfc_p: 
        ax.plot(xlims, [lfc_p.m, lfc_p.m], 'purple', linestyle='--', label='LFC')
    if el_p: 
        ax.plot(xlims, [el_p.m, el_p.m], 'red', linestyle='--', label='EL')
    
    ax.legend()
    plt.title('Diagrama Skew-T Log-p', fontsize=14)
    plt.tight_layout()
    return fig

def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type):
    fig, ax = plt.subplots(figsize=(5, 8))
    
    # Calcular alçada terra
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    
    # Configuració gràfica
    ax.set(ylim=(0,16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud (km)")
    ax.set_title("Visualització del Núvol", fontsize=12)
    ax.grid(True, linestyle='dashdot', alpha=0.5)
    ax.set_facecolor('#6495ED')
    
    # Sol
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    
    # Terra
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ground_pattern = '///' if precipitation_type == 'snow' else None
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, 
                 color=ground_color, alpha=0.8, zorder=3, hatch=ground_pattern))
    
    # Capes saturació
    _draw_saturation_layers(ax, p_levels, t_profile, td_profile)
    
    # Dibuix núvols
    if base_km and top_km:
        if "Nimbostratus" in cloud_type:
            _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Supercèl·lula"]:
            _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus":
            _draw_cumulus_castellanus(ax, base_km, top_km)
    else:
        _draw_clear_sky(ax)
    
    # Precipitació
    if precipitation_type and base_km:
        is_castellanus = (cloud_type == "Castellanus")
        precip_base_km = lfc_h.m/1000 if is_castellanus and lfc_h else base_km
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
    
    # Configuració inicial
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set_title("Estructura Vertical i Cisallament", fontsize=12)
    ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitud (km)", xticks=[])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Configuració eix cisallament
    ax_shear.set(xlim=(-1, 1), xticks=[])
    ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values():
        spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    
    # Obtenir paràmetres
    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    # Comprovar si hi ha estructura convectiva
    if not base_km or not top_km or not cape or cape.m < 100:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva", 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=11, color='white', bbox=dict(facecolor='darkblue', alpha=0.7))
        ax_shear.axis('off')
        return fig
    
    # Dibuixar núvol
    visual_base_km = max(base_km, ground_height_km + 0.5)
    
    try:
        # Calcular components vent
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        h_km = mpcalc.pressure_to_height_std(p_levels).to('km').m
        
        # Filtrar dades vàlides
        valid_mask = ~np.isnan(h_km) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2:
            return fig
        
        h_valid = h_km[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        # Interpolar per obtenir perfils regulars
        unique_h, idx = np.unique(h_valid, return_index=True)
        f_u = interp1d(unique_h, u_valid.m[idx], bounds_error=False, fill_value='extrapolate')
        f_v = interp1d(unique_h, v_valid.m[idx], bounds_error=False, fill_value='extrapolate')
        
        # Dibuixar barbes de vent
        barb_heights = np.arange(0, min(20, h_valid.max()), 1)
        ax_shear.barbs(np.zeros_like(barb_heights), barb_heights, 
                       (f_u(barb_heights) * units('m/s')).to('knots').m, 
                       (f_v(barb_heights) * units('m/s')).to('knots').m, 
                       length=7, pivot='middle', color='k')
        
        # Dibuixar estructura del núvol
        altitudes = np.linspace(visual_base_km, top_km, num=50)
        u_at_alts = f_u(altitudes)
        horizontal_offsets = u_at_alts * 0.02
        
        # Factor de cisallament
        shear_0_6, s_0_1, srh_0_1, srh_0_3, brn, ehi = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        shear_factor = np.clip(shear_0_6 / 35, 0.4, 2.5)
        
        # Amplada del núvol
        updraft_widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (altitudes - visual_base_km) / (top_km - visual_base_km + 0.01))) * shear_factor
        
        # Extensió anell
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
        
        # Punts per contorn
        r_pts = [(updraft_widths[i] + horizontal_offsets[i] + anvil_extension[i], altitudes[i]) for i in range(len(altitudes))]
        l_pts = [(-updraft_widths[i] + horizontal_offsets[i], altitudes[i]) for i in range(len(altitudes))]
        
        # Dibuixar núvol
        ax.add_patch(Polygon(r_pts + l_pts[::-1], facecolor='white', edgecolor='lightgray', alpha=0.95, zorder=10))
        
        # Característiques especials
        _, _, lcl_p, lcl_h, _, _, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
        feature = None
        
        if top_km - base_km > 4.0 and cape.m > 500:
            if (srh_0_1 >= 150 and lcl_h and lcl_h.m <= 1000 and shear_0_6 > 15):
                feature = 'tornado'
            elif (srh_0_1 > 100 and lcl_h and lcl_h.m < 1200 and shear_0_6 > 12):
                feature = 'funnel'
            elif srh_0_3 > 150 and shear_0_6 > 18 and cape.m > 1000:
                feature = 'wall_cloud'
            elif s_0_1 > 8 and lcl_h and lcl_h.m < 1500:
                feature = 'lowering'
        
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_height_km)
            
    except Exception as e:
        st.warning(f"Error en dibuixar estructura: {str(e)}")
    
    plt.tight_layout()
    return fig

# =========================================================================
# === 4. NOVES FUNCIONS PER A L'ESTRUCTURA DE L'APP ======================
# =========================================================================

def show_welcome_screen():
    st.title("Benvingut al Visor de Sondejos de Tempestes.cat")
    logo_fig = create_logo_figure()
    st.pyplot(logo_fig)
    
    st.subheader("Tria un mode per començar")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛰️ Mode en Viu")
        st.info("Visualitza els sondejos atmosfèrics basats en dades reals i la teva hora local. Navega entre les diferents hores disponibles.")
        if st.button("Accedir al Mode en Viu", use_container_width=True, type="secondary"):
            st.session_state.app_mode = 'live'
            st.rerun()
    
    with col2:
        st.markdown("### 🧪 Laboratori de Sondejos")
        st.info("Experimenta amb un sondeig de proves. Modifica paràmetres com la temperatura i la humitat o carrega escenaris predefinits per entendre com afecten el temps.")
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
            st.session_state.app_mode = 'welcome'
            st.rerun()
        
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active',
                 help="Simula l'efecte de la convergència de vents en la formació de núvols")
    
    # Inicialització dades
    if 'live_initialized' not in st.session_state:
        base_files = ['12am.txt'] + [f'{i}am.txt' for i in range(1, 12)] + ['12pm.txt'] + [f'{i}pm.txt' for i in range(1, 12)]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        
        if not st.session_state.existing_files:
            st.error("No s'ha trobat cap arxiu de sondeig per al mode en viu.")
            return
        
        # Obtenir hora actual a Madrid
        madrid_tz = pytz.timezone('Europe/Madrid')
        now_madrid = datetime.now(madrid_tz)
        
        # Determinar fitxer actual
        hour_12 = now_madrid.hour % 12
        hour_12 = 12 if hour_12 == 0 else hour_12
        am_pm = 'am' if now_madrid.hour < 12 else 'pm'
        current_hour_file = f"{hour_12}{am_pm}.txt"
        
        # Seleccionar índex inicial
        initial_index = 0
        if current_hour_file in st.session_state.existing_files:
            initial_index = st.session_state.existing_files.index(current_hour_file)
        
        st.session_state.sounding_index = initial_index
        st.session_state.loaded_sounding_index = -1
        st.session_state.live_initialized = True
    
    # Carregar dades seleccionades
    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        selected_file = st.session_state.existing_files[st.session_state.sounding_index]
        soundings = parse_all_soundings(selected_file)
        
        if soundings:
            st.session_state.live_data = soundings[0]
            st.session_state.loaded_sounding_index = st.session_state.sounding_index
        else:
            st.error(f"No s'han pogut carregar dades de {selected_file}")
            st.session_state.sounding_index = st.session_state.loaded_sounding_index
            return
    
    # Controles de navegació
    with st.sidebar:
        def sync_index_from_selectbox():
            st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget)
        
        st.selectbox("Selecciona una hora:", 
                    options=st.session_state.existing_files, 
                    index=st.session_state.sounding_index, 
                    key='selectbox_widget', 
                    on_change=sync_index_from_selectbox)
    
    # Botons navegació
    main_cols = st.columns([1, 10, 1])
    with main_cols[0]:
        if st.button('←', use_container_width=True, disabled=(st.session_state.sounding_index == 0)):
            st.session_state.sounding_index -= 1
            st.rerun()
    
    with main_cols[2]:
        if st.button('→', use_container_width=True, disabled=(st.session_state.sounding_index >= len(st.session_state.existing_files) - 1)):
            st.session_state.sounding_index += 1
            st.rerun()
    
    # Mostrar dades
    data = st.session_state.live_data
    run_display_logic(
        p=data['p_levels'], 
        t=data['t_initial'], 
        td=data['td_initial'], 
        ws=data['wind_speed_kmh'].to('m/s'), 
        wd=data['wind_dir_deg'], 
        obs_time=data.get('observation_time', 'Hora no disponible')
    )

def run_display_logic(p, t, td, ws, wd, obs_time):
    # Cabecera
    st.markdown(f"#### {obs_time}")
    
    # Paràmetres clau
    convergence_active = st.session_state.get('convergence_active', True)
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3, brn, ehi = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    
    # Determinar tipus de núvol
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
    except Exception: 
        pass
    
    sfc_temp = t[0]
    if fz_h and fz_h < 1500 * units.m or sfc_temp.m < 5: 
        cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape and cape.m < 350:
        if pwat_0_4 and pwat_0_4.m > 25: 
            cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4 and pwat_0_4.m > 15: 
            cloud_type = "Nimbostratus (Moderat)"
        else: 
            cloud_type = "Nimbostratus (Fluix)"
    elif cape and cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150: 
        cloud_type = "Supercèl·lula"
    elif cape and cape.m > 500:
        cloud_type = "Cumulonimbus (Multicèl·lula)"
        if lfc_h and lfc_h.m >= 3000: 
            cloud_type = "Castellanus"
    elif base_km and top_km:
        if (top_km - base_km) > 2.0 and lfc_h and lfc_h.m < 3000: 
            cloud_type = "Cumulus Mediocris"
        elif (top_km - base_km) > 0: 
            cloud_type = "Cumulus Fractus"
    
    # Generar avís públic
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    
    # Mostrar avís
    st.markdown(f"""
    <div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color:white; text-align:center;">{title}</h3>
        <p style="color:white; text-align:center; font-size:16px;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Diagrama Skew-T
    st.subheader("Diagrama Skew-T", anchor=False)
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    
    st.divider()
    
    # Anàlisi detallada
    chat_log, precipitation_type = generate_detailed_analysis(
        p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4
    )
    
    # Organització en pestanyes
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Anàlisi Detallada", 
        "📊 Paràmetres Detallats", 
        "☁️ Visualització de Núvols", 
        "📡 Simulació Radar",
        "🌪️ Hodògraf"
    ])
    
    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_buffer = io.BytesIO()
        create_logo_figure().savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        logo_base64 = base64.b64encode(logo_buffer.getvalue()).decode()
        
        # Estils CSS
        css_styles = f"""
        <style>
            .chat-container {{
                background-color: #f0f2f5;
                padding: 15px;
                border-radius: 10px;
                font-family: Arial, sans-serif;
                max-height: 450px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}
            .message-row {{
                display: flex;
                align-items: flex-end;
                gap: 10px;
            }}
            .message-row-right {{
                justify-content: flex-end;
            }}
            .message {{
                padding: 8px 14px;
                border-radius: 18px;
                max-width: 80%;
                box-shadow: 0 1px 1px rgba(0,0,0,0.1);
                position: relative;
                color: black;
            }}
            .yo {{
                background-color: #0078D4;
                color: white;
            }}
            .tempestes-cat {{
                background-color: #FFFFFF;
                border: 1px solid #e0e0e0;
            }}
            .sistema {{
                background-color: #E1F2FB;
                align-self: center;
                text-align: center;
                font-style: italic;
                font-size: 0.9em;
                color: #555;
                width: auto;
                max-width: 90%;
            }}
            .message strong {{
                display: block;
                margin-bottom: 3px;
                font-weight: bold;
            }}
            .yo strong {{
                color: #FFFFFF;
            }}
            .tempestes-cat strong {{
                color: #075E54;
            }}
            .profile-pic {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .online-status {{
                text-align: center;
                font-size: 0.9em;
                color: #666;
                padding: 5px;
            }}
        </style>
        """
        
        # Construir HTML del chat
        html_chat = "<div class='online-status'>Tempestes.cat • en línia</div><div class='chat-container'>"
        
        for speaker, message in chat_log:
            css_class = speaker.lower().replace('.', '-')
            
            if speaker == "Tempestes.cat":
                html_chat += f"""
                <div class="message-row">
                    <img src="data:image/png;base64,{logo_base64}" class="profile-pic">
                    <div class="message {css_class}">
                        <strong>{speaker}</strong>
                        {message}
                    </div>
                </div>
                """
            elif speaker == "Yo":
                html_chat += f"""
                <div class="message-row message-row-right">
                    <div class="message {css_class}">
                        <strong>{speaker}</strong>
                        {message}
                    </div>
                </div>
                """
            else:
                html_chat += f"<div class='message sistema'>{message}</div>"
        
        html_chat += "</div>"
        
        # Mostrar el chat
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        
        # Columnes per a mètriques
        param_cols = st.columns(4)
        
        # Termodinàmics
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg" if cape else "N/A")
        param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg" if cin else "N/A")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm" if pwat_total else "N/A")
        param_cols[3].metric("0°C", f"{fz_h.m/1000:.2f} km" if fz_h else "N/A")
        
        # Nivells
        param_cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A")
        param_cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A")
        param_cols[3].metric("Shear 0-6km", f"{shear_0_6:.1f} m/s")
        
        # Dinàmics
        param_cols[0].metric("SRH 0-1km", f"{srh_0_1:.1f} m²/s²")
        param_cols[1].metric("SRH 0-3km", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("BRN", f"{brn:.1f}")
        param_cols[3].metric("EHI", f"{ehi:.1f}")
        
        # Humitat
        param_cols[0].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm" if pwat_0_4 else "N/A")
        
        # RH mitjana
        rh_display = "N/A"
        try:
            if hasattr(rh_0_4, 'm'):
                rh_display = f"{rh_0_4.m*100:.0f}%"
            else:
                rh_display = f"{rh_0_4*100:.0f}%"
        except:
            pass
        param_cols[1].metric("RH Mitja 0-4km", rh_display)
    
    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(
                p, t, td, convergence_active, precipitation_type, 
                lfc_h, cape, base_km, top_km, cloud_type
            )
            st.pyplot(fig_clouds, use_container_width=True)
            st.caption("Visualització 3D del núvol i precipitació")
        
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(
                p, t, td, ws, wd, convergence_active
            )
            st.pyplot(fig_structure, use_container_width=True)
            st.caption("Estructura interna i cisallament")
    
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)
        st.caption("Reflectivitat radar simulada basada en el perfil")
    
    with tab5:
        st.subheader("Anàlisi de Vent i Hodògraf")
        fig_hodo = create_hodograph_figure(p, ws, wd)
        st.pyplot(fig_hodo, use_container_width=True)
        st.caption("Hodògraf que mostra el canvi de vent amb l'altura")

# =========================================================================
# === 5. PUNT D'ENTRADA DE L'APLICACIÓ ====================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(
        layout="wide", 
        page_title="Visor de Sondejos",
        page_icon="🌦️"
    )
    
    # Inicialització estat
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    
    # Navegació entre modes
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
