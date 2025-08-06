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
import time

# Bloqueo global para cálculos con MetPy
integrator_lock = threading.Lock()

# =============================================================================
# === 1. FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS ==========================
# =============================================================================
def clean_and_convert(text):
    """Limpia y convierte texto a número, manejando valores faltantes."""
    cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
    if not cleaned_text or cleaned_text == '-': 
        return None
    try: 
        return float(cleaned_text)
    except ValueError: 
        return None

def process_sounding_block(block_lines):
    """Procesa un bloque de líneas de sondeo para extraer datos meteorológicos."""
    if not block_lines: 
        return None
    
    p_list, t_list, td_list, wdir_list, wspd_list = [], [], [], [], []
    time_lines = []
    time_keywords = ['observació', 'hora', 'time', 'locale', 'run', 'z', 'date']
    
    days_fr_to_ca = {
        'Lundi': 'Dilluns', 'Mardi': 'Dimarts', 'Mercredi': 'Dimecres', 
        'Jeudi': 'Dijous', 'Vendredi': 'Divendres', 'Samedi': 'Dissabte', 
        'Dimanche': 'Diumenge'
    }
    
    months_fr_to_ca = {
        'janvier': 'de gener', 'février': 'de febrer', 'mars': 'de març', 
        'avril': 'd\'abril', 'mai': 'de maig', 'juin': 'de juny', 
        'juillet': 'de juliol', 'août': 'd\'agost', 'septembre': 'de setembre', 
        'octobre': 'd\'octubre', 'novembre': 'de novembre', 'décembre': 'de desembre'
    }

    for line in block_lines:
        line_strip = line.strip()
        
        # Extraer información de tiempo
        if any(keyword in line_strip.lower() for keyword in time_keywords) and not (line_strip and line_strip[0].isdigit()):
            time_lines.append(line_strip)
            continue
            
        if not line_strip or line_strip.startswith('#') or 'Pression' in line_strip: 
            continue
            
        try:
            # Procesar datos meteorológicos
            parts = re.split(r'\s{2,}|[\t]', line_strip)
            if len(parts) < 7: 
                continue
                
            p = clean_and_convert(parts[1])
            t = clean_and_convert(parts[2])
            td = clean_and_convert(parts[4])
            
            if p is None or t is None or td is None: 
                continue
                
            p_list.append(p)
            t_list.append(t)
            td_list.append(td)
            
            # Procesar viento (dirección/velocidad)
            wdir, wspd = 0.0, 0.0
            try:
                wind_str = parts[6].strip()
                if '/' in wind_str:
                    wind_parts = wind_str.split('/')
                    if len(wind_parts) == 2:
                        wdir_val = clean_and_convert(wind_parts[0])
                        wspd_val = clean_and_convert(wind_parts[1])
                        if wdir_val is not None: 
                            wdir = wdir_val
                        if wspd_val is not None: 
                            wspd = wspd_val
            except IndexError: 
                pass
                
            wdir_list.append(wdir)
            wspd_list.append(wspd)
            
        except Exception as e:
            st.warning(f"Advertencia: Error procesando línea '{line_strip}'. Error: {e}")
            continue

    if not p_list or len(p_list) < 2: 
        return None

    # Traducir información de tiempo
    translated_lines = []
    for line in time_lines:
        translated_line = re.sub(r'\(.*?\)|locale', '', line, flags=re.IGNORECASE).strip()
        for fr, ca in days_fr_to_ca.items(): 
            translated_line = translated_line.replace(fr, ca)
        for fr, ca in months_fr_to_ca.items(): 
            translated_line = translated_line.replace(fr, ca)
        translated_lines.append(translated_line)
    
    observation_time = "\n".join(translated_lines) if translated_lines else "Hora no disponible"
    
    # Ordenar datos por presión (descendente)
    sorted_indices = np.argsort(p_list)[::-1]
    
    return {
        'p_levels': np.array(p_list)[sorted_indices] * units.hPa,
        't_initial': np.array(t_list)[sorted_indices] * units.degC,
        'td_initial': np.array(td_list)[sorted_indices] * units.degC,
        'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph,
        'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees,
        'observation_time': observation_time
    }

def parse_all_soundings(filepath):
    """Analiza un archivo de sondeos y devuelve todos los sondeos encontrados."""
    all_soundings_data = []
    current_sounding_lines = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{filepath}'. Asegúrate de que existe en el mismo directorio.")
        return []
    except UnicodeDecodeError:
        st.error(f"Error: No se pudo leer el archivo '{filepath}'. Problema de codificación.")
        return []
    
    for line in lines:
        if 'Pression' in line and (line.strip().startswith('Nivell') or line.strip().startswith('# Nivell')):
            if current_sounding_lines:
                processed_data = process_sounding_block(current_sounding_lines)
                if processed_data: 
                    all_soundings_data.append(processed_data)
            current_sounding_lines = []
        current_sounding_lines.append(line)
    
    if current_sounding_lines:
        processed_data = process_sounding_block(current_sounding_lines)
        if processed_data: 
            all_soundings_data.append(processed_data)
            
    return all_soundings_data

# =========================================================================
# === 2. FUNCIONES DE CÁLCULO Y ANÁLISIS ==================================
# =========================================================================
def calculate_thermo_parameters(p_levels, t_profile, td_profile):
    """Calcula parámetros termodinámicos clave a partir de perfiles de sondeo."""
    try:
        # Filtrar valores no válidos
        valid_indices = ~np.isnan(p_levels.magnitude) & ~np.isnan(t_profile.magnitude) & ~np.isnan(td_profile.magnitude)
        if np.sum(valid_indices) < 2: 
            raise ValueError("No hay suficientes datos válidos.")
            
        p = p_levels[valid_indices]
        t = t_profile[valid_indices]
        td = td_profile[valid_indices]
        
        p_sfc = p[0]
        t_sfc = t[0]
        td_sfc = td[0]
        
        # Calcular perfil de parcela
        with integrator_lock:
            parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
            lcl_p, _ = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
            lfc_p, _ = mpcalc.lfc(p, t, td, parcel_prof)
            el_p, _ = mpcalc.el(p, t, td, parcel_prof)
        
        # Calcular nivel de congelación
        try:
            t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")
            p_range = np.arange(p.m.min(), p.m.max())
            t_range = t_interp(p_range)
            fz_idx = np.where(t_range < 0)[0]
            fz_lvl = p_range[fz_idx[0]] * units.hPa if fz_idx.size > 0 else np.nan * units.hPa
        except Exception: 
            fz_lvl = np.nan * units.hPa
            
        if el_p is None and cape.magnitude > 0: 
            el_p = p[-1]
            
        # Convertir a alturas
        lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p else 0
        lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.inf
        el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else lfc_h
        fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
        
        return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
        
    except Exception as e:
        st.warning(f"Error en cálculo termodinámico: {str(e)}")
        return (
            units.Quantity(0, 'J/kg'), 
            units.Quantity(0, 'J/kg'), 
            None, 0, None, np.inf, None, 0, 0
        )

def calculate_storm_parameters(p_levels, wind_speed, wind_dir):
    """Calcula parámetros de tormenta como cisallamiento y helicidad."""
    try:
        # Convertir viento a componentes u, v
        u, v = mpcalc.wind_components(wind_speed.to('m/s'), wind_dir)
        
        # Calcular alturas
        heights = mpcalc.pressure_to_height_std(p_levels).to('meter')
        
        # Filtrar valores no válidos
        valid_mask = ~np.isnan(heights.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2: 
            return 0.0, 0.0, 0.0, 0.0
            
        p = p_levels[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]
        heights = heights[valid_mask]
        
        # Eliminar duplicados en altura
        _, unique_indices = np.unique(heights.m, return_index=True)
        if len(unique_indices) < 2: 
            return 0.0, 0.0, 0.0, 0.0
            
        p = p[unique_indices]
        u = u[unique_indices]
        v = v[unique_indices]
        heights = heights[unique_indices]
        
        # Interpolar a una resolución fija
        h_min = heights.m.min()
        h_max = min(heights.m.max(), 16000)  # Limitar a 16 km
        
        if h_max <= h_min: 
            return 0.0, 0.0, 0.0, 0.0
            
        h_interp = np.arange(h_min, h_max, 50) * units.meter
        u_i = np.interp(h_interp.m, heights.m, u.m) * units('m/s')
        v_i = np.interp(h_interp.m, heights.m, v.m) * units('m/s')
        
        # Calcular cisallamiento 0-6 km
        with integrator_lock:
            u_6, v_6 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=6000 * units.meter)
            shear_0_6 = mpcalc.wind_speed(u_6, v_6).m
            
            # Calcular cisallamiento 0-1 km
            u_1, v_1 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=1000 * units.meter)
            shear_0_1 = mpcalc.wind_speed(u_1, v_1).m
            
            # Calcular helicidad relativa a la tormenta (SRH)
            srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter, bottom=0 * units.meter)[0].m
            srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter, bottom=0 * units.meter)[0].m
        
        return shear_0_6, shear_0_1, srh_0_1, srh_0_3
        
    except Exception as e:
        st.warning(f"Error en cálculo de parámetros de tormenta: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def determine_cloud_type(cape, cin, shear_0_6, srh_0_1, lcl_h, fz_h):
    """Determina el tipo de nube predominante basado en parámetros termodinámicos."""
    if cape.m > 3000 and shear_0_6 > 20 and srh_0_1 > 150 and lcl_h < 1500:
        return "Supercèl·lula"
    elif cape.m > 1000 and shear_0_6 > 15:
        return "Cumulonimbus (Multicèl·lula)"
    elif cape.m > 500 and lcl_h > 2500:
        return "Castellanus"
    elif cape.m > 100:
        return "Cumulus Mediocris"
    elif fz_h < 1500:
        return "Hivernal"
    else:
        return "Estratificat"

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """Genera un análisis detallado del sondeo atmosférico."""
    # Calcular parámetros termodinámicos
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(
        p_levels, t_profile, td_profile
    )
    
    # Calcular parámetros de tormenta
    shear_0_6, shear_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(
        p_levels, wind_speed, wind_dir
    )
    
    # Determinar tipo de nube
    cloud_type = determine_cloud_type(cape, cin, shear_0_6, srh_0_1, lcl_h, fz_h)
    
    # Determinar tipo de precipitación
    precipitation_type = None
    sfc_temp = t_profile[0].m
    
    if fz_h < 1500 or sfc_temp < 5:
        precipitation_type = 'snow' if sfc_temp <= 0.5 else 'sleet'
    elif cape.m > 3000:
        precipitation_type = 'hail'
    elif cape.m > 500:
        precipitation_type = 'rain'
    elif "Estratificat" in cloud_type:
        precipitation_type = 'rain'
    elif lfc_p and el_p and (lfc_p.magnitude > el_p.magnitude if lfc_p and el_p else False):
        precipitation_type = 'virga'
    
    # Generar conversación de análisis
    chat_log = [("Tempestes.cat", f"Hola! He analitzat el sondeig i detecto una situació compatible amb la formació de núvols de tipus {cloud_type}.")]
    
    if cloud_type == "Hivernal":
        chat_log.extend([
            ("Jo", f"La isoterma de 0°C està molt baixa, a uns {fz_h:.0f} metres."),
            ("Tempestes.cat", "Exacte, aquest és el factor clau. Combinat amb la humitat present, afavoreix precipitacions hivernals."),
            ("Jo", f"La temperatura a la superfície és de {sfc_temp:.1f}°C. Què podem esperar?"),
        ])
        
        if sfc_temp <= 0.5:
            chat_log.append(("Tempestes.cat", "Amb temperatures negatives o properes a zero a tots els nivells, la precipitació serà de neu fins a les cotes més baixes."))
        else:
            chat_log.append(("Tempestes.cat", "Atenció. Hi ha una petita capa càlida just sobre la superfície. La neu podria fondre's en travessar-la i tornar-se a congelar en contacte amb el terra (pluja gelant) o arribar com aguanieve. És un fenomen perillós."))
            
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([
            ("Jo", f"Veig uns valors d'inestabilitat i cisallament molt alts."),
            ("Tempestes.cat", f"Correcte. Tenim un CAPE de {cape.m:.0f} J/kg, que és el combustible de la tempesta. A més, el cisallament de {shear_0_6:.1f} m/s i sobretot l'helicitat (SRH) de {srh_0_1:.1f} m²/s² a nivells baixos són ideals per a la rotació."),
            ("Jo", f"I el CIN de {cin.m:.0f} J/kg? Actua com a fre?"),
            ("Tempestes.cat", "Exactament. Aquest CIN actua com una 'tapadera' que impedeix que es formin tempestes dèbils. Si la convecció aconsegueix trencar aquesta tapadora, el desenvolupament pot ser explosiu, donant lloc a la supercèl·lula."),
            ("Jo", "Quin és el risc principal en aquest cas?"),
            ("Tempestes.cat", "El risc és molt alt. Cal esperar calamarsa gran o molt gran, ratxes de vent destructives i, amb aquests valors d'SRH, hi ha un risc significatiu de tornados."),
        ])
        
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
        chat_log.extend([
            ("Jo", f"Veig un CAPE de {cape.m:.0f} J/kg. És un valor considerable."),
            ("Tempestes.cat", "Sí, indica energia suficient per a tempestes fortes, però no tan organitzades com una supercèl·lula."),
        ])
        
        if cin.m < -100:
            chat_log.append(("Tempestes.cat", f"Tot i això, el CIN és molt fort ({cin.m:.0f} J/kg), el que podria impedir que les tempestes arribin a formar-se."))
        else:
            chat_log.append(("Jo", "I per què no s'organitzen més?"))
            chat_log.append(("Tempestes.cat", f"La clau és el cisallament del vent, de només {shear_0_6:.1f} m/s. És massa feble per induir una rotació sostinguda. Les tempestes competiran entre elles."))
            
        if lfc_h > 5000:
            chat_log.append(("Tempestes.cat", "A més, amb un LFC tan alt, parlem de convecció de base elevada. El risc principal en aquests casos són els esclafits secs."))
            
    elif "Estratificat" in cloud_type:
        # Calcular agua precipitable en los primeros 4 km
        try:
            heights = mpcalc.pressure_to_height_std(p_levels).to('m')
            layer_mask = (heights <= heights[0] + 4000 * units.meter)
            pwat_0_4 = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask]).to('mm')
        except:
            pwat_0_4 = units.Quantity(0, 'mm')
            
        chat_log.extend([
            ("Jo", "Aquí veig molta humitat però gairebé no hi ha inestabilitat."),
            ("Tempestes.cat", f"Exacte. No hi ha un motor convectiu (CAPE de només {cape.m:.0f} J/kg), però l'atmosfera està saturada en una capa molt gruixuda. Això és característic de la pluja estratiforme, sovint associada a sistemes frontals."),
            ("Jo", "La intensitat de la pluja depèn de l'aigua precipitable (PWAT), oi?"),
        ])
        
        if pwat_0_4.m > 25:
            chat_log.append(("Tempestes.cat", f"Sí. El PWAT en els primers 4 km és de {pwat_0_4.m:.1f} mm, un valor molt alt. Això es traduirà en pluges contínues i abundants, amb risc d'acumulacions importants."))
        elif pwat_0_4.m > 15:
            chat_log.append(("Tempestes.cat", f"Correcte. El PWAT en els primers 4 km és de {pwat_0_4.m:.1f} mm. És un valor considerable que alimentarà pluges moderades i persistents."))
        else:
            chat_log.append(("Tempestes.cat", f"Exactament. El PWAT és de {pwat_0_4.m:.1f} mm. És suficient per a ruixats febles i intermitents o plugims, però no s'esperen grans quantitats."))
    else:
        chat_log.extend([
            ("Jo", "Sembla un dia bastant tranquil, oi?"),
            ("Tempestes.cat", f"Sí, totalment. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),
            ("Jo", "Veurem algun núvol?"),
            ("Tempestes.cat", f"Probablement només alguns núvols de tipus {cloud_type} sense desenvolupament vertical ni risc de precipitació."),
        ])
    
    return chat_log, precipitation_type, cloud_type

def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """Genera una advertencia pública basada en el análisis del sondeo."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(
        p_levels, t_profile, td_profile
    )
    
    sfc_temp = t_profile[0].m
    
    # Advertencias por condiciones invernales
    if fz_h < 1500 or sfc_temp < 5:
        if sfc_temp <= 0.5:
            return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Precaució a la carretera.", "navy"
        else:
            p_low = p_levels[p_levels.magnitude > (p_levels.magnitude[0] - 300)]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp < 2.5:
                return "AVÍS PER PLUJA GEBRADORA / AGUANIEVE", "Risc de pluja gelant o aguanieve. Extremi les precaucions.", "dodgerblue"
    
    # Advertencias por precipitación estratiforme
    try:
        heights = mpcalc.pressure_to_height_std(p_levels).to('m')
        layer_mask = (heights <= heights[0] + 4000 * units.meter)
        
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
    
    # Advertencias por convección severa
    if cape.m >= 1000:
        shear_0_6, shear_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        
        if srh_0_1 > 150 and shear_0_6 > 15 and lcl_h < 1500:
            return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        if lfc_h > 3000:
            return "AVÍS PER TEMPESTES DE BASE ALTA", "Nuclis de base alta. Risc de ratxes de vent fortes i sobtades (downbursts).", "darkorange"
        if cape.m > 2000:
            return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa. Protegiu vehicles.", "purple"
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
    
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"

# =========================================================================
# === 3. FUNCIONES DE VISUALIZACIÓN =======================================
# =========================================================================
def create_logo_figure():
    """Crea el logo de Tempestes.cat."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    fig.patch.set_alpha(0)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Colores
    bg_color = '#F5F1E9'
    cloud_color = '#4B2A4B'
    senyera_red = '#DA121A'
    senyera_yellow = '#FCDD09'
    
    # Fondo circular
    ax.add_patch(Circle((5, 5), 5, facecolor=bg_color))
    
    # Nube
    cloud_verts = [
        (2, 5.8), (1.5, 6.8), (2.5, 7.8), (4, 8.3), 
        (6, 8.3), (7.5, 7.8), (8.5, 6.8), (8, 5.8), 
        (7, 5.3), (3, 5.3)
    ]
    ax.add_patch(Polygon(cloud_verts, facecolor=cloud_color, zorder=10))
    
    # Texto
    ax.text(5, 6.6, 'tempestes.cat', ha='center', va='center', 
            fontsize=3.3, color='white', weight='bold', 
            fontfamily='sans-serif', zorder=20)
    
    # Barras de lluvia (Senyera)
    bar_heights = [0.8, 1.0, 0.9, 0.7, 0.95, 0.85, 0.6, 0.75, 0.5]
    start_x = 3.0
    bar_width = 0.4
    rain_start_y = 5.3
    
    for i, h in enumerate(bar_heights):
        x_pos = start_x + i * bar_width
        color = senyera_red if i % 2 == 0 else senyera_yellow
        bar_height = h * 4.0
        
        # Sombra
        ax.add_patch(Rectangle(
            (x_pos + 0.05, rain_start_y - bar_height - 0.05), 
            bar_width, bar_height, 
            facecolor='black', alpha=0.3, lw=0, zorder=4
        ))
        
        # Barra principal
        ax.add_patch(Rectangle(
            (x_pos, rain_start_y - bar_height), 
            bar_width, bar_height, 
            facecolor=color, lw=0, zorder=5
        ))
    
    return fig

def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """Crea un diagrama Skew-T Log-P a partir de los datos del sondeo."""
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    
    # Configuración de ejes
    ax.set_ylim(1050, 100)
    ax.set_xlim(-50, 45)
    
    # Añadir adiabáticas
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    
    # Asegurar que Td no exceda a T
    td_profile = np.minimum(t_profile, td_profile)
    
    # Trazar perfiles
    skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
    
    # Perfil de parcela
    parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
    skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    
    # Temperatura de bulbo húmedo
    wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
    skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='Tª Bombolla Humida')
    
    # Áreas de CAPE y CIN
    skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
    skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    
    # Niveles significativos
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    xlims = ax.get_xlim()
    
    if lcl_p:
        ax.plot(xlims, [lcl_p.m, lcl_p.m], 'gray', linestyle='--', label='LCL')
    if lfc_p:
        ax.plot(xlims, [lfc_p.m, lfc_p.m], 'purple', linestyle='--', label='LFC')
    if el_p:
        ax.plot(xlims, [el_p.m, el_p.m], 'red', linestyle='--', label='EL')
    
    ax.legend()
    plt.tight_layout()
    return fig

def create_hodograph_figure(p_levels, wind_speed, wind_dir):
    """Crea un hodógrafo a partir de los datos de viento del sondeo."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=80.)
    
    # Añadir rejilla
    h.add_grid(increment=20, color='gray', linestyle='--')
    
    # Convertir viento a componentes u, v
    u, v = mpcalc.wind_components(wind_speed.to('knots'), wind_dir)
    
    # Trazar hodógrafo
    h.plot(u, v, color='blue', linewidth=2)
    
    # Calcular alturas
    heights = mpcalc.pressure_to_height_std(p_levels)
    max_h = heights.m.max()
    
    # Niveles de referencia (0, 1, 3, 6, 9 km)
    altitudes_km = np.array([0, 1, 3, 6, 9]) * units.km
    valid_altitudes_km = altitudes_km[altitudes_km.to('m').m <= max_h]
    
    if len(valid_altitudes_km) > 0:
        # Interpolar viento a altitudes específicas
        p_interp_func = interp1d(heights.m, p_levels.m, bounds_error=False, fill_value="extrapolate")
        p_points_mag = p_interp_func(valid_altitudes_km.to('m').m)
        
        interp_ws_knots = np.interp(p_points_mag, p_levels.m[::-1], wind_speed.to('knots').m[::-1])
        interp_wd_deg = np.interp(p_points_mag, p_levels.m[::-1], wind_dir.m[::-1])
        
        u_points, v_points = mpcalc.wind_components(
            interp_ws_knots * units.knots, 
            interp_wd_deg * units.degrees
        )
        
        # Marcar puntos en el hodógrafo
        for i, (u_pt, v_pt, alt) in enumerate(zip(u_points, v_points, valid_altitudes_km)):
            ax.scatter(u_pt.m, v_pt.m, color='orange', s=50, zorder=10)
            ax.text(
                u_pt.m + 4, v_pt.m + 4, f'{alt.m:.0f}', 
                fontsize=10, weight='bold', ha='center', va='center', zorder=10
            )
    
    ax.set_title("Hodògraf (nusos / km)", fontsize=10)
    return fig

def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type, cloud_type):
    """Crea una visualización artística de las nubes basada en el sondeo."""
    fig, ax = plt.subplots(figsize=(5, 8))
    
    # Configuración básica
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud (km)")
    ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5)
    ax.set_facecolor('#6495ED')
    
    # Sol
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    
    # Suelo
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle(
        (-1.5, 0), 3, ground_height_km, 
        color=ground_color, alpha=0.8, zorder=3, 
        hatch='//' if ground_color == '#228B22' else ''
    ))
    
    # Capas saturadas
    _draw_saturation_layers(ax, p_levels, t_profile, td_profile)
    
    # Calcular base y tope de nubes
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    # Dibujar nubes según el tipo
    if base_km is not None and top_km is not None:
        if "Estratificat" in cloud_type: 
            _draw_nimbostratus(ax, base_km, top_km, cloud_type)
        elif cloud_type == "Cumulonimbus (Multicèl·lula)" or cloud_type == "Supercèl·lula": 
            _draw_cumulonimbus(ax, base_km, top_km)
        elif cloud_type == "Castellanus": 
            _draw_cumulus_castellanus(ax, base_km, top_km)
        elif cloud_type == "Cumulus Mediocris": 
            _draw_cumulus_mediocris(ax, base_km, top_km)
        elif cloud_type == "Cumulus Fractus": 
            _draw_cumulus_fractus(ax, base_km, top_km - base_km)
    elif not np.any((t_profile.m - td_profile.m) <= 1.5):
        _draw_clear_sky(ax)
    
    # Dibujar precipitación si corresponde
    if precipitation_type and base_km is not None:
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
        precip_base_km = lfc_h / 1000.0 if cloud_type == "Castellanus" and lfc_h > 0 else base_km
        
        # Calcular humedad relativa sub-nube
        sub_cloud_rh_mean = 0.4
        try:
            p_base_precip = mpcalc.height_to_pressure_std(precip_base_km * units.kilometer)
            sub_cloud_mask = (p_levels >= p_base_precip) & (p_levels <= p_levels[0])
            if np.any(sub_cloud_mask):
                sub_cloud_rh_mean = np.mean(mpcalc.relative_humidity_from_dewpoint(
                    t_profile[sub_cloud_mask], td_profile[sub_cloud_mask]
                )).magnitude
        except Exception: 
            pass
            
        _draw_precipitation(ax, precip_base_km, ground_height_km, precipitation_type, sub_cloud_rh=sub_cloud_rh_mean)
    
    plt.tight_layout()
    return fig

def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active):
    """Crea una visualización de la estructura vertical de la nube con cisallamiento."""
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_shear = fig.add_subplot(gs[0, 1], sharey=ax)
    
    # Configuración básica
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10)
    ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitud (km)", xticks=[])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Configuración del panel de cisallamiento
    ax_shear.set(xlim=(-1, 1), xticks=[])
    ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): 
        spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    
    # Calcular parámetros termodinámicos
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(
        p_levels, t_profile, td_profile
    )
    
    # Calcular base y tope de nubes
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    # Si no hay estructura convectiva significativa
    if not base_km or not top_km or cape.m < 100:
        ax.text(
            0.5, 0.5, "Sense Estructura Convectiva", 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=9, color='white', bbox=dict(facecolor='darkblue', alpha=0.7)
        )
        ax_shear.axis('off')
        return fig
    
    visual_base_km = max(base_km, ground_height_km + 0.5)
    feature_label = "Base plana"
    
    try:
        # Interpolar viento con altura
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        h_km = mpcalc.pressure_to_height_std(p_levels).to('km').m
        unique_h, idx = np.unique(h_km, return_index=True)
        
        if len(unique_h) < 2: 
            return fig
            
        f_u = interp1d(unique_h, u.m[idx], bounds_error=False, fill_value='extrapolate')
        f_v = interp1d(unique_h, v.m[idx], bounds_error=False, fill_value='extrapolate')
        
        # Dibujar barbas de viento
        barb_heights = np.arange(0, min(20, h_km.max()), 1)
        ax_shear.barbs(
            np.zeros_like(barb_heights), barb_heights, 
            (f_u(barb_heights) * units('m/s')).to('knots').m, 
            (f_v(barb_heights) * units('m/s')).to('knots').m, 
            length=7, pivot='middle', color='k'
        )
        
        # Perfil de la nube
        altitudes = np.linspace(visual_base_km, top_km, num=50)
        u_at_alts = f_u(altitudes)
        horizontal_offsets = u_at_alts * 0.02
        
        # Calcular parámetros de tormenta
        shear_0_6, shear_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        shear_factor = np.clip(shear_0_6 / 35, 0.4, 2.5)
        
        # Forma de la nube basada en cisallamiento
        updraft_widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (altitudes - visual_base_km) / (top_km - visual_base_km + 0.01))) * shear_factor
        
        # Extensión del yunque (anvil) basada en vientos superiores
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
        
        # Puntos para el contorno de la nube
        r_pts = [(updraft_widths[i] + horizontal_offsets[i] + anvil_extension[i], altitudes[i]) for i in range(len(altitudes))]
        l_pts = [(-updraft_widths[i] + horizontal_offsets[i], altitudes[i]) for i in range(len(altitudes))]
        
        # Dibujar la nube
        ax.add_patch(Polygon(
            r_pts + l_pts[::-1], 
            facecolor='white', edgecolor='lightgray', alpha=0.95, zorder=10
        ))
        
        # Determinar características especiales en la base
        feature = None
        if top_km - base_km > 4.0 and cape.m > 500:
            if (srh_0_1 >= 150 and shear_0_6 > 20 and lcl_h <= 1000): 
                feature = 'tornado'
                feature_label = "Tornado"
            elif (srh_0_1 > 100 and shear_0_6 > 18 and lcl_h < 1200): 
                feature = 'funnel'
                feature_label = "Fibló (Funnel Cloud)"
            elif srh_0_3 > 250 and shear_0_6 > 20: 
                feature = 'wall_cloud'
                feature_label = "Núvol Mur (Wall Cloud)"
            elif shear_0_6 > 25: 
                feature = 'shelf_cloud'
                feature_label = "Núvol Prestatge (Shelf Cloud)"
            elif shear_0_6 > 15: 
                feature = 'base_rugosa'
                feature_label = "Base Rugosa"
        
        # Dibujar característica especial si corresponde
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_height_km)
            
    except Exception as e:
        st.warning(f"Error al dibujar estructura de nube: {str(e)}")
    
    # Etiqueta de característica
    ax.text(
        0.5, 0.02, feature_label, 
        ha='center', va='bottom', fontsize=12, color='white', 
        transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5)
    )
    
    plt.tight_layout()
    return fig

def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """Crea una visualización de radar simulada basada en el sondeo."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray')
    ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.grid(True, linestyle=':', alpha=0.3, color='white')
    
    # Calcular parámetros termodinámicos
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(
        p_levels, t_profile, td_profile
    )
    
    # Precipitación estratiforme
    try:
        heights = mpcalc.pressure_to_height_std(p_levels).to('m')
        layer_mask = (heights <= heights[0] + 4000 * units.meter)
        
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
                
                # Colores de radar
                radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900']
                radar_levels = [0, 15, 20, 25, 30, 35, 45]
                radar_cmap = ListedColormap(radar_colors)
                radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
                
                ax.contourf(x, y, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
                return fig
    except Exception:
        pass
    
    # Si no hay precipitación significativa
    if cape.m < 100:
        ax.text(0, 0, "Sense precipitació significativa", ha='center', va='center', color='white', fontsize=9)
        return fig
    
    # Precipitación convectiva
    shear_0_6, shear_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    
    # Viento medio en la capa LFC-EL
    mean_u, mean_v = (0, 0) * units('m/s')
    if lfc_p and el_p:
        p_mask = (p_levels >= el_p) & (p_levels <= lfc_p)
        if np.sum(p_mask) > 1:
            u, v = mpcalc.wind_components(wind_speed[p_mask], wind_dir[p_mask])
            mean_u, mean_v = np.mean(u), np.mean(v)
    
    # Intensidad y forma del eco de radar
    max_dbz = np.clip(20 + (cape.m / 3000) * 55, 20, 75)
    elongation = np.clip(1 + (shear_0_6 / 20), 1, 2.5)
    angle_rad = np.arctan2(mean_u.m, mean_v.m)
    
    # Generar campo de reflectividad
    x, y = np.linspace(-50, 50, 150), np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    x_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    y_rot = -xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    
    sigma_x, sigma_y = 15, 15 / elongation
    Z = max_dbz * np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
    Z += gaussian_filter(np.random.randn(150, 150), sigma=6) * (max_dbz * 0.1)
    Z = np.clip(Z, 0, 75)
    
    # Colores de radar para convección
    radar_colors = [
        '#00a0f0', '#0000ff', '#00ff00', '#008000', 
        '#ffff00', '#ff9900', '#ff0000', '#c80000', 
        '#ff00ff', '#960096'
    ]
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
    radar_cmap = ListedColormap(radar_colors)
    radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
    
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    return fig

# =========================================================================
# === 4. FUNCIONES AUXILIARES DE DIBUJO ===================================
# =========================================================================
def _get_cloud_color(y, base, top, b_min=0.6, b_max=0.95):
    """Devuelve el color de la nube basado en la altura."""
    if top <= base: 
        return (b_min,) * 3
    return (np.clip(b_min + (b_max - b_min) * ((y - base) / (top - base))**0.7, 0, 1),) * 3

def _draw_cumulonimbus(ax, base_km, top_km):
    """Dibuja un cumulonimbo con torre y yunque."""
    updraft_center_x, num_points = 0, 20
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    
    if len(tower_indices) == 0: 
        tower_indices = np.arange(len(altitudes))
        
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    widths += np.random.uniform(-0.05, 0.05, len(tower_indices))
    
    # Contorno principal de la torre
    r_pts = [(updraft_center_x + widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    l_pts = [(updraft_center_x - widths[i], tower_alts[i]) for i in range(len(tower_indices))]
    main_poly_pts = [(l_pts[0][0], l_pts[0][1])] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d8d8d8', lw=0, zorder=10))
    
    # Detalles de textura en la torre
    for _ in range(120):
        idx = random.randint(1, len(tower_alts) - 1) if len(tower_alts) > 1 else 0
        y = tower_alts[idx] + random.uniform(-0.3, 0.3)
        max_x_at_y = np.interp(y, tower_alts, widths, left=widths[0], right=widths[-1])
        x = updraft_center_x + random.uniform(-max_x_at_y, max_x_at_y)
        size = random.uniform(0.2, 0.6) * (1 + (y - base_km) / (top_km - base_km))
        brightness = np.clip(0.85 + 0.15 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
        
        ax.add_patch(Circle(
            (x, y), size, 
            facecolor=(brightness,) * 3, 
            alpha=random.uniform(0.1, 0.35), lw=0, zorder=11
        ))
    
    # Yunque
    anvil_altitudes = np.linspace(anvil_base_alt, top_km, 10)
    anvil_spread = 1.5 + random.uniform(-0.2, 0.2)
    
    for _ in range(80):
        y = random.uniform(anvil_base_alt, top_km)
        height_factor = 1 + (y - anvil_base_alt) / (top_km - anvil_base_alt)
        x = updraft_center_x + random.uniform(-anvil_spread * height_factor, anvil_spread * height_factor)
        width = random.uniform(0.5, 1.2) * height_factor
        height = random.uniform(0.05, 0.15)
        color = tuple([random.uniform(0.95, 1.0)] * 3)
        
        ax.add_patch(Ellipse(
            (x, y), width, height, 
            facecolor=color, alpha=random.uniform(0.1, 0.3), lw=0, zorder=12
        ))

def _draw_cumulus_mediocris(ax, base_km, top_km):
    """Dibuja un cumulus mediocris."""
    center_x = 0
    num_particles = 250
    cloud_height = top_km - base_km
    altitudes = np.linspace(base_km, top_km, 20)
    
    # Forma base de la nube
    base_width = 0.4 * (1 + 0.8 * np.sin(np.pi * (altitudes - base_km) / (cloud_height + 0.01)))
    noise = np.random.uniform(-0.1, 0.1, len(altitudes))
    widths = base_width + noise
    widths[0] = max(widths[0], 0.3)
    
    # Contorno principal
    r_pts = [(center_x + widths[i], altitudes[i]) for i in range(len(altitudes))]
    l_pts = [(center_x - widths[i], altitudes[i]) for i in range(len(altitudes))]
    main_poly_pts = [l_pts[0]] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d0d0d0', lw=0, zorder=10))
    
    # Detalles de textura
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
    """Dibuja cumulus castellanus con múltiples torres."""
    base_thickness = min(0.8, (top_km - base_km) * 0.25)
    
    # Base de la nube
    patches_base = []
    for _ in range(120):
        x = random.uniform(-1.7, 1.7)
        y = base_km + (random.random() ** 2) * base_thickness
        b = random.uniform(0.8, 0.9)
        
        patch = Ellipse(
            (x, y), 
            width=random.uniform(0.7, 1.6), 
            height=random.uniform(0.1, 0.25), 
            facecolor=(b, b, b), 
            alpha=random.uniform(0.1, 0.3), 
            lw=0
        )
        patches_base.append(patch)
    
    ax.add_collection(PatchCollection(patches_base, match_original=True, zorder=8))
    
    # Torres
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
            
            patch = Circle(
                (x, y), size, 
                facecolor=(brightness, brightness, brightness), 
                alpha=random.uniform(0.2, 0.5), lw=0
            )
            patches_turret.append(patch)
        
        ax.add_collection(PatchCollection(patches_turret, match_original=True, zorder=9 + i))

def _draw_nimbostratus(ax, base_km, top_km, cloud_type):
    """Dibuja un nimbostratus según su intensidad."""
    if "Intens" in cloud_type:
        color, alpha = '#808080', 0.95
    elif "Moderat" in cloud_type:
        color, alpha = '#a9a9a9', 0.9
    else:
        color, alpha = '#c0c0c0', 0.85
    
    # Capa principal
    ax.add_patch(Rectangle(
        (-1.7, base_km), 3.4, top_km - base_km, 
        facecolor=color, lw=0, zorder=8, alpha=alpha
    ))
    
    # Detalles de textura
    patches = []
    for _ in range(150):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.6, 0.75)
        
        patch = Ellipse(
            (x, y), 
            width=random.uniform(0.8, 1.5), 
            height=random.uniform(0.1, 0.3), 
            facecolor=(b, b, b), 
            alpha=random.uniform(0.2, 0.4), 
            lw=0
        )
        patches.append(patch)
    
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))

def _draw_cumulus_fractus(ax, base_km, thickness):
    """Dibuja cumulus fractus (nubes rotas)."""
    patches = [
        Ellipse(
            (random.gauss(0, 0.5), random.uniform(base_km, base_km + thickness)), 
            random.uniform(0.2, 0.4), 
            random.uniform(0.3, 0.7) * random.uniform(0.2, 0.4), 
            angle=random.uniform(-25, 25), 
            facecolor=_get_cloud_color(
                random.uniform(base_km, base_km + thickness), 
                base_km, base_km + thickness, b_min=0.6, b_max=0.8
            ), 
            alpha=0.5, lw=0
        ) 
        for _ in range(150)
    ]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))

def _draw_clear_sky(ax):
    """Dibuja cielo despejado con algunos cirros."""
    patches = [
        Ellipse(
            (random.uniform(-1.5, 1.5), random.uniform(10, 14)), 
            random.uniform(0.5, 1.0), 
            random.uniform(0.1, 0.2), 
            facecolor='white', 
            alpha=random.uniform(0.05, 0.1), 
            lw=0
        ) 
        for _ in range(15)
    ]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=5))

def _draw_precipitation(ax, precip_base_km, ground_km, p_type, center_x=0.0, sub_cloud_rh=0.4):
    """Dibuja diferentes tipos de precipitación."""
    if p_type == 'virga':
        alpha = np.clip(sub_cloud_rh * 0.6, 0.15, 0.55)
        fall_percentage = sub_cloud_rh / 0.5
        fall_distance = (precip_base_km - ground_km) * fall_percentage
        end_y = precip_base_km - fall_distance
        
        if sub_cloud_rh < 0.5: 
            end_y = max(end_y, ground_km + 0.3)
        else: 
            end_y = ground_km
            
        top_width = random.uniform(0.6, 0.9)
        bottom_width = top_width * 0.5
        points = [
            (center_x - top_width / 2, precip_base_km), 
            (center_x + top_width / 2, precip_base_km), 
            (center_x + bottom_width / 2, end_y), 
            (center_x - bottom_width / 2, end_y)
        ]
        
        ax.add_patch(Polygon(
            points, 
            facecolor='cornflowerblue', 
            alpha=alpha, lw=0, zorder=7
        ))
        
    elif p_type in ['rain', 'sleet']:
        color = 'mediumpurple' if p_type == 'sleet' else 'cornflowerblue'
        width = 1.6
        ax.add_patch(Rectangle(
            (center_x - width / 2, ground_km), 
            width, precip_base_km - ground_km, 
            facecolor=color, alpha=0.45, lw=0, zorder=5
        ))
        
    elif p_type == 'hail':
        ax.scatter(
            center_x + np.random.normal(0, 0.3, 150), 
            np.random.uniform(ground_km, precip_base_km, 150), 
            s=np.random.uniform(5, 40, 150), 
            c='white', alpha=0.8, marker='o', 
            edgecolor='gray', linewidth=0.5, zorder=8
        )
        
    elif p_type == 'snow':
        ax.scatter(
            center_x + np.random.normal(0, 0.5, 300), 
            np.random.uniform(ground_km, precip_base_km, 300), 
            s=np.random.uniform(20, 70, 300), 
            c='white', alpha=np.random.uniform(0.4, 0.9, 300), 
            marker='*', zorder=8
        )

def _draw_saturation_layers(ax, p_levels, t_profile, td_profile):
    """Dibuja capas saturadas en la atmósfera."""
    try:
        saturated_indices = np.where(t_profile.m - td_profile.m <= 1.5)[0]
        if not len(saturated_indices): 
            return
            
        i = 0
        while i < len(saturated_indices):
            start_idx = saturated_indices[i]
            j = i
            
            while j + 1 < len(saturated_indices) and saturated_indices[j + 1] == saturated_indices[j] + 1: 
                j += 1
                
            end_idx = saturated_indices[j]
            h_bottom = mpcalc.pressure_to_height_std(p_levels[start_idx]).to('km').m
            h_top = mpcalc.pressure_to_height_std(p_levels[end_idx]).to('km').m
            
            if h_top - h_bottom < 0.05: 
                i = j + 1
                continue
                
            patches = []
            for _ in range(int(100 + 300 * (h_top - h_bottom))):
                y = random.uniform(h_bottom, h_top)
                x = random.uniform(-1.5, 1.5)
                brightness = random.uniform(0.65, 0.85)
                
                patches.append(Ellipse(
                    (x, y), 
                    random.uniform(0.3, 0.8), 
                    random.uniform(0.05, 0.1) * (1 + h_top - h_bottom), 
                    facecolor=(brightness,) * 3, 
                    alpha=random.uniform(0.1, 0.5), lw=0
                ))
            
            ax.add_collection(PatchCollection(patches, match_original=True, zorder=7))
            i = j + 1
            
    except Exception: 
        pass

def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    """Calcula la base y tope de nubes basado en el sondeo."""
    _, _, lcl_p, lcl_h, _, _, _, el_h, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    
    if not lcl_p: 
        return None, None
        
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
                    if rh[idx] < 0.5: 
                        p_top = p_levels[idx]
                        break
                        
            cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
        except: 
            cloud_top_km = cloud_base_km
            
    return (cloud_base_km, cloud_top_km) if cloud_base_km and cloud_top_km and cloud_top_km > cloud_base_km else (None, None)

def _draw_base_feature(ax, f_type, base_x_left, base_x_right, base_y, ground_y):
    """Dibuja características especiales en la base de la nube."""
    center_x = (base_x_left + base_x_right) / 2
    width = base_x_right - base_x_left
    
    if f_type == 'wall_cloud':
        top_l = center_x - (width * 0.75 / 2)
        top_r = center_x + (width * 0.75 / 2)
        bot_l = center_x - (width * 0.55 / 2)
        bot_r = center_x + (width * 0.55 / 2)
        
        ax.add_patch(Polygon(
            [(top_l, base_y), (top_r, base_y), (bot_r, base_y - 0.35), (bot_l, base_y - 0.35)], 
            facecolor='#383838', edgecolor='#202020', lw=0.5, zorder=12
        ))
        
    elif f_type == 'funnel':
        ax.add_patch(Polygon(
            [(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, max(base_y - 0.8, ground_y + 0.5))], 
            facecolor='darkgray', alpha=0.8, zorder=12
        ))
        
    elif f_type == 'tornado':
        ax.add_patch(Polygon(
            [(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, ground_y)], 
            facecolor='#505050', zorder=12
        ))
        
        ax.add_patch(Ellipse(
            (center_x, ground_y + 0.05), 
            width=0.7, height=0.25, 
            facecolor='#654321', alpha=0.7, zorder=13
        ))
        
    elif f_type == 'shelf_cloud':
        shelf_pts = [
            (base_x_left - 0.3, base_y), 
            (base_x_right + 0.3, base_y), 
            (base_x_right, base_y - 0.2), 
            (base_x_left, base_y - 0.2)
        ]
        
        ax.add_patch(Polygon(
            shelf_pts, 
            facecolor='darkgray', edgecolor='gray', lw=0.5, zorder=12
        ))
        
    elif f_type == 'base_rugosa':
        patches = []
        for _ in range(40):
            x = center_x + random.uniform(-width / 2, width / 2)
            y = base_y - random.uniform(0.05, 0.25)
            size = random.uniform(0.1, 0.3)
            
            patches.append(Circle(
                (x, y), size, 
                facecolor='gray', 
                alpha=random.uniform(0.3, 0.6), lw=0
            ))
        
        ax.add_collection(PatchCollection(patches, match_original=True, zorder=12))

# =========================================================================
# === 5. FUNCIONES DE INTERFAZ Y MODO DE OPERACIÓN ========================
# =========================================================================
def create_welcome_figure():
    """Crea una figura de bienvenida con efectos visuales."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    def draw_lightning(start_x, start_y, end_y, segments=10, color='cyan'):
        """Dibuja un relámpago con posibles ramificaciones."""
        x = [start_x]
        y = [start_y]
        
        for i in range(segments):
            next_x = x[-1] + random.uniform(-4, 4)
            next_y = y[-1] - (start_y - end_y) / segments * random.uniform(0.5, 1.5)
            x.append(next_x)
            y.append(next_y)
            
            # Ramificaciones aleatorias
            if random.random() < 0.2:
                branch_x = [next_x]
                branch_y = [next_y]
                
                for j in range(random.randint(2, 5)):
                    branch_x.append(branch_x[-1] + random.uniform(-3, 3))
                    branch_y.append(branch_y[-1] - random.uniform(2, 4))
                
                ax.plot(branch_x, branch_y, color=color, linewidth=10, alpha=0.1, zorder=1)
                ax.plot(branch_x, branch_y, color=color, linewidth=5, alpha=0.2, zorder=2)
                ax.plot(branch_x, branch_y, color='white', linewidth=1, alpha=0.8, zorder=3)

        ax.plot(x, y, color=color, linewidth=15, alpha=0.1, zorder=1)
        ax.plot(x, y, color=color, linewidth=8, alpha=0.2, zorder=2)
        ax.plot(x, y, color='white', linewidth=1.5, alpha=0.9, zorder=3)

    # Dibujar relámpagos
    draw_lightning(random.uniform(20, 40), 70, 0, color='#8E44AD')
    draw_lightning(random.uniform(60, 80), 70, 0, color='#3498DB')
    
    plt.tight_layout(pad=0)
    return fig

def show_welcome_screen():
    """Muestra la pantalla de bienvenida de la aplicación."""
    welcome_fig = create_welcome_figure()
    buf = io.BytesIO()
    welcome_fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, facecolor=welcome_fig.get_facecolor())
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(welcome_fig)

    # Estilos CSS para la pantalla de bienvenida
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
        background-color: rgba(0, 0, 0, 0.6); 
        border-radius: 10px;
        padding: 2rem; 
        text-align: center; 
        backdrop-filter: blur(8px);
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
    
    time.sleep(2)
    st.rerun()

def apply_preset(preset_name):
    """Aplica un perfil predefinido al modo laboratorio."""
    original_data = st.session_state.sandbox_original_data
    p_levels = st.session_state.sandbox_p_levels.copy()
    
    # Inicializar perfiles con los datos originales
    t_new = original_data['t_initial'].to('degC').magnitude.copy()
    td_new = original_data['td_initial'].to('degC').magnitude.copy()
    ws_new = original_data['wind_speed_kmh'].to('m/s').magnitude.copy()
    wd_new = original_data['wind_dir_deg'].magnitude.copy()
    
    # Aplicar perfil según selección
    if preset_name == 'supercel':
        # Supercélula clásica
        t_new[0] = 28.0
        td_new[0] = 22.0
        
        # Inversión baja
        inversion_mask = (p_levels.magnitude > 800) & (p_levels.magnitude < 900)
        t_new[inversion_mask] += 3
        
        # Perfil de viento supercelular
        p_profile_points = np.array([1000, 925, 850, 700, 500, 300])
        ws_profile_points_ms = np.array([10, 15, 20, 25, 35, 50])
        wd_profile_points_deg = np.array([140, 160, 180, 210, 240, 270])
        
        ws_new = np.interp(p_levels.magnitude, p_profile_points[::-1], ws_profile_points_ms[::-1])
        wd_new = np.interp(p_levels.magnitude, p_profile_points[::-1], wd_profile_points_deg[::-1])
        
    elif preset_name == 'everest':
        # Condiciones tipo Everest
        p_levels = np.linspace(350, 100, len(p_levels)) * units.hPa
        t_new = np.linspace(-40, -60, len(p_levels))
        td_new = t_new - 20
        ws_new = np.linspace(40, 80, len(p_levels))
        wd_new[:] = 270
        
    elif preset_name == 'sahara':
        # Desierto del Sahara
        t_new[0] = 45.0
        td_new[0] = -10.0
        p_levels_hpa = p_levels.magnitude
        
        # Perfil adiabático seco
        dry_adiabatic_mask = (p_levels_hpa > 700)
        t_new[dry_adiabatic_mask] = t_new[0] - (1000 - p_levels_hpa[dry_adiabatic_mask]) * 0.1
        td_new = t_new - 40
        
    elif preset_name == 'tropical':
        # Ambiente tropical húmedo
        t_new[:] = np.linspace(30, -70, len(p_levels))
        td_new = t_new - np.linspace(5, 50, len(p_levels))
        ws_new[:] = random.uniform(5, 10)
        
    elif preset_name == 'cyclone':
        # Ciclón tropical
        t_new[:] = np.linspace(28, -60, len(p_levels))
        t_new[p_levels.magnitude < 500] += 8  # Núcleo cálido
        td_new = t_new - np.linspace(2, 30, len(p_levels))
        ws_new = np.linspace(60, 10, len(p_levels))
        wd_new[:] = 200
        
    elif preset_name == 'monsoon':
        # Monzón
        t_new[0] = 29
        td_new[0] = 26
        td_new = t_new - 3
        ws_new[:] = np.linspace(15, 25, len(p_levels))
        wd_new[:] = 225
        
    elif preset_name == 'siberian':
        # Invierno siberiano
        t_new[:] = np.linspace(-30, -70, len(p_levels))
        t_new[p_levels.magnitude > 850] += 15  # Fuerte inversión
        td_new = t_new - 15

    # Asegurar que Td no exceda a T
    td_new = np.minimum(t_new, td_new)
    
    # Actualizar estado de la sesión
    st.session_state.sandbox_p_levels = p_levels
    st.session_state.sandbox_t_profile = t_new * units.degC
    st.session_state.sandbox_td_profile = td_new * units.degC
    st.session_state.sandbox_ws = ws_new * units('m/s')
    st.session_state.sandbox_wd = wd_new * units.degrees

def run_display_logic(p, t, td, ws, wd, obs_time):
    """Lógica común para mostrar los datos del sondeo en ambos modos."""
    # Calcular parámetros clave
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, shear_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    
    # Generar análisis y advertencia
    chat_log, precipitation_type, cloud_type = generate_detailed_analysis(p, t, td, ws, wd)
    warning_title, warning_msg, warning_color = generate_public_warning(p, t, td, ws, wd)
    
    # Mostrar advertencia
    st.markdown(
        f"<div style='background-color:{warning_color}; padding:10px; border-radius:5px;'>"
        f"<h3 style='color:white; text-align:center;'>{warning_title}</h3>"
        f"<p style='color:white; text-align:center;'>{warning_msg}</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Mostrar información del sondeo
    st.markdown(f"**Hora d'observació:** {obs_time}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CAPE", f"{cape.m:.0f} J/kg", help="Energía de convección disponible")
        st.metric("CIN", f"{cin.m:.0f} J/kg", help="Inhibición convectiva")
        
    with col2:
        st.metric("Cisallament 0-6 km", f"{shear_0_6:.1f} m/s", help="Diferencia de viento entre superficie y 6 km")
        st.metric("SRH 0-1 km", f"{srh_0_1:.1f} m²/s²", help="Helicidad relativa a la tormenta en capa baja")
        
    with col3:
        st.metric("Nivell LCL", f"{lcl_h:.0f} m", help="Nivel de condensación por ascenso")
        st.metric("Nivell LFC", f"{lfc_h:.0f} m" if lfc_h != np.inf else "N/A", help="Nivel de convección libre")
    
    # Mostrar gráficos principales
    tab1, tab2, tab3, tab4 = st.tabs(["Skew-T", "Hodògraf", "Estructura del Núvol", "Radar Simulat"])
    
    with tab1:
        st.pyplot(create_skewt_figure(p, t, td, ws, wd))
        
    with tab2:
        st.pyplot(create_hodograph_figure(p, ws, wd))
        
    with tab3:
        convergence_active = st.session_state.get('convergence_active', True)
        st.pyplot(create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type, cloud_type))
        
    with tab4:
        st.pyplot(create_radar_figure(p, t, td, ws, wd))
    
    # Mostrar conversación de análisis
    st.subheader("Anàlisi Detallat")
    for speaker, message in chat_log:
        if speaker == "Tempestes.cat":
            st.markdown(f"<div style='background-color:#f0f0f0; padding:10px; border-radius:5px; margin:5px;'>"
                       f"<strong>{speaker}:</strong> {message}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#e6f3ff; padding:10px; border-radius:5px; margin:5px;'>"
                       f"<strong>{speaker}:</strong> {message}</div>", 
                       unsafe_allow_html=True)

def run_live_mode():
    """Ejecuta el modo en vivo con datos reales."""
    st.title("🛰️ Mode en Viu")
    
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Mode en Viu)")
        
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'
            st.rerun()
            
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    
    # Cargar datos de sondeo
    soundings = parse_all_soundings("sondejos_reals.txt")
    
    if not soundings:
        st.error("No s'han pogut carregar els sondejos. Assegura't que el fitxer 'sondejos_reals.txt' existeix i té el format correcte.")
        return
    
    # Selector de sondeo
    selected_sounding = st.selectbox(
        "Selecciona un sondeig", 
        options=range(len(soundings)), 
        format_func=lambda i: soundings[i]['observation_time'].split('\n')[0]
    )
    
    # Obtener datos del sondeo seleccionado
    sounding_data = soundings[selected_sounding]
    
    # Mostrar datos
    run_display_logic(
        p=sounding_data['p_levels'],
        t=sounding_data['t_initial'],
        td=sounding_data['td_initial'],
        ws=sounding_data['wind_speed_kmh'].to('m/s'),
        wd=sounding_data['wind_dir_deg'],
        obs_time=sounding_data['observation_time']
    )

def run_sandbox_mode():
    """Ejecuta el modo laboratorio con datos manipulables."""
    st.title("🧪 Laboratori de Sondejos")
    
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Laboratori)")
        
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'
            st.rerun()
            
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    
    # Inicializar datos del laboratorio si no existen
    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        
        if not soundings:
            st.error("No s'ha trobat o no s'ha pogut llegir 'sondeigproves.txt'. Aquest mode no pot funcionar.")
            return
            
        st.session_state.sandbox_original_data = soundings[0]
        st.session_state.sandbox_p_levels = st.session_state.sandbox_original_data['p_levels'].copy()
        st.session_state.sandbox_t_profile = st.session_state.sandbox_original_data['t_initial'].copy()
        st.session_state.sandbox_td_profile = st.session_state.sandbox_original_data['td_initial'].copy()
        st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True
    
    # Controles en la barra lateral
    with st.sidebar:
        if st.button("🔄 Reiniciar al perfil original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_p_levels = data['p_levels'].copy()
            st.session_state.sandbox_t_profile = data['t_initial'].copy()
            st.session_state.sandbox_td_profile = data['td_initial'].copy()
            st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
            st.rerun()
            
        st.markdown("---")
        st.subheader("Modificació Manual")
        
        # Controles para modificar el perfil
        sfc_t = st.session_state.sandbox_t_profile[0].magnitude
        new_sfc_t = st.slider(
            "🌡️ Temperatura en Superfície (°C)", 
            -40.0, 50.0, sfc_t, 0.5
        )
        
        sfc_td = st.session_state.sandbox_td_profile[0].magnitude
        new_sfc_td = st.slider(
            "💧 Punt de Rosada en Superfície (°C)", 
            -40.0, new_sfc_t, sfc_td, 0.5
        )
        
        # Aplicar cambios
        st.session_state.sandbox_t_profile[0] = new_sfc_t * units.degC
        st.session_state.sandbox_td_profile[0] = new_sfc_td * units.degC
        
        st.markdown("---")
        st.subheader("Escenaris Predefinits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tempestes Severes**")
            if st.button("🌪️ Supercèl·lula Clàssica", use_container_width=True): 
                apply_preset('supercel')
                st.rerun()
                
            st.write("**Precipitació**")
            if st.button("❄️ Nevada Severa", use_container_width=True): 
                apply_preset('neu')
                st.rerun()
                
            if st.button("💧 Aguanieve", use_container_width=True): 
                apply_preset('aguanieve')
                st.rerun()
                
            if st.button("🌧️ Pluja Estratiforme", use_container_width=True): 
                apply_preset('pluja')
                st.rerun()
                
        with col2:
            st.write("**Climes Extrems**")
            if st.button("🏔️ Cim de l'Everest", use_container_width=True): 
                apply_preset('everest')
                st.rerun()
                
            if st.button("🏜️ Desert del Sàhara", use_container_width=True): 
                apply_preset('sahara')
                st.rerun()
                
            if st.button("🌴 Clima Tropical", use_container_width=True): 
                apply_preset('tropical')
                st.rerun()
                
            if st.button("🌀 Cicló Tropical", use_container_width=True): 
                apply_preset('cyclone')
                st.rerun()
                
            if st.button("🌊 Monsó", use_container_width=True): 
                apply_preset('monsoon')
                st.rerun()
                
            if st.button("🥶 Fred Siberià", use_container_width=True): 
                apply_preset('siberian')
                st.rerun()
    
    # Mostrar datos del laboratorio
    run_display_logic(
        p=st.session_state.sandbox_p_levels,
        t=st.session_state.sandbox_t_profile,
        td=st.session_state.sandbox_td_profile,
        ws=st.session_state.sandbox_ws,
        wd=st.session_state.sandbox_wd,
        obs_time="Sondeig de Prova - Mode Laboratori"
    )

# =========================================================================
# === 6. PUNTO DE ENTRADA DE LA APLICACIÓN ================================
# =========================================================================
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")
    
    # Inicializar modo de aplicación si no existe
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    
    # Ejecutar el modo correspondiente
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
