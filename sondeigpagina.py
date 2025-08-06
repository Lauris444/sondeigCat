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
from scipy.signal import medfilt
import threading
import base64

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
integrator_lock = threading.Lock()


# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES (Sense canvis) =========
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
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI (Adaptades de la classe original) ===
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
        u_6, v_6 = mpcalc.bulk_shear(p_levels, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
        return s_0_6, srh_0_1, srh_0_3
    except Exception as e:
        return 0.0, 0.0, 0.0

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    pwat = mpcalc.precipitable_water(p_levels, td_profile).to('mm').m
    
    precipitation_type = None
    if fz_h < 1500 or t_profile[0].m < 5:
        precipitation_type = 'snow' if t_profile[0].m <= 0.5 else 'sleet'
    elif cape.m > 3000:
        precipitation_type = 'hail'
    elif cape.m > 500:
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
    elif cloud_type == "Nimbostratus":
        chat_log.extend([
            ("Yo", "Veig molta humitat però poca inestabilitat."),
            ("Tempestes.cat", f"Exacte. El CAPE és gairebé inexistent ({cape.m:.0f} J/kg), però tenim una capa saturada de més de {top_km - base_km:.1f} km de gruix."),
            ("Yo", "Llavors, no hi haurà tempestes?"),
            ("Tempestes.cat", "No. Aquesta estructura és típica de fronts càlids o oclusos. No hi haurà aparell elèctric ni calamarsa."),
            ("Yo", "Què podem esperar?"),
            ("Tempestes.cat", "Pluja feble a moderada, però contínua i persistent durant hores. És la típica situació de dia gris i plujós.")
        ])
    else: # Cúmuls, Estrats, Cel Serè...
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
    
    heights_km = mpcalc.pressure_to_height_std(p_levels).to('km').magnitude
    rh = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile).magnitude
    mask = (heights_km < 3.5) & (rh > 0.9)
    if np.any(mask):
        saturated_layer_top = np.max(heights_km[mask])
        saturated_layer_base = np.min(heights_km[mask])
        saturated_thickness = saturated_layer_top - saturated_layer_base
        if cape.magnitude < 250 and saturated_thickness > 1.5:
             return "AVÍS PER PLUJA PERSISTENT", "Cel cobert amb pluja contínua i feble a moderada. Visibilitat reduïda.", "steelblue"

    if cape.m >= 1000:
        shear_0_6, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        
        if srh_0_1 > 150 and shear_0_6 > 15: # Ajustat el shear per a tornados
            return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        
        if lfc_h > 3000:
            return "AVÍS PER TEMPESTES DE BASE ALTA", "Nuclis de base alta. Risc de ratxes de vent fortes i sobtades (downbursts).", "darkorange"

        if cape.m > 2000:
            return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa. Protegiu vehicles.", "purple"
            
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
            
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"


# ... (La resta de funcions de dibuix i càlcul es mantenen exactament igual que a la versió anterior) ...


# =========================================================================
# === 4. LÒGICA DE L'APLICACIÓ STREAMLIT =================================
# =========================================================================

# NOU: Funció per codificar la imatge a base64
def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def load_new_sounding_data():
    filepath = st.session_state.selected_file
    soundings = parse_all_soundings(filepath)
    if not soundings:
        st.error(f"No s'han pogut carregar dades vàlides de {filepath}")
        return
    data = soundings[0]
    st.session_state.original_data = data
    reset_working_profiles()
    
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
        base_files = [
            "1am.txt", "2am.txt", "3am.txt", "4am.txt", "5am.txt", "6am.txt", 
            "7am.txt", "8am.txt", "9am.txt", "10am.txt", "11am.txt", 
            "12pm.txt", 
            "1pm.txt", "2pm.txt", "3pm.txt", "4pm.txt", "5pm.txt", "6pm.txt", 
            "7pm.txt", "8pm.txt", "9pm.txt", "10pm.txt", "11pm.txt",
            "12am.txt"
        ]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files:
            st.error("Error: No s'ha trobat cap arxiu de sondeig! Assegura't que els arxius .txt i el logo estiguin al mateix directori.")
            st.stop()
        st.session_state.selected_file = st.session_state.existing_files[0]
        st.session_state.convergence_active = True
        st.session_state.initialized = True
        load_new_sounding_data()

    with st.sidebar:
        st.title("⚙️ Controls")
        st.selectbox("Selecciona una hora (arxiu de sondeig):", options=st.session_state.existing_files, key='selected_file', on_change=load_new_sounding_data)
        st.toggle("Activar convergència (per al càlcul del núvol)", value=st.session_state.get('convergence_active', True), key='convergence_active')
        if st.button("🔄 Reiniciar Perfils"):
            reset_working_profiles()
            st.success("Perfils reiniciats als valors originals.")
        with st.expander("🔬 Modificació Avançada (experimental)"):
            sfc_temp_val = st.session_state.t_profile[0].magnitude
            new_sfc_temp = st.slider("Temperatura en Superfície (°C)", min_value=sfc_temp_val - 20, max_value=sfc_temp_val + 20, value=sfc_temp_val, step=0.5)
            if new_sfc_temp != sfc_temp_val:
                st.session_state.t_profile[0] = new_sfc_temp * units.degC

    st.title("Visor de Sondejos Atmosfèrics")

    full_obs_time = st.session_state.observation_time
    time_parts = full_obs_time.split('\n')
    cleaned_time_str = ""
    for part in time_parts:
        if 'local' in part.lower():
            cleaned_time_str = part.strip()
            break
    if not cleaned_time_str and time_parts:
        for part in time_parts:
            if part.strip():
                cleaned_time_str = part.strip()
                break
    st.markdown(f"#### {cleaned_time_str}")

    p, t, td, ws, wd = (st.session_state.p_levels, st.session_state.t_profile, st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir)
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)

    st.subheader("Diagrama Skew-T")
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    st.divider()

    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat = mpcalc.precipitable_water(p, td).to('mm')
    
    # Lògica de determinació de núvols per a la conversa
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, st.session_state.convergence_active)
    cloud_type = "Cel Serè"
    if base_km and top_km:
        cloud_thickness = top_km - base_km
        if sfc_temp.m < 5 or fz_h < 1500:
            cloud_type = "Hivernal"
        elif cape.magnitude < 250 and cloud_thickness > 1.5 and base_km < 3.0:
            cloud_type = "Nimbostratus"
        elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150:
            cloud_type = "Supercèl·lula"
        elif cape.m > 500:
             cloud_type = "Cumulonimbus (Multicèl·lula)"
             if lfc_h > 3000:
                 cloud_type = "Castellanus"
        elif cloud_thickness > 2.0:
            cloud_type = "Cumulus Mediocris"
        else:
            cloud_type = "Cumulus Fractus"

    chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])

    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_path = "input_file_0.png"
        logo_base64 = ""
        if os.path.exists(logo_path):
            logo_base64 = image_to_base64(logo_path)
            
        css_styles = f"""
        <style>
            .chat-container {{
                background-color: #f0f0f0;
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
                padding-right: 40px;
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
                color: #005A9E;
            }}
            .tempestes-cat strong {{ color: #075E54; }}
            .tempestes-cat::after {{
                content: '✔✔';
                position: absolute;
                bottom: 6px;
                right: 10px;
                font-size: 0.8em;
                color: #4FC3F7;
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
        html_chat = "<div class='online-status'>En línia</div><div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower().replace('.', '-')
            if speaker == "Tempestes.cat":
                html_chat += f"""
                    <div class="message-row">
                        <img src="data:image/png;base64,{logo_base64}" class="profile-pic">
                        <div class="message {css_class}"><strong>{speaker}</strong>{message}</div>
                    </div>
                """
            elif speaker == "Yo":
                 html_chat += f"""
                    <div class="message-row message-row-right">
                        <div class="message {css_class}"><strong>{speaker}</strong>{message}</div>
                    </div>
                """
            else: # Sistema
                 html_chat += f"<div class='message sistema'>{message}</div>"

        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg")
        param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT", f"{pwat.m:.1f} mm")
        param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A")
        param_cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A")
        param_cols[3].metric("Shear 0-6", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1", f"{srh_0_1:.1f} m²/s²")
        param_cols[1].metric("SRH 0-3", f"{srh_0_3:.1f} m²/s²")

    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, st.session_state.convergence_active, precipitation_type, lfc_h, cape)
            st.pyplot(fig_clouds, use_container_width=True)
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(p, t, td, ws, wd, st.session_state.convergence_active)
            st.pyplot(fig_structure, use_container_width=True)
            
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)

if __name__ == '__main__':
    main()
