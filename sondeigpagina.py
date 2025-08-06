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
        chat_log.extend([("Jo", f"El CAPE és de {cape.m:.0f} J/kg. És un valor considerable."),("Tempestes.cat", "Sí, indica energia suficient per a tempestes fortes, però no tan organitzades com una supercèl·lula."),("Jo", "Per què no s'organitzen més?"),("Tempestes.cat", f"La clau és el cisallament del vent, de només {shear_0_6:.1f} m/s. És massa feble per induir una rotació sostinguda. Les tempestes competiran entre elles en lloc de formar una única estructura dominant."),("Jo", "Quins fenòmens hem de vigilar?"),("Tempestes.cat", "Principalment xàfecs intensos que poden deixar calamarsa petita o moderada. Pels Castellanus, si la base del núvol és molt alta, el risc principal són els esclafits secs (downbursts).")])
    
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

# ... (Totes les funcions de dibuix _draw_... i create_...figure van aquí) ...
# ... (S'han omès per brevetat en aquesta explicació, però estan al codi final) ...

# =========================================================================
# === 4. NOVES FUNCIONS PER A L'ESTRUCTURA DE L'APP ======================
# =========================================================================
def show_welcome_screen():
    image_url = "https://i.imgur.com/rD9QN3B.jpeg"
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover; background-position: center;
        background-repeat: no-repeat; background-attachment: fixed;
    }}
    .welcome-container {{
        background-color: rgba(0, 0, 0, 0.5); border-radius: 10px;
        padding: 2rem; text-align: center; backdrop-filter: blur(5px);
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
                st.session_state.app_mode = 'live'; st.rerun()
        with col2:
            st.markdown("### 🧪 Laboratori de Sondejos")
            st.markdown("<p>Experimenta amb un sondeig de proves. Modifica paràmetres com la temperatura i la humitat o carrega escenaris predefinits per entendre com afecten el temps.</p>", unsafe_allow_html=True)
            if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
                st.session_state.app_mode = 'sandbox'; st.rerun()
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
