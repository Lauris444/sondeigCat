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
from datetime import datetime

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

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

def create_wintry_mix_profile():
    p = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
    t = np.array([1.5, 3.0, 1.0, -5.0, -20.0, -45.0, -60.0]) * units.degC
    td = np.array([0.5, 1.0, -1.0, -6.0, -22.0, -48.0, -65.0]) * units.degC
    ws = np.full_like(p.magnitude, 15) * units.knots
    wd = np.full_like(p.magnitude, 180) * units.degrees
    return {'p_levels': p, 't_initial': t, 'td_initial': td, 'wind_speed_kmh': ws.to('kph'), 'wind_dir_deg': wd}

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
    """Genera l'anàlisi estàtica per al mode 'Live'."""
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
    chat_log = [("Sistema", f"Anàlisi iniciada per a un escenari de **{cloud_type}**.")]
    if cloud_type == "Hivernal":
        chat_log.extend([("Usuari", f"La isoterma 0°C està molt baixa, a {fz_h:.0f}m. Què implica?"),("Analista", "És el factor clau. Combinat amb la humitat present, afavoreix la precipitació en forma de neu o aiguaneu."),("Usuari", f"La temperatura a superfície és de {t_profile[0].m:.1f}°C."),])
        if t_profile[0].m <= 0.5:
            chat_log.append(("Analista", "Amb temperatures negatives o properes a 0°C a tots els nivells, la precipitació serà neu fins a cotes molt baixes."))
        else:
            chat_log.append(("Analista", "Atenció. Una petita capa càlida just a la superfície podria fondre parcialment la neu, donant lloc a aiguaneu o, en casos extrems, pluja gelant."))
    elif cloud_type == "Supercèl·lula":
        chat_log.extend([("Usuari", f"El CAPE és altíssim, {cape.m:.0f} J/kg. Què significa?"),("Analista", f"Això representa l'energia per a la tempesta. Un valor tan alt indica potencial per a corrents ascendents extremadament violents, capaços de sostenir calamarsa de gran mida."),("Usuari", "I el cisallament del vent? Veig valors elevats."),("Analista", f"Correcte. El cisallament de {shear_0_6:.0f} m/s i l'helicitat (SRH) de {srh_0_3:.0f} m²/s² són els ingredients que permeten que la tempesta roti, formant una supercèl·lula."),("Usuari", "Quin és el risc principal?"),("Analista", f"Molt alt. Cal esperar calamarsa gran (>4cm), ratxes de vent destructives i, amb un SRH 0-1km de {srh_0_1:.1f}, hi ha un risc significatiu de formació de tornados.")])
    elif cloud_type in ["Cumulonimbus (Multicèl·lula)", "Castellanus"]:
        chat_log.extend([("Usuari", f"El CAPE és de {cape.m:.0f} J/kg. És molt?"),("Analista", "És un valor moderat a alt. Indica energia suficient per a tempestes fortes, però no explosives."),("Usuari", "Per què no s'organitzen com una supercèl·lula?"),("Analista", f"El cisallament ({shear_0_6:.0f} m/s) és massa feble. Les tempestes competiran entre elles. Si són Castellanus, la convecció s'inicia a nivells més alts."),("Usuari", "Quins fenòmens podem esperar?"),("Analista", "Principalment xàfecs intensos i calamarsa petita a moderada. En el cas dels Castellanus, el risc principal són els esclafits secs (downbursts) si la base està molt elevada.")])
    elif "Nimbostratus" in cloud_type:
        chat_log.extend([("Usuari", "Veig molta humitat però gairebé gens d'inestabilitat (CAPE)."),("Analista", f"Exacte. No hi ha motor convectiu (CAPE de {cape.m:.0f} J/kg), però l'atmosfera està saturada en una capa molt gruixuda. És un escenari típic de pluja estratiforme, associada a fronts."),("Usuari", "La intensitat depèn de l'aigua precipitable (PWAT), oi?"),])
        if "Intens" in cloud_type:
            chat_log.append(("Analista", f"Sí. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**, un valor molt alt. Això es traduirà en pluges **contínues i abundants**, amb risc d'acumulacions importants."))
        elif "Moderat" in cloud_type:
            chat_log.append(("Analista", f"Correcte. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. Alimentarà xàfecs **moderats i persistents**."))
        else:
            chat_log.append(("Analista", f"Exactament. El PWAT a 0-4 km és de **{pwat_0_4.m:.1f} mm**. És suficient per a **ruixats febles i intermitents** o plugims."))
    else:
        chat_log.extend([("Usuari", "Sembla un dia tranquil."),("Analista", f"Efectivament. Amb un CAPE de només {cape.m:.0f} J/kg, l'atmosfera és molt estable."),("Usuari", "Veurem algun núvol?"),("Analista", f"Probablement només alguns {cloud_type} sense desenvolupament vertical ni risc de precipitació.")])
    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd):
    """Genera anàlisi dinàmica per al mode laboratori lliure."""
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    chat_log = []
    
    chat_log.append(("Analista", "Estàs en **Mode Lliure**. Fes servir les eines de la barra lateral per modificar el perfil. Jo reaccionaré als teus canvis aquí."))
    if cape.m < 100:
        chat_log.append(("Analista", "Ara mateix, l'atmosfera està molt estable (CAPE quasi nul). Perfecte per un dia de pícnic, però no per tempestes. Prova d'escalfar i humitejar les capes baixes!"))
    elif 100 <= cape.m < 1000:
        chat_log.append(("Analista", f"**Bon començament!** Has generat **{cape.m:.0f} J/kg** de CAPE. És suficient per a tempestes d'estiu normals, com les que veiem a la costa de Barcelona o València. Podem fer-ho més interessant?"))
    elif 1000 <= cape.m < 2500:
        chat_log.append(("Analista", f"**Excel·lent!** Ara tenim **{cape.m:.0f} J/kg** de CAPE. Això ja és territori de tempestes fortes. Aquest tipus d'energia es veu sovint a les valls del nord d'Itàlia, conegudes per les seves pedregades."))
    elif cape.m >= 2500:
        chat_log.append(("Analista", f"**Impressionant!** Has creat un monstre energètic amb **{cape.m:.0f} J/kg** de CAPE! Aquests són valors típics del 'Tornado Alley' a Oklahoma (EUA). Qualsevol tempesta que es formi aquí serà severa."))
    if cin.m < -200:
        chat_log.append(("Analista", "Compte, has creat una 'tapadera' (CIN) molt forta. L'aire calent ho tindrà molt difícil per pujar. És com intentar obrir una olla a pressió! Potser cal escalfar més la superfície per debilitar-la."))
    elif -200 <= cin.m < -25:
        chat_log.append(("Analista", "Has deixat una mica de CIN, i això és bo! Aquesta petita 'tapadera' permet que l'energia s'acumuli a sota durant el dia, per alliberar-se de cop més tard. Estratègia de professional!"))
    if lcl_h > 2000 and cape.m > 500:
         chat_log.append(("Analista", "La base dels núvols és bastant alta. Si es desenvolupa una tempesta, hi ha risc d'esclafits secs (downbursts), que són corrents d'aire descendents molt perillosos."))
    elif lcl_h < 1000 and cape.m > 1000:
         chat_log.append(("Analista", "Molt interessant. Amb la base dels núvols tan baixa, si aconseguim afegir cisallament del vent, augmenta molt el potencial de formació de tornados."))
    if shear_0_6 > 18 and srh_0_3 > 150 and cape.m > 1500:
         chat_log.append(("Analista", "**Ho tens tot!** Has combinat molta energia (CAPE) amb un fort cisallament i helicitat (SRH). Aquesta és la recepta de llibre per a una **supercèl·lula rotatòria**. El perill de temps sever és màxim."))
    if fz_h < 500:
        chat_log.append(("Analista", f"La isoterma de 0°C està pràcticament a terra! Has creat un escenari perfecte per a una **nevada a cotes molt baixes**. Abriga't bé!"))
    elif cape.m > 2000 and 2500 < fz_h < 4000:
        chat_log.append(("Analista", f"La isoterma de 0°C està a una alçada ideal ({fz_h/1000:.1f} km) per a la formació de **calamarsa de gran mida**. Els corrents ascendents forts (pel CAPE alt) poden mantenir la pedra creixent durant molt de temps en aquesta capa freda."))
        
    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Genera l'anàlisi del xat per a un pas específic d'un tutorial."""
    chat_log = []
    if scenario == 'neu':
        if step == 0: chat_log.append(("Analista", "Benvingut al tutorial de nevades! Hem carregat un perfil típic d'aiguaneu. Observa la 'panxa' càlida a 850 hPa. Aquest és el nostre enemic. L'objectiu és eliminar-la."))
        elif step == 1: chat_log.append(("Analista", "**Perfecte!** Has refredat la capa mitjana. Si miressis el gràfic Skew-T, veuries que la línia vermella ja no creua la línia de 0°C en aquest nivell. Els flocs de neu ja no es fondran aquí."))
        elif step == 2: chat_log.append(("Analista", "**Molt bé!** Ara la superfície també és prou freda per mantenir la neu. Hem construït una columna d'aire freda de dalt a baix. Només queda un pas."))
        elif step == 3: chat_log.append(("Analista", "**Genial!** Has saturat l'aire a la superfície. Amb una atmosfera freda i humida, tenim tots els ingredients per a una bona nevada. Has completat la missió!"))
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem el tutorial de supercèl·lula. El primer pas és sempre crear energia. Necessitem un dia càlid d'estiu. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "**Correcte!** Molta calor. Ara, afegim el combustible: la humitat. A l'anàlisi final veuràs com augmenta el valor de CAPE quan les línies de temperatura i punt de rosada s'acosten."))
        elif step == 2: chat_log.append(("Analista", "**Fantàstic!** Has afegit cisallament. Aquest és l'ingredient secret que fa que les tempestes rotin. Ara tenim energia, humitat i rotació: la recepta perfecta!"))
        elif step == 3: chat_log.append(("Analista", "**Missió complerta!** Has creat un perfil amb molta energia (CAPE alt), humitat i cisallament. Aquest és un entorn perfecte per a tempestes severes."))

    return chat_log, None
    
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    if fz_h < 1500 or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            try:
                p_arr, t_arr = p_levels.m, t_profile.m
                warm_layer_mask = (p_arr < 950) & (p_arr > 600) & (t_arr > 0)
                if np.any(warm_layer_mask):
                    return "AVÍS PER AIGUANEU O PLUJA GEBRADORA", "Capa càlida en altura pot fondre la neu. Risc d'aiguaneu o pluja gelant.", "mediumorchid"
                else:
                    return "AVÍS PER NEU", "Perfil atmosfèric favorable a nevades a cotes baixes.", "navy"
            except:
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
# === 3. FUNCIONS DE DIBUIX (COMPLETES I RESTAURADES) =====================
# =========================================================================

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
# === 4. ESTRUCTURA DE L'APLICACIÓ =======================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">Analitzador Avançat de Sondejos Atmosfèrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Una eina per a la visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️ Mode Temps Real</h3><p>Visualitza els sondejos atmosfèrics més recents basats en dades de models. Navega entre les diferents execucions horàries disponibles.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode Temps Real", use_container_width=True):
            st.session_state.app_mode = 'live'; st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪 Laboratori de Sondejos</h3><p>Aprèn de forma interactiva com es formen els fenòmens severs modificant pas a pas un sondeig o experimenta lliurement amb els controls.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'; st.rerun()

def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False):
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

    if is_sandbox_mode:
         chat_log, precipitation_type = generate_dynamic_analysis(p, t, td, ws, wd)
    else: # Mode Live
        chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Assistent d'Anàlisi", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])
    with tab1:
        css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>"""
        html_chat = "<div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower()
            html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
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
        precipitation_type_visual = "snow" if "NEU" in title else "sleet" if "AIGUANEU" in title else precipitation_type
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type_visual, lfc_h, cape, base_km, top_km, cloud_type)
            st.pyplot(fig_clouds, use_container_width=True)
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(p, t, td, ws, wd, convergence_active)
            st.pyplot(fig_structure, use_container_width=True)
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)

def run_live_mode():
    st.title("🛰️ Mode Temps Real: Sondejos de Model")
    with st.sidebar:
        st.header("Controls")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active', help="Simula l'efecte de la convergència a nivells baixos.")
    if 'live_initialized' not in st.session_state:
        base_files = ['12am.txt'] + [f'{i}am.txt' for i in range(1, 12)] + ['12pm.txt'] + [f'{i}pm.txt' for i in range(1, 12)]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files:
            st.error("No s'ha trobat cap arxiu de sondeig per al mode en viu."); return
        now = datetime.now()
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
        st.selectbox("Selecciona una hora d'execució del model:", options=st.session_state.existing_files, index=st.session_state.sounding_index, key='selectbox_widget', on_change=sync_index_from_selectbox)
    main_cols = st.columns([1, 10, 1])
    with main_cols[0]:
        if st.button('←', use_container_width=True, disabled=(st.session_state.sounding_index == 0)):
            st.session_state.sounding_index -= 1; st.rerun()
    with main_cols[2]:
        if st.button('→', use_container_width=True, disabled=(st.session_state.sounding_index >= len(st.session_state.existing_files) - 1)):
            st.session_state.sounding_index += 1; st.rerun()
    data = st.session_state.live_data
    show_full_analysis_view(p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], obs_time=data.get('observation_time', 'Hora no disponible'), is_sandbox_mode=False)

# =================================================================================
# === NOVES FUNCIONS PER AL LABORATORI-TUTORIAL ===================================
# =================================================================================

def start_tutorial(scenario_name):
    st.session_state.sandbox_mode = 'tutorial'
    st.session_state.tutorial_active = True
    st.session_state.tutorial_scenario = scenario_name
    st.session_state.tutorial_step = 0
    if scenario_name == 'neu':
        profile_data = create_wintry_mix_profile()
    else:
        profile_data = st.session_state.sandbox_original_data
    st.session_state.sandbox_p_levels = profile_data['p_levels'].copy()
    st.session_state.sandbox_t_profile = profile_data['t_initial'].copy()
    st.session_state.sandbox_td_profile = profile_data['td_initial'].copy()
    st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
    st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()

def exit_tutorial():
    """Surt del mode tutorial però MANTÉ l'estat actual del sondeig."""
    st.session_state.sandbox_mode = 'free'
    st.session_state.tutorial_active = False
    if 'tutorial_scenario' in st.session_state: del st.session_state['tutorial_scenario']
    if 'tutorial_step' in st.session_state: del st.session_state['tutorial_step']

def apply_profile_modification(action):
    """Funció centralitzada per modificar el perfil atmosfèric."""
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
    elif action == 'cool_mid': t[mid_mask] -= 2.0
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
    elif action == 'add_shear':
        heights = mpcalc.pressure_to_height_std(p * units.hPa).to('m').m
        shear_layer_mask = (heights < 6000)
        num_points = np.sum(shear_layer_mask)
        # Augmenta la velocitat del vent en 20 m/s (uns 72 km/h) en els primers 6 km
        ws[shear_layer_mask] += np.linspace(0, 20, num_points)
        # Gira el vent 60 graus en els primers 6 km (e.g., de SE a SW)
        wd[shear_layer_mask] = (wd[shear_layer_mask] + np.linspace(0, 60, num_points)) % 360
        st.session_state.sandbox_ws = ws * units('m/s')
        st.session_state.sandbox_wd = wd * units.degrees

    td = np.minimum(t, td)
    st.session_state.sandbox_t_profile = t * units.degC
    st.session_state.sandbox_td_profile = td * units.degC

def perform_tutorial_action(action_id):
    """Executa una acció dins del tutorial i actualitza el pas si és correcte."""
    apply_profile_modification(action_id)
    tutorials = get_tutorial_data()
    scenario = st.session_state.tutorial_scenario
    step_index = st.session_state.tutorial_step
    steps = tutorials[scenario]
    if step_index < len(steps) and steps[step_index]['action_id'] == action_id:
        st.session_state.tutorial_step += 1

def show_tutorial_interface():
    """Mostra la interfície minimalista del tutorial a la pantalla principal."""
    tutorials = get_tutorial_data()
    scenario = st.session_state.tutorial_scenario
    step_index = st.session_state.tutorial_step
    steps = tutorials[scenario]
    
    st.title("🧪 Laboratori de Sondejos - Mode Tutorial")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"### Tutorial: {scenario.replace('_', ' ').title()}")
            st.markdown("---")
            if step_index >= len(steps):
                st.success("🎉 **Enhorabona, has completat el tutorial!** 🎉")
                st.markdown("El sondeig que has construït ja està a punt. Fes clic a 'Finalitzar' per veure'n l'anàlisi completa.")
                if st.button("Finalitzar i Veure Resultat", use_container_width=True, type="primary"):
                    exit_tutorial()
                    st.rerun()
            else:
                current_step = steps[step_index]
                st.markdown(f"#### {current_step['title']}")
                
                with st.container(border=True):
                    st.markdown(current_step['instruction'])
                    if current_step['action_id'] != 'conceptual':
                        st.button(current_step['button_label'], 
                                  on_click=perform_tutorial_action, 
                                  args=(current_step['action_id'],), 
                                  key=f"tut_action_{step_index}",
                                  use_container_width=True, type="primary")
                    else:
                        if st.button(current_step['button_label'], use_container_width=True):
                            st.session_state.tutorial_step += 1
                            st.rerun()

                st.markdown(f"*{current_step['explanation']}*")

        with col2:
            chat_log, _ = generate_tutorial_analysis(scenario, step_index)
            css_styles = """<style>.chat-container { background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: sans-serif; height: 350px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }.message-row { display: flex; align-items: flex-start; gap: 10px; }.message-row-right { justify-content: flex-end; }.message { padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }.usuari { background-color: #dcf8c6; align-self: flex-end; }.analista { background-color: #ffffff; }.sistema { background-color: #e1f2fb; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }.message strong { display: block; margin-bottom: 3px; font-weight: bold; color: #075E54; }.usuari strong { color: #005C4B; }</style>"""
            html_chat = "<h6>Assistent d'Anàlisi</h6><div class='chat-container'>"
            for speaker, message in chat_log:
                css_class = speaker.lower()
                html_chat += f"""<div class="message-row {'message-row-right' if css_class == 'usuari' else ''}"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            html_chat += "</div>"
            st.markdown(css_styles + html_chat, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("Abandonar Tutorial", use_container_width=True):
            exit_tutorial()
            st.rerun()

def show_sandbox_selection_screen():
    st.title("🧪 Benvingut al Laboratori de Sondejos")
    st.markdown("Tria com vols començar. Pots seguir un tutorial guiat per aprendre els conceptes clau o anar directament al mode lliure per experimentar por tu mateix.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="mode-card"><h4>🌪️ Tutorial: Supercèl·lula</h4><p>Aprèn a crear un entorn amb una inestabilitat explosiva i el cisallament necessari per a les tempestes més severes i organitzades.</p></div>""", unsafe_allow_html=True)
        if st.button("Començar Tutorial de Supercèl·lula", use_container_width=True): 
            start_tutorial('supercel')
            st.rerun()
    with c2:
        st.markdown("""<div class="mode-card"><h4>❄️ Tutorial: Nevada</h4><p>Comença amb un perfil d'aiguaneu i aprèn a modificar-lo per eliminar les capes càlides i transformar la precipitació en neu.</p></div>""", unsafe_allow_html=True)
        if st.button("Començar Tutorial de Nevada", use_container_width=True): 
            start_tutorial('neu')
            st.rerun()
    with c3:
        st.markdown("""<div class="mode-card"><h4>🛠️ Mode Lliure</h4><p>Salta directament a l'acció. Tindràs el control total sobre el perfil atmosfèric des del principi per crear els teus propis escenaris.</p></div>""", unsafe_allow_html=True)
        if st.button("Anar al Mode Lliure", use_container_width=True, type="primary"):
            st.session_state.sandbox_mode = 'free'; st.rerun()
    st.markdown("---")
    if st.button("⬅️ Tornar a l'inici"):
        st.session_state.app_mode = 'welcome'; st.rerun()
        
def run_sandbox_mode():
    if 'sandbox_mode' not in st.session_state:
        st.session_state.sandbox_mode = 'selection'

    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings: st.error("No s'ha trobat 'sondeigproves.txt'."); return
        st.session_state.sandbox_original_data = soundings[0]
        data = st.session_state.sandbox_original_data
        st.session_state.sandbox_p_levels = data['p_levels'].copy()
        st.session_state.sandbox_t_profile = data['t_initial'].copy()
        st.session_state.sandbox_td_profile = data['td_initial'].copy()
        st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True

    with st.sidebar:
        st.header("Caixa d'Eines")
        if st.button("⬅️ Tornar al Menú del Laboratori", use_container_width=True):
            for key in ['sandbox_mode', 'tutorial_active', 'tutorial_scenario', 'tutorial_step']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
        st.markdown("---")
        st.subheader("Modificacions del Perfil")
        st.markdown("**Capes Baixes (> 850 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_low',), use_container_width=True); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_low',), use_container_width=True); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_low',), use_container_width=True); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_low',), use_container_width=True)
        st.markdown("**Capes Mitjanes (850-600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_mid',), use_container_width=True, key='w_mid'); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_mid',), use_container_width=True, key='c_mid'); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_mid',), use_container_width=True, key='m_mid'); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_mid',), use_container_width=True, key='d_mid')
        st.markdown("**Capes Altes (< 600 hPa)**")
        c1, c2 = st.columns(2); c1.button("☀️ Escalfar", on_click=apply_profile_modification, args=('warm_high',), use_container_width=True, key='w_h'); c2.button("❄️ Refredar", on_click=apply_profile_modification, args=('cool_high',), use_container_width=True, key='c_h'); c1.button("💧 Humitejar", on_click=apply_profile_modification, args=('moisten_high',), use_container_width=True, key='m_h'); c2.button("💨 Assecar", on_click=apply_profile_modification, args=('dry_high',), use_container_width=True, key='d_h')
        st.markdown("---"); st.subheader("Eines Globals")
        c1, c2 = st.columns(2); c1.button("🔥 Escalfar Tot", on_click=apply_profile_modification, args=('warm_all',), use_container_width=True); c2.button("🧊 Refredar Tot", on_click=apply_profile_modification, args=('cool_all',), use_container_width=True)
        c1.button("💦 Humitejar Tot", on_click=apply_profile_modification, args=('moisten_all',), use_container_width=True); c2.button("🌬️ Assecar Tot", on_click=apply_profile_modification, args=('dry_all',), use_container_width=True)
        st.button("Tapadera (Inversió)", on_click=apply_profile_modification, args=('add_inversion',), use_container_width=True)
        st.button("🌪️ Augmentar Cisallament", on_click=apply_profile_modification, args=('add_shear',), use_container_width=True)
        st.markdown("---")
        if st.button("🔄 Reiniciar Perfil Original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_p_levels = data['p_levels'].copy(); st.session_state.sandbox_t_profile = data['t_initial'].copy(); st.session_state.sandbox_td_profile = data['td_initial'].copy()
            if st.session_state.get('tutorial_active', False): exit_tutorial(); st.rerun()
            else: st.rerun()

    if st.session_state.sandbox_mode == 'selection':
        show_sandbox_selection_screen()
    elif st.session_state.sandbox_mode == 'tutorial':
        show_tutorial_interface()
    elif st.session_state.sandbox_mode == 'free':
        st.title("🧪 Laboratori de Sondejos - Mode Lliure")
        show_full_analysis_view(
            p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile, 
            td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws, 
            wd=st.session_state.sandbox_wd, obs_time="Sondeig de Prova - Mode Laboratori",
            is_sandbox_mode=True
        )

# =========================================================================
# === PUNT D'ENTRADA DE L'APLICACIÓ =======================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Analitzador de Sondejos")
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
