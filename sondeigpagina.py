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

# Crear un bloqueig global per a l'integrador de SciPy/MetPy.
# Això evita errors de concurrència en entorns multithread com Streamlit.
integrator_lock = threading.Lock()


# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES (Sense canvis) =========
# =============================================================================
def parse_all_soundings(filepath):
    """
    Llegeix un fitxer de text que pot contenir múltiples sondejos i els retorna
    com una llista de diccionaris.
    Tradueix automàticament el text de l'hora del francès al català.
    """
    all_soundings_data = []
    current_sounding_lines = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'.")
        return []

    def clean_and_convert(text):
        """Neteja i converteix el text a float."""
        cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
        if not cleaned_text or cleaned_text == '-':
            return None
        try:
            return float(cleaned_text)
        except ValueError:
            return None

    def process_sounding_block(block_lines):
        """Processa un bloc de línies d'un sondeig i tradueix l'hora."""
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
        general_fr_to_ca = {
            'Run': 'Model', 'locale': 'local', 'du': 'del'
        }

        for line in block_lines:
            line_strip = line.strip()
            line_lower = line_strip.lower()
            
            if any(keyword in line_lower for keyword in time_keywords) and not (line_strip and line_strip[0].isdigit()):
                time_lines.append(line_strip)
                continue

            if not line_strip or line_strip.startswith('#') or 'Pression' in line_strip:
                continue

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
            for fr_day, ca_day in days_fr_to_ca.items():
                translated_line = translated_line.replace(fr_day, ca_day)
            for fr_month, ca_month in months_fr_to_ca.items():
                translated_line = re.sub(fr_month, ca_month, translated_line, flags=re.IGNORECASE)
            for fr_word, ca_word in general_fr_to_ca.items():
                translated_line = re.sub(r'\b' + fr_word + r'\b', ca_word, translated_line, flags=re.IGNORECASE)
            translated_lines.append(translated_line)
        
        observation_time = "\n".join(translated_lines) if translated_lines else "Hora no disponible"

        sorted_indices = np.argsort(p_list)[::-1]
        return {
            'p_levels': np.array(p_list)[sorted_indices] * units.hPa,
            't_initial': np.array(t_list)[sorted_indices] * units.degC,
            'td_initial': np.array(td_list)[sorted_indices] * units.degC,
            'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph,
            'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees,
            'observation_time': observation_time
        }

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
        p_i = np.interp(h_interp.m, h_u.m, p_u.m) * units.hPa
        u_6, v_6 = mpcalc.bulk_shear(p_i, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        u_1, v_1 = mpcalc.bulk_shear(p_i, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
        srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
        return s_0_6, s_0_1, srh_0_3, srh_0_1
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def calculate_flood_risk(p_levels, td_profile):
    try:
        pwat = mpcalc.precipitable_water(p_levels, td_profile).to('mm').m
        if pwat > 45: return f"RISC EXTREM D'INUNDACIONS ({pwat:.0f} mm)", "maroon"
        if pwat > 35: return f"RISC ALT D'INUNDACIONS ({pwat:.0f} mm)", "darkred"
        if pwat > 25: return f"RISC MODERAT D'INUNDACIONS ({pwat:.0f} mm)", "#DAA520"
        return f"RISC BAIX D'INUNDACIONS ({pwat:.0f} mm)", "darkgreen"
    except: return "RISC INDETERMINAT", "darkgray"

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, shear_0_1, srh_0_3, srh_0_1 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
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

    text = ""
    # ANÀLISI D'HIVERN
    if fz_h < 1500 or t_profile[0].m < 5:
        text = "--- XAT D'HIVERN ---\n"; text += f"Marc: Iso 0°C?\n> {fz_h:.0f}m. Molt baixa.\n"; text += "Laia: Llavors neu o gel.\n"; text += f"Marc: Humitat en superfície?\n> {mpcalc.relative_humidity_from_dewpoint(t_profile[0], td_profile[0]).m*100:.0f}%. Saturat.\n"
        if t_profile[0].m <= 0.5:
            text += "Laia: El perfil és 100% nival?\n> Sí. Fred a tots els nivells.\nMarc: Conclusió?\n> Nevada segura. Prepara les cadenes.\n"
        else:
            text += "Laia: Compte, veig una capa càlida.\n> Correcte, a mitja altura.\nMarc: Llavors?\n> Risc alt de pluja gelant. Molt perillós.\n"
    # ANÀLISI DE TEMPESTA SEVERA ORGANITZADA
    elif cape.m > 2000 and shear_0_6 > 15:
        text = "--- XAT DE CAÇA (SEVER) ---\n"; is_supercell = shear_0_6 > 18 and srh_0_3 > 150
        text += "Marc: Ok, dades del sondeig. A punt.\n"; text += f"Laia: CAPE?\n> {cape.m:.0f}. Extremadament potent.\n"
        text += f"Marc: CIN?\n> {cin.m:.0f}. Feble. La 'tapa' és de paper.\n"
        text += "Laia: Temps d'iniciació?\n> " + ("Explosiu. Creixerà en 15-30 min.\n" if cin.m < -80 else "Ràpid. En menys d'una hora.\n")
        text += f"\nMarc: Base del núvol? LCL?\n> {lcl_h:.0f} m. Baixa. Perfecte.\n"
        text += f"Laia: I l'LFC? On comença la festa?\n> {lfc_h/1000:.1f} km. Molt a prop. Updrafts forts.\n"
        text += f"\nMarc: Anem a la cinemàtica. Shear 0-6km?\n> {shear_0_6:.0f} m/s. Excel·lent.\n"
        text += "Laia: L'hodògraf?\n> Llarg i corbat. De manual.\n"
        text += "\nMarc: Diagnòstic final?\n"
        if is_supercell:
            text += f"> Supercèl·lula. SRH 0-3km a {srh_0_3:.0f}.\n"; text += "Laia: Avui cacem bèsties, Marc.\n"; text += "Marc: Confirmo. Prepara la càmera bona.\n"
        else:
            text += "> Multicèl·lula organitzada.\n"; text += "Laia: Ok, a buscar la cèl·lula dominant.\n"
        text += "\nMarc: Risc de tornado?\n"
        if srh_0_1 > 150 and lcl_h < 1000:
            text += f"> ALT. SRH 0-1km a {srh_0_1:.0f}. ALERTA MÀXIMA.\n"; text += "Laia: Rebut. Ulls a la base, buscant 'wall cloud'.\n"
        else:
            text += "> Baix/Moderat. Vigila 'funnels'.\n"; text += "Laia: Ok, qualsevol embut el canto.\n"
        text += f"\nMarc: Parlem de la pedra. Isoterma 0°C?\n> {fz_h/1000:.1f} km. Prou alta.\n"
        text += "Laia: Amb aquest CAPE, què implica?\n"
        if cape.m > 4000: text += "> Pedra gegant. Destructiva.\n"
        elif cape.m > 3000: text += "> Molt grossa (>4cm).\n"
        else: text += "> Severa (2-4cm).\n"
        text += f"Marc: Inundacions? PWAT?\n> {pwat:.1f}mm. Sí, risc de pluges torrencials.\n"
        text += "\nLaia: Estratègia?\n> La de sempre. Flanc sud-est.\n"; text += "Marc: Vies d'escapament clares, sempre.\n"; text += "Laia: Rebut. Comença l'espectacle.\n"
    # SECCIÓ GRANULAR PER RANGS DE CAPE
    elif cape.m >= 100:
        text = f"--- XAT DE TARDA (CAPE: {int(cape.m)}) ---\n"
        if cape.m < 500: text += f"Marc: CAPE a {cape.m:.0f}. Molt marginal.\nLaia: Llavors, pràcticament res?\nMarc: Correcte. Un 'cumulillo' i gràcies.\nLaia: Algun xàfec molt aïllat?\nMarc: Sí, virga o quatre gotes.\nLaia: Ok, no cal ni moure's.\n"
        elif cape.m < 1000: text += f"Marc: CAPE moderat-baix: {cape.m:.0f}.\nLaia: Ara ja parlem de tronades?\nMarc: Sí, les típiques de tarda.\nLaia: Poden portar alguna sorpresa?\nMarc: Ràfegues de vent sobtades en col·lapsar.\nLaia: Calamarsa?\nMarc: Petita, si de cas. L'isoterma 0°C mana.\n> Està a {fz_h/1000:.1f} km. Normaleta.\nLaia: I pluja forta? PWAT?\n> {pwat:.0f}mm. Sí, pot descarregar amb ganes.\n"
        elif cape.m < 2000: text += f"Marc: Compte, CAPE a {cape.m:.0f}.\nLaia: Entrem en territori perillós.\nMarc: Molt. Qualsevol tempesta serà potent.\nLaia: Risc principal?\nMarc: Calamarsa >2cm i 'downbursts'.\nLaia: Ok, a vigilar els nuclis de prop.\nMarc: El cim del núvol (EL) estarà a {el_h/1000:.1f}km.\nLaia: Molt alt. Molt de recorregut per créixer.\n"
        else: text += f"Marc: Laia, confirma. Veig {cape.m:.0f} de CAPE.\nLaia: Estàs de broma?\nMarc: Gens. El sondeig és explosiu.\nLaia: Això és perill de vida.\nMarc: Totalment. Avui no s'hi juga.\n"
        if shear_0_6 < 15: text += f"\nMarc: El cisallament és baix ({shear_0_6:.1f} m/s).\n> Laia: Entesos. Això limita el perill. No s'organitzarà.\n"
    # ANÀLISI DE TEMPS DE BONANÇA
    else:
        text = "--- XAT DE TEMPS (BONANÇA) ---\n"
        text += f"Laia: Tenim alguna cosa avui?\n> Marc: Negatiu. CAPE a {cape.m:.0f}.\n"
        text += "Laia: Totalment estable, doncs.\n> Marc: Sí, l'atmosfera està 'planxada'.\n"
        text += "\nLaia: Llavors, quins núvols veurem?\n"
        if not lcl_p: text += "> Res de res. Cel serè.\n"
        elif not lfc_p or lfc_h == np.inf: text += "> Humilis/fractus. Sense creixement.\n"
        elif lcl_h/1000 > 3.0: text += "> Altocumulus o Cirrus.\n"
        else: text += "> Estrats o boirina.\n"
    return text, precipitation_type

def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    sfc_temp = t_profile[0]
    
    if fz_h < 1500 or sfc_temp.m < 5:
        if sfc_temp.m <= 0.5:
            return "AVÍS PER NEU", "Es preveu nevada a cotes baixes amb acumulacions significatives. Preveu problemes de circulació i temperatures baixes.", "navy"
        else:
            p_low = p_levels[p_levels > (p_levels[0].m - 300) * units.hPa]
            if np.any(t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5:
                return "AVÍS PER PLUJA GEBRADORA", "Pluja gelant o aiguaneu que pot crear glaçades a les carreteres. Extremi les precaucions.", "dodgerblue"
            else:
                return "CEL ENNUVOLAT", "Cel tancat amb possibilitat de pluja feble o boira. Temperatures baixes.", "steelblue"
    elif cape.m >= 1000:
        shear_0_6, shear_0_1, srh_0_3, srh_0_1 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        if srh_0_1 > 150 and shear_0_1 > 15:
            return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i esteu atents a alertes.", "darkred"
        elif cape.m > 2000:
            return "AVÍS PER PEDRA", "Tempestes violentes amb pedra grossa possible. Protegiu vehicles i propietats.", "purple"
        else:
            return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
    else:
        return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"

# =========================================================================
# === 3. FUNCIONS DE DIBUIX (Adaptades per retornar figures de Matplotlib) ===
# =========================================================================

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
    center_x, num_particles = 0, 200
    altitudes = np.linspace(base_km, top_km, 15)
    widths = 0.3 * (1 + np.sin(np.pi * (altitudes - base_km) / (top_km - base_km + 0.01)))
    widths += np.random.uniform(-0.05, 0.05, 15)
    r_pts = [ (center_x + widths[i], altitudes[i]) for i in range(15) ]
    l_pts = [ (center_x - widths[i], altitudes[i]) for i in range(15) ]
    main_poly_pts = [ (l_pts[0][0], l_pts[0][1]) ] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#e0e0e0', lw=0, zorder=10))
    for _ in range(num_particles):
        idx = random.randint(1, 14)
        y = altitudes[idx] + random.uniform(-0.2, 0.2)
        max_x_at_y = np.interp(y, altitudes, widths, left=widths[0], right=widths[-1])
        x = center_x + random.uniform(-max_x_at_y, max_x_at_y)
        size = random.uniform(0.1, 0.4)
        brightness = np.clip(0.8 + 0.2 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
        ax.add_patch(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.15, 0.4), lw=0, zorder=11))

def _draw_cumulus_fractus(ax, base_km, thickness):
    patches=[Ellipse((random.gauss(0,0.5),random.uniform(base_km,base_km+thickness)),
                     random.uniform(0.2,0.4), random.uniform(0.3,0.7)*random.uniform(0.2,0.4),
                     angle=random.uniform(-25,25), 
                     facecolor=_get_cloud_color(random.uniform(base_km,base_km+thickness),base_km,base_km+thickness,b_min=0.6,b_max=0.8),
                     alpha=0.5,lw=0) for _ in range(150)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))

def _draw_clear_sky(ax):
    patches = [Ellipse((random.uniform(-1.5,1.5), random.uniform(10,14)), 
               random.uniform(0.5,1.0), random.uniform(0.1,0.2), 
               facecolor='white', alpha=random.uniform(0.05,0.1), lw=0) for _ in range(15)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=5))

def _draw_precipitation(ax, base_km, ground_km, p_type, center_x=0.0):
    if p_type == 'virga':
        end_y = max(base_km - random.uniform(1.0, 2.5), ground_km + 0.3)
        for _ in range(50):
            xs = center_x + random.uniform(-0.6, 0.6)
            ax.plot([xs, xs + random.uniform(-0.1, 0.1)],[base_km*0.95,end_y],color='lightblue',alpha=random.uniform(0.1,0.3),lw=1.2,zorder=5)
    elif p_type in ['rain', 'sleet']: 
        width = 1.6
        ax.add_patch(Rectangle((center_x-width/2,ground_km),width,base_km-ground_km,facecolor='cornflowerblue',alpha=0.25,lw=0,zorder=5))
        for _ in range(100):
            x = center_x+random.uniform(-width/2,width/2)
            ax.plot([x,x],[base_km*0.95,ground_km],color='blue',alpha=random.uniform(0.1,0.3),lw=0.8,zorder=6)
    elif p_type == 'hail':
        ax.scatter(center_x+np.random.normal(0,0.3,150),np.random.uniform(ground_km,base_km,150),
                   s=np.random.uniform(5,40,150),c='white',alpha=0.8,marker='o',edgecolor='gray',linewidth=0.5,zorder=8)
    elif p_type == 'snow':
        ax.scatter(center_x+np.random.normal(0,0.5,300),np.random.uniform(ground_km,base_km,300),
                   s=np.random.uniform(20,70,300),c='white',alpha=np.random.uniform(0.4,0.9,300),marker='*',zorder=8)

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
                patches.append(Ellipse((x,y),random.uniform(0.3,0.8),random.uniform(0.05,0.1)*(1+h_top-h_bottom),
                                 facecolor=(brightness,)*3,alpha=random.uniform(0.1,0.5),lw=0))
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
    return (cloud_base_km, cloud_top_km) if cloud_top_km > cloud_base_km else (None, None)

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

def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type):
    fig, ax = plt.subplots(figsize=(5, 8))
    
    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set(ylim=(0,16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud (km)")
    ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5)
    ax.set_facecolor('#6495ED')
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    
    ground_color = 'white' if precipitation_type == 'snow' else '#228B22'
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//' if ground_color=='#228B22' else ''))
    
    _draw_saturation_layers(ax, p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    if base_km and top_km:
        visual_base_km = max(base_km, ground_height_km+0.5)
        visual_top_km = visual_base_km + (top_km-base_km)
        cloud_depth = top_km - base_km
        
        if cloud_depth > 5.0: _draw_cumulonimbus(ax, visual_base_km, visual_top_km)
        elif cloud_depth > 2.0: _draw_cumulus_mediocris(ax, visual_base_km, visual_top_km)
        else: _draw_cumulus_fractus(ax, visual_base_km, cloud_depth)
        
        if precipitation_type:
            _draw_precipitation(ax, visual_base_km, ground_height_km, precipitation_type)
    elif not np.any((t_profile.m - td_profile.m) <= 1.5):
        _draw_clear_sky(ax)
        
    plt.tight_layout()
    return fig

def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active):
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_shear = fig.add_subplot(gs[0, 1], sharey=ax)

    ground_height_km = mpcalc.pressure_to_height_std(p_levels[0]).to('km').m
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10)
    ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, 0), 3, ground_height_km, color='darkgreen', alpha=0.7, zorder=1, hatch='//'))
    ax.set(ylim=(0, 20), xlim=(-1.5, 1.5), ylabel="Altitud (km)", xticks=[])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax_shear.set(xlim=(-1, 1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)

    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)

    if not base_km or not top_km or cape.m < 100:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva", ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='white',
                bbox=dict(facecolor='darkblue', alpha=0.7))
        ax_shear.axis('off')
        return fig

    visual_base_km = max(base_km, ground_height_km + 0.5)

    try:
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        h_km = mpcalc.pressure_to_height_std(p_levels).to('km').m
        unique_h, idx = np.unique(h_km, return_index=True)
        if len(unique_h) < 2: return fig
        
        f_u, f_v = interp1d(unique_h, u.m[idx]), interp1d(unique_h, v.m[idx])
        barb_heights = np.arange(0, min(20, h_km.max()), 1)
        ax_shear.barbs(np.zeros_like(barb_heights), barb_heights, 
                       (f_u(barb_heights) * units('m/s')).to('knots').m, 
                       (f_v(barb_heights) * units('m/s')).to('knots').m, 
                       length=7, pivot='middle', color='k')
        
        altitudes = np.linspace(visual_base_km, top_km, num=50)
        u_at_alts = f_u(altitudes)
        horizontal_offsets = u_at_alts * 0.02
        shear_0_6, *_ = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
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
        shear_0_6, shear_0_1, srh_0_3, srh_0_1 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        feature = None
        if top_km - base_km > 4.0 and cape.m > 500:
            if (srh_0_1 >= 150 and lcl_h <= 1000 and shear_0_1 >= 15): feature = 'tornado'
            elif (srh_0_1 > 100 and lcl_h < 1200 and shear_0_1 > 12): feature = 'funnel'
            elif srh_0_3 > 150 and shear_0_6 > 18 and cape.m > 1000: feature = 'wall_cloud'
            elif shear_0_1 > 8 and lcl_h < 1500: feature = 'lowering'
        
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, ground_height_km)

    except Exception as e:
        pass

    plt.tight_layout()
    return fig

def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray')
    ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.grid(True, linestyle=':', alpha=0.3, color='white')

    cape, *_ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    if cape.m < 100:
        ax.text(0, 0, "Sense precipitació convectiva", ha='center', va='center', color='white', fontsize=9)
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
    x = np.linspace(-50, 50, 150); y = np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    x_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    y_rot = -xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    sigma_x = 15; sigma_y = sigma_x / elongation
    Z = max_dbz * np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
    noise = gaussian_filter(np.random.randn(150, 150), sigma=6)
    Z += noise * (max_dbz * 0.1); Z = np.clip(Z, 0, 75)
    
    radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
    radar_cmap = ListedColormap(radar_colors)
    radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
    
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    
    return fig


# =========================================================================
# === 4. LÒGICA DE L'APLICACIÓ STREAMLIT =================================
# =========================================================================

def load_new_sounding_data():
    """Carrega les dades del fitxer seleccionat a l'estat de la sessió."""
    filepath = st.session_state.selected_file
    soundings = parse_all_soundings(filepath)
    if not soundings:
        st.error(f"No s'han pogut carregar dades vàlides de {filepath}")
        return

    data = soundings[0]
    st.session_state.original_data = data
    reset_working_profiles()
    
def reset_working_profiles():
    """Reinicia els perfils de treball a partir de les dades originals guardades."""
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
        # Llista de fitxers ordenada cronològicament des de la 1am
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
            st.error("Error: No s'ha trobat cap arxiu de sondeig! Assegura't que els arxius .txt estiguin al mateix directori.")
            st.stop()
            
        st.session_state.selected_file = st.session_state.existing_files[0]
        st.session_state.convergence_active = True
        st.session_state.initialized = True
        load_new_sounding_data()

    with st.sidebar:
        st.title("⚙️ Controls")
        
        st.selectbox(
            "Selecciona una hora (arxiu de sondeig):",
            options=st.session_state.existing_files,
            key='selected_file',
            on_change=load_new_sounding_data
        )
        st.toggle(
            "Activar convergència (per al càlcul del núvol)",
            value=st.session_state.get('convergence_active', True),
            key='convergence_active'
        )
        if st.button("🔄 Reiniciar Perfils"):
            reset_working_profiles()
            st.success("Perfils reiniciats als valors originals.")
        
        with st.expander("🔬 Modificació Avançada (experimental)"):
            sfc_temp_val = st.session_state.t_profile[0].magnitude
            new_sfc_temp = st.slider(
                "Temperatura en Superfície (°C)",
                min_value=sfc_temp_val - 20, max_value=sfc_temp_val + 20,
                value=sfc_temp_val, step=0.5
            )
            if new_sfc_temp != sfc_temp_val:
                st.session_state.t_profile[0] = new_sfc_temp * units.degC

    st.title("Visor de Sondejos Atmosfèrics")

    # --- NOU: Lògica per netejar l'hora de l'observació ---
    full_obs_time = st.session_state.observation_time
    time_parts = full_obs_time.split('\n')
    cleaned_time_str = ""
    # Busca la línia que conté "local"
    for part in time_parts:
        if 'local' in part.lower():
            cleaned_time_str = part.strip()
            break
    # Si no la troba, agafa la primera línia no buida com a alternativa
    if not cleaned_time_str and time_parts:
        for part in time_parts:
            if part.strip():
                cleaned_time_str = part.strip()
                break
    
    st.markdown(f"#### {cleaned_time_str}")

    p, t, td, ws, wd = (st.session_state.p_levels, st.session_state.t_profile, 
                       st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir)

    risk_text, risk_color = calculate_flood_risk(p, td)
    st.markdown(f'<p style="background-color:{risk_color}; color:white; font-size:20px; border-radius:7px; padding:10px; text-align:center; font-weight:bold;">{risk_text}</p>', unsafe_allow_html=True)
    
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""
    <div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color:white; text-align:center;">{title}</h3>
        <p style="color:white; text-align:center; font-size:16px;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Diagrama Skew-T")
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])

    with tab1:
        st.subheader("Anàlisi conversacional")
        analysis_text, _ = generate_detailed_analysis(p, t, td, ws, wd)
        st.text_area("Transcripció de l'anàlisi:", value=analysis_text, height=400, disabled=True)
    
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
        shear_0_6, shear_0_1, srh_0_3, srh_0_1 = calculate_storm_parameters(p, ws, wd)
        pwat = mpcalc.precipitable_water(p, td).to('mm')

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
        _, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd)
        
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, st.session_state.convergence_active, precipitation_type)
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
