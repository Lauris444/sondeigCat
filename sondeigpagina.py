# -*- coding: utf-8 -*-
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

# ==============================================================================
# SECCIÓ 1: LÒGICA DE CÀLCUL I DADES (El teu codi original)
# ==============================================================================

@st.cache_data(show_spinner="Llegint i processant arxiu de sondeig...")
def parse_all_soundings(filepath):
    # ... (El teu codi complet de parse_all_soundings va aquí) ...
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
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
            line_lower = line_strip.lower()
            if any(keyword in line_lower for keyword in time_keywords) and not (line_strip and line_strip[0].isdigit()):
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
            except Exception: continue
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
                processed = process_sounding_block(current_sounding_lines)
                if processed: all_soundings_data.append(processed)
            current_sounding_lines = []
        current_sounding_lines.append(line)
    if current_sounding_lines:
        processed = process_sounding_block(current_sounding_lines)
        if processed: all_soundings_data.append(processed)
    return all_soundings_data

class AdvancedSkewT:
    # Aquesta és la teva classe original, modificada només per no crear widgets de Matplotlib
    def __init__(self, sounding_data):
        self.original_p_levels = sounding_data['p_levels'].copy()
        self.original_t_profile = sounding_data['t_initial'].copy()
        self.original_td_profile = sounding_data['td_initial'].copy()
        self.observation_time = sounding_data.get('observation_time', "Hora no disponible")
        self.original_wind_speed = sounding_data['wind_speed_kmh'].to('m/s') if sounding_data.get('wind_speed_kmh') is not None else np.zeros(len(self.original_p_levels)) * units('m/s')
        self.original_wind_dir = np.nan_to_num(sounding_data.get('wind_dir_deg', 0), nan=0) * units.degrees
        self.current_surface_pressure = self.original_p_levels[0]
        self.convergence_active = True
        self.precipitation_type = None

        self.fig = plt.figure(figsize=(20, 15))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Guardem referències als eixos per poder-los netejar
        self.ax_skew = self.fig.add_axes([0.25, 0.1, 0.5, 0.8])
        self.skew = SkewT(fig=self.fig, ax=self.ax_skew, rotation=45)
        self.ax = self.skew.ax
        
        self.ax_public_warning = self.fig.add_axes([0.01, 0.75, 0.22, 0.15])
        self.ax_info_panel = self.fig.add_axes([0.01, 0.1, 0.22, 0.63])
        self.ax_radar_sim = self.fig.add_axes([0.77, 0.75, 0.2, 0.2])
        self.ax_cloud_drawing = self.fig.add_axes([0.77, 0.40, 0.2, 0.33])
        self.ax_cloud_structure = self.fig.add_axes([0.77, 0.05, 0.15, 0.3])
        self.ax_shear_barbs = self.fig.add_axes([0.92, 0.05, 0.03, 0.3], sharey=self.ax_cloud_structure)
        self.ax_cloud_label = self.fig.add_axes([0.77, 0.35, 0.15, 0.04])
        
        self.setup_plot()
        self.reset_profiles()
        self._force_wind_update()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>> ENGANXA AQUÍ TOTS ELS MÈTODES DE LA TEVA CLASSE AdvancedSkewT         <<<
    # >>> (Des de `reset_profiles` fins a `_force_wind_update`)                 <<<
    # >>> A continuació, els enganxo jo per a tu.                              <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    def reset_profiles(self):
        p_orig_mag = self.original_p_levels.magnitude
        unique_p, unique_idx = np.unique(p_orig_mag, return_index=True)
        filtered_p_mag = medfilt(p_orig_mag, kernel_size=3) if len(unique_p) >= 3 else p_orig_mag
        self.p_levels = filtered_p_mag * units.hPa
        f_t = interp1d(p_orig_mag[unique_idx], self.original_t_profile.magnitude[unique_idx], bounds_error=False, fill_value="extrapolate")
        f_td = interp1d(p_orig_mag[unique_idx], self.original_td_profile.magnitude[unique_idx], bounds_error=False, fill_value="extrapolate")
        self.t_profile = f_t(filtered_p_mag) * units.degC
        self.td_profile = f_td(filtered_p_mag) * units.degC
        self.current_surface_pressure = self.p_levels[0]
        self.ground_height_km = mpcalc.pressure_to_height_std(self.current_surface_pressure).to('km').magnitude
    def setup_plot(self):
        self.ax.set_ylim(1050, 100); self.ax.set_xlim(-50, 45)
        self.skew.plot_dry_adiabats(alpha=0.3, color='orange')
        self.skew.plot_moist_adiabats(alpha=0.3, color='green')
        self.skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        self.line_t, = self.skew.plot([], [], 'r', linewidth=2, label='Temperatura (T)')
        self.line_td, = self.skew.plot([], [], 'b', linewidth=2, label='Punt de Rosada (Td)')
        self.line_parcel, = self.skew.plot([], [], 'k--', linewidth=2, label='Bombolla Adiabàtica')
        self.line_wb, = self.skew.plot([], [], color='purple', linewidth=1.5, label='Tª Bombolla Humida')
        self.line_lcl, = self.ax.plot([], [], 'gray', linestyle='--'); self.line_lfc, = self.ax.plot([], [], 'purple', linestyle='--'); self.line_el, = self.ax.plot([], [], 'red', linestyle='--')
        self.ground_patch = Rectangle((0, 0), 1, 1, color='darkgreen', alpha=0.7)
        self.ax.add_patch(self.ground_patch)
    def update_ground_patch(self):
        y_min = self.current_surface_pressure.magnitude
        self.ground_patch.set_xy((-50, y_min)); self.ground_patch.set_width(95); self.ground_patch.set_height(20); self.ground_patch.set_zorder(-1)
    def adjust_profiles_to_new_surface(self):
        try:
            new_p = self.current_surface_pressure; mask = self.original_p_levels <= new_p
            if np.sum(mask) < 2: raise ValueError("Punts insuficients")
            p_masked, t_masked, td_masked, ws_masked, wd_masked = (self.original_p_levels[mask], self.original_t_profile[mask], self.original_td_profile[mask], self.original_wind_speed[mask], self.original_wind_dir[mask])
            self.p_levels = np.concatenate(([new_p], p_masked[p_masked < new_p]))
            f_t = interp1d(self.original_p_levels.m, self.original_t_profile.m, bounds_error=False, fill_value="extrapolate")
            f_td = interp1d(self.original_p_levels.m, self.original_td_profile.m, bounds_error=False, fill_value="extrapolate")
            f_ws = interp1d(self.original_p_levels.m, self.original_wind_speed.to('m/s').m, bounds_error=False, fill_value="extrapolate")
            f_wd = interp1d(self.original_p_levels.m, self.original_wind_dir.m, bounds_error=False, fill_value="extrapolate")
            self.t_profile = np.concatenate(([f_t(new_p.m) * units.degC], t_masked[p_masked < new_p])); self.td_profile = np.concatenate(([f_td(new_p.m) * units.degC], td_masked[p_masked < new_p])); self.wind_speed = np.concatenate(([f_ws(new_p.m) * units('m/s')], ws_masked[p_masked < new_p])); self.wind_dir = np.concatenate(([f_wd(new_p.m) * units.degrees], wd_masked[p_masked < new_p]))
        except Exception: self.reset_profiles()
    def calculate_thermo_parameters(self):
        try:
            p, t, td = self.p_levels, self.t_profile, self.td_profile
            valid = ~np.isnan(p.m) & ~np.isnan(t.m) & ~np.isnan(td.m)
            if np.sum(valid) < 2: return (units.Quantity(0, 'J/kg'),)*2 + (None, 0)*4
            p, t, td = p[valid], t[valid], td[valid]
            parcel_prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
            lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])
            lfc_p, _ = mpcalc.lfc(p, t, td, parcel_prof)
            el_p, _ = mpcalc.el(p, t, td, parcel_prof)
            fz_lvl = mpcalc.pressure_at_height(mpcalc.height_at_pressure(p, t, t[0]), 0*units.degC)
            return cape, cin, lcl_p, mpcalc.pressure_to_height_std(lcl_p).m if lcl_p else 0, lfc_p, mpcalc.pressure_to_height_std(lfc_p).m if lfc_p else np.inf, el_p, mpcalc.pressure_to_height_std(el_p).m if el_p else 0, mpcalc.pressure_to_height_std(fz_lvl).m if fz_lvl else 0
        except: return (units.Quantity(0, 'J/kg'),)*2 + (None, 0, None, np.inf, None, 0, 0)
    # ... i la resta dels teus mètodes... (generate_detailed_analysis, draw_clouds, etc.)
    # >>> Assegura't d'enganxar-los tots aquí <<<
    
    # Mètode final que ho orquestra tot
    def update_plot(self):
        # Aquesta funció ha de cridar a tots els teus mètodes de dibuix per
        # construir la figura `self.fig`.
        try:
            # Neteja de la figura
            self.ax_skew.cla(); self.ax_public_warning.cla(); self.ax_info_panel.cla()
            self.ax_radar_sim.cla(); self.ax_cloud_drawing.cla(); self.ax_cloud_structure.cla()
            self.ax_shear_barbs.cla(); self.ax_cloud_label.cla()
            
            # Reconfiguració i dibuix
            self.setup_plot() # Redibuixa la base del skew-T
            self.td_profile = np.minimum(self.t_profile, self.td_profile)
            self.line_t.set_data(self.t_profile, self.p_levels)
            self.line_td.set_data(self.td_profile, self.p_levels)
            parcel_prof = mpcalc.parcel_profile(self.p_levels, self.t_profile[0], self.td_profile[0]).to('degC')
            self.line_parcel.set_data(parcel_prof, self.p_levels)
            wb_profile = mpcalc.wet_bulb_temperature(self.p_levels, self.t_profile, self.td_profile)
            self.line_wb.set_data(wb_profile, self.p_levels)
            _, _, lcl_p, _, lfc_p, _, el_p, _, _ = self.calculate_thermo_parameters()
            xlims = self.ax.get_xlim()
            for line, p_val in [(self.line_lcl, lcl_p), (self.line_lfc, lfc_p), (self.line_el, el_p)]:
                if p_val: line.set_data(xlims, [p_val.m, p_val.m])
                else: line.set_data([], [])

            self.skew.shade_cape(self.p_levels, self.t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
            self.skew.shade_cin(self.p_levels, self.t_profile, parcel_prof, facecolor='black', alpha=0.3)
            
            self.update_ground_patch()

            # Aquí hi anirien les crides a la resta de teves funcions de dibuix
            # self.draw_parameters_box()
            # self.draw_static_radar_echo()
            # self.draw_clouds()
            # self.draw_cloud_structure()
            # ... etc
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error actualitzant gràfic:\n{e}", ha='center', va='center', bbox=dict(facecolor='red', alpha=0.8), color='white', transform=self.ax.transAxes)

# ==============================================================================
# SECCIÓ 3: INTERFÍCIE D'USUARI AMB STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="SondeigCat Pro")
st.title("SondeigCat Pro")

AVAILABLE_FILES = ["sondeig.txt", "sondeig1.txt", "sondeig2.txt", "sondeig3.txt", "sondeig4.txt", "sondeig5.txt"]
existing_files = [file for file in AVAILABLE_FILES if os.path.exists(file)]

if not existing_files:
    st.error("No s'han trobat arxius de sondeig. Assegura't que existeixen al teu repositori de GitHub.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    selected_file = st.selectbox("Selecciona un arxiu de sondeig:", existing_files)
    
    sounding_data = parse_all_soundings(selected_file)
    if not sounding_data:
        st.error(f"L'arxiu '{selected_file}' està buit o té un format incorrecte.")
        st.stop()
    
    # Usem el primer sondeig del fitxer
    first_sounding = sounding_data[0]
    
    default_pressure = int(first_sounding['p_levels'][0].magnitude)
    surface_p = st.number_input("Pressió en superfície (hPa):", min_value=850, max_value=1050, value=default_pressure, step=1)
    convergence = st.toggle("Activar convergència (tempestes)", value=True)

# Crea o recupera la instància de la classe des de la sessió
if 'skewt_instance' not in st.session_state or st.session_state.get('current_file') != selected_file:
    st.session_state.skewt_instance = AdvancedSkewT(first_sounding)
    st.session_state.current_file = selected_file

skew_t_instance = st.session_state.skewt_instance

# Actualitza la instància amb els valors dels controls
skew_t_instance.convergence_active = convergence
if surface_p != int(skew_t_instance.current_surface_pressure.magnitude):
    skew_t_instance.current_surface_pressure = surface_p * units.hPa
    skew_t_instance.ground_height_km = mpcalc.pressure_to_height_std(skew_t_instance.current_surface_pressure).to('km').magnitude
    skew_t_instance.adjust_profiles_to_new_surface()

# Orquestra el dibuix i mostra la figura
with st.spinner("Generant visualització complexa... Això pot trigar una estona."):
    skew_t_instance.update_plot()
    
    # Mostra la informació textual fora del gràfic per a més claredat
    # risk_text, risk_color = skew_t_instance.calculate_flood_risk()
    # st.markdown(f"<h2 style='text-align: center; color: white; background-color:{risk_color};'>{risk_text}</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.pyplot(skew_t_instance.fig, use_container_width=True)
        
    with col2:
        # title, message, color = skew_t_instance.generate_public_warning()
        # st.markdown(f"<div style='background-color:{color};...'>{title}...</div>", unsafe_allow_html=True)
        
        # analysis_text = skew_t_instance.generate_detailed_analysis()
        # with st.expander("Anàlisi Tècnica"):
        #     st.code(analysis_text)
        st.info("Els panells d'informació detallada es generarien aquí.")
