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
# SECCIÓ 1: LÒGICA DE CÀLCUL I DADES (El teu codi original, quasi intacte)
# ==============================================================================

@st.cache_data(show_spinner=False)
def parse_all_soundings(filepath):
    """
    Llegeix un fitxer de text que pot contenir múltiples sondejos i els retorna
    com una llista de diccionaris. Aquesta funció s'emmagatzema a la memòria cau
    per no haver de rellegir els arxius constantment.
    """
    all_soundings_data = []
    current_sounding_lines = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'. Assegura't que l'has pujat a GitHub.")
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
        days_fr_to_ca = {'Lundi':'Dilluns','Mardi':'Dimarts','Mercredi':'Dimecres','Jeudi':'Dijous','Vendredi':'Divendres','Samedi':'Dissabte','Dimanche':'Diumenge'}
        months_fr_to_ca = {'janvier':'de gener','février':'de febrer','mars':'de març','avril':'d\'abril','mai':'de maig','juin':'de juny','juillet':'de juliol','août':'d\'agost','septembre':'de setembre','octobre':'d\'octubre','novembre':'de novembre','décembre':'de desembre'}
        general_fr_to_ca = {'Run':'Model','locale':'local','du':'del'}

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
            for fr_day, ca_day in days_fr_to_ca.items(): translated_line = translated_line.replace(fr_day, ca_day)
            for fr_month, ca_month in months_fr_to_ca.items(): translated_line = re.sub(fr_month, ca_month, translated_line, flags=re.IGNORECASE)
            for fr_word, ca_word in general_fr_to_ca.items(): translated_line = re.sub(r'\b' + fr_word + r'\b', ca_word, translated_line, flags=re.IGNORECASE)
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


class AdvancedSkewT:
    def __init__(self, sounding_data):
        self.original_p_levels = sounding_data['p_levels'].copy()
        self.original_t_profile = sounding_data['t_initial'].copy()
        self.original_td_profile = sounding_data['td_initial'].copy()
        self.observation_time = sounding_data.get('observation_time', 'Hora no disponible')
        
        if 'wind_speed_kmh' in sounding_data and sounding_data['wind_speed_kmh'] is not None:
            self.original_wind_speed = sounding_data['wind_speed_kmh'].to('m/s')
        else:
            self.original_wind_speed = np.zeros(len(self.original_p_levels)) * units('m/s')

        if 'wind_dir_deg' in sounding_data and sounding_data['wind_dir_deg'] is not None:
            self.original_wind_dir = np.nan_to_num(sounding_data['wind_dir_deg'], nan=0) * units.degrees
        else:
            self.original_wind_dir = np.zeros(len(self.original_p_levels)) * units.degrees

        self.current_surface_pressure = self.original_p_levels[0]
        self.convergence_active = True
        
        self.fig = plt.figure(figsize=(20, 15))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        self.skew = SkewT(self.fig, rotation=45)
        self.ax = self.skew.ax
        
        self.ax_radar_sim = self.fig.add_axes([0.05, 0.75, 0.2, 0.2])
        self.ax_cloud_drawing = self.fig.add_axes([0.75, 0.55, 0.2, 0.4])
        self.ax_cloud_structure = self.fig.add_axes([0.75, 0.05, 0.15, 0.4])
        self.ax_shear_barbs = self.fig.add_axes([0.90, 0.05, 0.03, 0.4], sharey=self.ax_cloud_structure)
        
        self.setup_plot()
        self.reset_profiles()
        self._force_wind_update()

    # Copia i enganxa aquí TOTS els teus mètodes de la classe AdvancedSkewT
    # (reset_profiles, setup_radar_sim, calculate_steering_wind, etc.)
    # Per brevetat, no els enganxo tots aquí, però és crucial que ho facis.
    # ... (El teu codi va aquí) ...
    # Enganxo un parell de mètodes clau per a l'exemple:
    def reset_profiles(self):
        p_orig_mag = self.original_p_levels.magnitude
        unique_p, unique_idx = np.unique(p_orig_mag, return_index=True)
        if len(unique_p) < 3: filtered_p_mag = p_orig_mag
        else: filtered_p_mag = medfilt(p_orig_mag, kernel_size=3)
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

    # I aquí anirien la resta de mètodes TEUS:
    # calculate_thermo_parameters, draw_clouds, generate_public_warning, etc...
    # ...
    # És molt IMPORTANT que els enganxis tots!

    # Finalment, el mètode que ho orquestra tot:
    def update_plot(self):
        # Aquest mètode ha de contenir tota la lògica per cridar als altres
        # mètodes de dibuix i actualitzar la figura `self.fig`.
        # Per exemple:
        try:
            self.td_profile = np.minimum(self.t_profile, self.td_profile)
            self.line_t.set_data(self.t_profile, self.p_levels)
            # ... i així successivament amb totes les teves línies i gràfics
            
            # Cridem als teus mètodes de dibuix
            # self.draw_parameters_box()
            # self.draw_static_radar_echo()
            # self.draw_clouds()
            # self.draw_cloud_structure()
            # ...
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error actualitzant:\n{e}", ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7))


# ==============================================================================
# SECCIÓ 2: INTERFÍCIE D'USUARI AMB STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="SondeigCat Pro")

st.title("SondeigCat Pro")
st.markdown("Anàlisi interactiva de sondejos atmosfèrics.")

AVAILABLE_FILES = ["sondeig.txt", "sondeig1.txt", "sondeig2.txt", "sondeig3.txt", "sondeig4.txt", "sondeig5.txt"]
existing_files = [file for file in AVAILABLE_FILES if os.path.exists(file)]

if not existing_files:
    st.error("Error: No s'ha trobat cap arxiu de sondeig. Assegura't que estiguin al repositori de GitHub.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    selected_file = st.selectbox("Selecciona un sondeig:", existing_files)
    
    all_soundings = parse_all_soundings(selected_file)
    if not all_soundings:
        st.error(f"L'arxiu '{selected_file}' no conté dades vàlides.")
        st.stop()
    sounding_data = all_soundings[0]

    @st.cache_resource(show_spinner="Inicialitzant motor gràfic...")
    def get_skewt_instance(data_key):
        return AdvancedSkewT(sounding_data)

    skew_t_instance = get_skewt_instance(selected_file)

    default_pressure = int(skew_t_instance.current_surface_pressure.magnitude)
    surface_p = st.number_input("Pressió en superfície (hPa):", 850, 1050, default_pressure, 1)
    convergence = st.toggle("Activar convergència", value=True)

skew_t_instance.convergence_active = convergence
if surface_p != default_prayer:
    skew_t_instance.current_surface_pressure = surface_p * units.hPa
    skew_t_instance.ground_height_km = mpcalc.pressure_to_height_std(skew_t_instance.current_surface_pressure).to('km').magnitude
    skew_t_instance.adjust_profiles_to_new_surface()

skew_t_instance.update_plot()

risk_text, risk_color = skew_t_instance.calculate_flood_risk()
st.markdown(f"<h2 style='text-align: center; color: white; background-color:{risk_color}; padding: 10px; border-radius: 5px;'>{risk_text}</h2>", unsafe_allow_html=True)
st.markdown(f"**Font:** `{selected_file}` | **Hora:** `{skew_t_instance.observation_time}`")

col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.pyplot(skew_t_instance.fig, use_container_width=True)

with col2:
    title, message, color = skew_t_instance.generate_public_warning()
    st.markdown(f"<div style='background-color:{color}; color:white; padding:15px; border-radius:10px;'><h3 style='margin-top:0;'>{title}</h3><p>{message}</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    analysis_text = skew_t_instance.generate_detailed_analysis()
    with st.expander("🕵️‍♂️ Xat d'Anàlisi Tècnica", expanded=True):
        st.code(analysis_text)
