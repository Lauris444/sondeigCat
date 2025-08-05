import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
import io

# ==============================================================================
# 1. FUNCIÓ DE PARSEIG DE DADES (Sense canvis)
# ==============================================================================
def parse_all_soundings(file_content):
    """
    Llegeix el contingut d'un fitxer de sondeig i el retorna com una llista de diccionaris.
    """
    all_soundings_data = []
    current_sounding_lines = []
    lines = file_content.strip().split('\n')

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


# ==============================================================================
# 2. CLASSE PRINCIPAL DE VISUALITZACIÓ (Adaptada per nou disseny)
# S'eliminen els eixos de la barra lateral esquerra per donar més espai al gràfic.
# ==============================================================================
class AdvancedSkewT:
    def __init__(self, p_levels, t_initial, td_initial, wind_speed_kmh=None, wind_dir_deg=None, observation_time="Hora no disponible"):
        # Dades inicials
        self.original_p_levels = p_levels.copy()
        self.original_t_profile = t_initial.copy()
        self.original_td_profile = td_initial.copy()
        self.original_wind_speed = wind_speed_kmh.to('m/s') if wind_speed_kmh is not None else np.zeros(len(p_levels)) * units('m/s')
        self.original_wind_dir = np.nan_to_num(wind_dir_deg, nan=0) * units.degrees if wind_dir_deg is not None else np.zeros(len(p_levels)) * units.degrees

        # Variables d'estat
        self.current_surface_pressure = self.original_p_levels[0]
        self.p_levels, self.t_profile, self.td_profile, self.wind_speed, self.wind_dir = (None,) * 5
        self.observation_time = observation_time
        self.precipitation_type = None
        self.params_text_box = None
        self.convergence_active = True
        self.wind_barbs = None 

        # Configuració de la figura i eixos (layout ajustat)
        self.fig = plt.figure(figsize=(18, 14)) # Mida ajustada
        # Ajust de subplots per donar més espai al gràfic principal
        self.fig.subplots_adjust(left=0.08, right=0.75, top=0.93, bottom=0.08)
        
        self.skew = SkewT(self.fig, rotation=45)
        self.ax = self.skew.ax
        
        # Panells gràfics (reubicats o eliminats segons el nou disseny)
        self.ax_public_warning = self.fig.add_axes([0.08, 0.08, 0.22, 0.19]) # A la cantonada inferior esquerra
        self.ax_radar_sim = self.fig.add_axes([0.32, 0.74, 0.18, 0.20])
        self.ax_cloud_drawing = self.fig.add_axes([0.77, 0.58, 0.22, 0.39])
        self.ax_cloud_structure = self.fig.add_axes([0.77, 0.15, 0.18, 0.38])
        self.ax_shear_barbs = self.fig.add_axes([0.95, 0.15, 0.03, 0.38], sharey=self.ax_cloud_structure)
        self.ax_cloud_label = self.fig.add_axes([0.77, 0.10, 0.18, 0.05])
        
        self.time_text_ax = self.fig.add_axes([0.4, 0.03, 0.3, 0.04]) # Centrat a la part inferior
        self.time_text_ax.axis('off')
        self.time_text_obj = self.time_text_ax.text(0.5, 0.5, "", fontsize=12, weight='bold', ha='center', va='center', color='darkblue')

        self.setup_radar_sim()
        self.setup_plot()
        self.load_new_data({
            'p_levels': p_levels, 't_initial': t_initial, 'td_initial': td_initial,
            'wind_speed_kmh': wind_speed_kmh, 'wind_dir_deg': wind_dir_deg,
            'observation_time': observation_time
        })

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

    def setup_radar_sim(self):
        self.ax_radar_sim.set_facecolor('darkslategray')
        self.ax_radar_sim.set_title("Eco", fontsize=10)
        self.ax_radar_sim.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
        self.ax_radar_sim.set_xlim(-50, 50)
        self.ax_radar_sim.set_ylim(-50, 50)
        self.ax_radar_sim.grid(True, linestyle=':', alpha=0.3, color='white')
        self.radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
        self.radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
        self.radar_cmap = ListedColormap(self.radar_colors)
        self.radar_norm = BoundaryNorm(self.radar_levels, self.radar_cmap.N)

    def calculate_steering_wind(self):
        try:
            _, _, lcl_p, _, lfc_p, _, el_p, _, _ = self.calculate_thermo_parameters()
            if not lfc_p or not el_p: return 0 * units('m/s'), 0 * units('m/s')
            p_mask = (self.p_levels >= el_p) & (self.p_levels <= lfc_p)
            if np.sum(p_mask) < 2: return 0 * units('m/s'), 0 * units('m/s')
            u, v = mpcalc.wind_components(self.wind_speed[p_mask], self.wind_dir[p_mask])
            return np.mean(u), np.mean(v)
        except: return 0 * units('m/s'), 0 * units('m/s')

    def draw_static_radar_echo(self):
        self.ax_radar_sim.cla()
        self.ax_radar_sim.set_facecolor('darkslategray')
        self.ax_radar_sim.set_title("Eco", fontsize=10)
        self.ax_radar_sim.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
        self.ax_radar_sim.set_xlim(-50, 50)
        self.ax_radar_sim.set_ylim(-50, 50)
        self.ax_radar_sim.grid(True, linestyle=':', alpha=0.3, color='white')
        
        cape, *_ = self.calculate_thermo_parameters()
        if cape.m < 100: 
            self.ax_radar_sim.text(0, 0, "Sense precipitació convectiva", ha='center', va='center', color='white', fontsize=9)
            return
            
        shear_0_6, *_ = self.calculate_storm_parameters()
        mean_u, mean_v = self.calculate_steering_wind()
        max_dbz = np.clip(20 + (cape.m / 3000) * 55, 20, 75)
        elongation = np.clip(1 + (shear_0_6 / 20), 1, 2.5) 
        angle_rad = np.arctan2(mean_v.m, mean_u.m)  # Corregit: canviada l'ordre dels paràmetres
        
        x = np.linspace(-50, 50, 150)
        y = np.linspace(-50, 50, 150)
        xx, yy = np.meshgrid(x, y)
        
        # Rotació correcta del sistema de coordenades
        x_rot = xx * np.cos(angle_rad) - yy * np.sin(angle_rad)
        y_rot = xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
        
        sigma_x = 15
        sigma_y = sigma_x / elongation
        Z = max_dbz * np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
        
        noise = gaussian_filter(np.random.randn(150, 150), sigma=6)
        Z += noise * (max_dbz * 0.1)
        Z = np.clip(Z, 0, 75)
        
        self.ax_radar_sim.contourf(xx, yy, Z, levels=self.radar_levels, cmap=self.radar_cmap, norm=self.radar_norm)

    def setup_plot(self):
        self.ax.set_ylim(1050, 100)
        self.ax.set_xlim(-50, 45)
        self.skew.plot_dry_adiabats(alpha=0.3, color='orange')
        self.skew.plot_moist_adiabats(alpha=0.3, color='green')
        self.skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        
        self.line_t, = self.skew.plot([], [], 'r', linewidth=2, label='Temperatura (T)')
        self.line_td, = self.skew.plot([], [], 'b', linewidth=2, label='Punt de Rosada (Td)')
        self.line_parcel, = self.skew.plot([], [], 'k--', linewidth=2, label='Bombolla Adiabàtica')
        self.line_wb, = self.skew.plot([], [], color='purple', linewidth=1.5, label='Tª Bombolla Humida')
        self.line_lcl, = self.ax.plot([], [], 'gray', linestyle='--', label='LCL')
        self.line_lfc, = self.ax.plot([], [], 'purple', linestyle='--', label='LFC')
        self.line_el, = self.ax.plot([], [], 'red', linestyle='--', label='EL')
        
        self.ground_patch = Rectangle((0, 0), 1, 1, color='darkgreen', alpha=0.7)
        self.ax.add_patch(self.ground_patch)
        self.update_ground_patch()
    
    def update_ground_patch(self):
        y_min = self.current_surface_pressure.magnitude
        self.ground_patch.set_xy((-50, y_min))
        self.ground_patch.set_width(95)
        self.ground_patch.set_height(20)
        self.ground_patch.set_zorder(-1)

    def change_surface_pressure(self, new_p_val):
        try:
            new_p = float(new_p_val) * units.hPa
            if self.original_p_levels[-1].m < new_p.m <= self.original_p_levels[0].m:
                self.current_surface_pressure = new_p
                self.ground_height_km = mpcalc.pressure_to_height_std(self.current_surface_pressure).to('km').magnitude
                self.adjust_profiles_to_new_surface()
                self.update_plot()
                self.update_ground_patch()
            else: 
                st.warning(f"La pressió ha d'estar entre {self.original_p_levels[-1].m:.0f} i {self.original_p_levels[0].m:.0f} hPa.")
        except (ValueError, TypeError) as e: 
            st.error(f"Error en el valor de pressió: {str(e)}")

    def safe_interp1d(self, x, y):
        if len(x) != len(y): 
            raise ValueError("Arrays de longitud diferent.")
        return interp1d(x, y, bounds_error=False, fill_value="extrapolate")

    def adjust_profiles_to_new_surface(self):
        try:
            new_p = self.current_surface_pressure
            mask = self.original_p_levels <= new_p
            if np.sum(mask) < 2: 
                raise ValueError("No hi ha prou punts per interpolar")
                
            p_masked = self.original_p_levels[mask]
            t_masked = self.original_t_profile[mask]
            td_masked = self.original_td_profile[mask]
            ws_masked = self.original_wind_speed[mask]
            wd_masked = self.original_wind_dir[mask]
            
            self.p_levels = np.concatenate(([new_p], p_masked[p_masked < new_p]))
            
            f_t = self.safe_interp1d(self.original_p_levels.m, self.original_t_profile.m)
            f_td = self.safe_interp1d(self.original_p_levels.m, self.original_td_profile.m)
            f_ws = self.safe_interp1d(self.original_p_levels.m, self.original_wind_speed.to('m/s').m)
            f_wd = self.safe_interp1d(self.original_p_levels.m, self.original_wind_dir.m)
            
            self.t_profile = np.concatenate(([f_t(new_p.m) * units.degC], t_masked[p_masked < new_p]))
            self.td_profile = np.concatenate(([f_td(new_p.m) * units.degC], td_masked[p_masked < new_p]))
            self.wind_speed = np.concatenate(([f_ws(new_p.m) * units('m/s')], ws_masked[p_masked < new_p]))
            self.wind_dir = np.concatenate(([f_wd(new_p.m) * units.degrees], wd_masked[p_masked < new_p]))
            
        except Exception as e: 
            st.error(f"Error ajustant els perfils: {str(e)}")
            self.reset_profiles()

    def calculate_thermo_parameters(self):
        try:
            p, t, td = self.p_levels, self.t_profile, self.td_profile
            valid_indices = ~np.isnan(p.magnitude) & ~np.isnan(t.magnitude) & ~np.isnan(td.magnitude)
            if np.sum(valid_indices) < 2: 
                raise ValueError("No hi ha prou dades.")
                
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
            except Exception: 
                fz_lvl = np.nan * units.hPa
                
            if el_p is None and cape.magnitude > 0: 
                el_p = p[-1] 
                
            lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p else 0
            lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.inf
            el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else lfc_h
            fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
            
            return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
            
        except Exception: 
            return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0
    
    def calculate_storm_parameters(self):
        try:
            p, ws, wd = self.p_levels, self.wind_speed, self.wind_dir
            u, v = mpcalc.wind_components(ws, wd)
            heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
            
            valid_mask = ~np.isnan(heights_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
            if np.sum(valid_mask) < 2: 
                return 0.0, 0.0, 0.0, 0.0
                
            p_c = p[valid_mask]
            u_c = u[valid_mask]
            v_c = v[valid_mask]
            h_c = heights_raw[valid_mask]
            
            _, unique_indices = np.unique(h_c.m, return_index=True)
            if len(unique_indices) < 2: 
                return 0.0, 0.0, 0.0, 0.0
                
            p_u = p_c[unique_indices]
            u_u = u_c[unique_indices]
            v_u = v_c[unique_indices]
            h_u = h_c[unique_indices]
            
            h_min = h_u.m.min()
            h_max = min(h_u.m.max(), 16000)
            
            if h_max <= h_min: 
                return 0.0, 0.0, 0.0, 0.0
                
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
            
        except Exception: 
            return 0.0, 0.0, 0.0, 0.0

    def calculate_flood_risk(self):
        try:
            pwat = mpcalc.precipitable_water(self.p_levels, self.td_profile).to('mm').m
            if pwat > 45: 
                return f"RISC EXTREM D'INUNDACIONS ({pwat:.0f} mm)", "maroon"
            if pwat > 35: 
                return f"RISC ALT D'INUNDACIONS ({pwat:.0f} mm)", "darkred"
            if pwat > 25: 
                return f"RISC MODERAT ({pwat:.0f} mm)", "#DAA520"
            return f"RISC BAIX ({pwat:.0f} mm)", "darkgreen"
        except: 
            return "RISC INDETERMINAT", "darkgray"

    def draw_parameters_box(self):
        if self.params_text_box: 
            self.params_text_box.remove()
            
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = self.calculate_thermo_parameters()
        pwat = mpcalc.precipitable_water(self.p_levels, self.td_profile).to('mm')
        
        lfc_t = f"{lfc_p.m:>6.0f} hPa" if lfc_p else "   N/A "
        el_t = f"{el_p.m:>6.0f} hPa" if el_p else "   N/A "
        lcl_t = f"{lcl_p.m:>6.0f} hPa" if lcl_p else "   N/A "
        
        params_text = (f"\n-------------------\nCAPE: {cape.m:>7.0f} J/kg\nCIN:  {cin.m:>7.0f} J/kg\nPWAT: {pwat.m:>7.0f} mm\nLCL:  {lcl_t}\nLFC:  {lfc_t}\nEL:   {el_t}\n0°C:  {fz_h/1000:>7.1f} km")
        
        self.params_text_box = self.ax.text(
            0.98, 0.98, params_text, 
            transform=self.ax.transAxes, 
            fontsize=10, 
            fontfamily='monospace', 
            verticalalignment='top', 
            horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    
    def generate_detailed_analysis(self):
        self.precipitation_type = None
        p, t, td = self.p_levels, self.t_profile, self.td_profile
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = self.calculate_thermo_parameters()
        shear_0_6, shear_0_1, srh_0_3, srh_0_1 = self.calculate_storm_parameters()
        pwat = mpcalc.precipitable_water(p, td).to('mm').m
        
        if fz_h < 1500 or t[0].m < 5:
            text = "--- XAT D'HIVERN ---\n"
            text += f"Marc: Iso 0°C?\n> {fz_h:.0f}m. Molt baixa.\n"
            text += "Laia: Llavors neu o gel.\n"
            text += f"Marc: Humitat en superfície?\n> {mpcalc.relative_humidity_from_dewpoint(t[0], td[0]).m*100:.0f}%. Saturat.\n"
            
            if t[0].m <= 0.5: 
                self.precipitation_type = 'snow'
                text += "Laia: El perfil és 100% nival?\n> Sí. Fred a tots els nivells.\nMarc: Conclusió?\n> Nevada segura. Prepara les cadenes.\n"
            else: 
                self.precipitation_type = 'sleet'
                text += "Laia: Compte, veig una capa càlida.\n> Correcte, a mitja altura.\nMarc: Llavors?\n> Risc alt de pluja gelant. Molt perillós.\n"
            return text
            
        elif cape.m > 2000 and shear_0_6 > 15:
            text = "--- XAT DE CAÇA (SEVER) ---\n"
            is_supercell = shear_0_6 > 18 and srh_0_3 > 150
            text += f"Marc: Ok, dades del sondeig. A punt.\nLaia: CAPE?\n> {cape.m:.0f}. Extremadament potent.\nMarc: CIN?\n> {cin.m:.0f}. Feble. La 'tapa' és de paper.\n"
            
            if cin.m < -80: 
                text += "Laia: Temps d'iniciació?\n> Explosiu. Creixerà en 15-30 min.\n"
            else: 
                text += "Laia: Temps d'iniciació?\n> Ràpid. En menys d'una hora.\n"
                
            text += f"\nMarc: Base del núvol? LCL?\n> {lcl_h:.0f} m. Baixa. Perfecte.\nLaia: I l'LFC? On comença la festa?\n> {lfc_h/1000:.1f} km. Molt a prop. Updrafts forts.\n\nMarc: Anem a la cinemàtica. Shear 0-6km?\n> {shear_0_6:.0f} m/s. Excel·lent.\nLaia: L'hodògraf?\n> Llarg i corbat. De manual.\n\nMarc: Diagnòstic final?\n"
            
            if is_supercell: 
                text += f"> Supercèl·lula. SRH 0-3km a {srh_0_3:.0f}.\nLaia: Avui cacem bèsties, Marc.\nMarc: Confirmo. Prepara la càmera bona.\n"
            else: 
                text += "> Multicèl·lula organitzada.\nLaia: Ok, a buscar la cèl·lula dominant.\n"
                
            text += "\nMarc: Risc de tornado?\n"
            
            if srh_0_1 >= 150 and lcl_h <= 1000 and shear_0_1 >= 15: 
                text += f"> ALT. SRH 0-1km a {srh_0_1:.0f}. ALERTA MÀXIMA.\nLaia: Rebut. Ulls a la base, buscant 'wall cloud'.\n"
            else: 
                text += "> Baix/Moderat. Vigila 'funnels'.\nLaia: Ok, qualsevol embut el canto.\n"
                
            text += f"\nMarc: Parlem de la pedra. Isoterma 0°C?\n> {fz_h/1000:.1f} km. Prou alta.\nLaia: Amb aquest CAPE, què implica?\n"
            
            if cape.m > 4000: 
                text += "> Pedra gegant. Destructiva.\n"
            elif cape.m > 3000: 
                text += "> Molt grossa (>4cm).\n"
            else: 
                text += "> Severa (2-4cm).\n"
                
            text += f"Marc: Inundacions? PWAT?\n> {pwat:.1f}mm. Sí, risc de pluges torrencials.\n\nLaia: Estratègia?\n> La de sempre. Flanc sud-est.\nMarc: Vies d'escapament clares, sempre.\nLaia: Rebut. Comença l'espectacle.\n"
            return text
            
        elif cape.m >= 100:
            text = f"--- XAT DE TARDA (CAPE: {int(cape.m)}) ---\n"
            
            if cape.m < 500: 
                text += f"Marc: CAPE a {cape.m:.0f}. Molt marginal.\nLaia: Llavors, pràcticament res?\nMarc: Correcte. Un 'cumulillo' i gràcies.\nLaia: Algun xàfec molt aïllat?\nMarc: Sí, virga o quatre gotes.\nLaia: Ok, no cal ni moure's.\nMarc: El CIN a {cin.m:.0f} segurament ho aguanta.\nLaia: Entesos. Dia tranquil.\nMarc: Exacte. Passem al següent avís.\nLaia: Rebut.\n"
            elif cape.m < 1000: 
                text += f"Marc: CAPE moderat-baix: {cape.m:.0f}.\nLaia: Ara ja parlem de tronades?\nMarc: Sí, les típiques de tarda.\nLaia: Poden portar alguna sorpresa?\nMarc: Ràfegues de vent sobtades en col·lapsar.\nLaia: Calamarsa?\nMarc: Petita, si de cas. L'isoterma 0°C mana.\n> Està a {fz_h/1000:.1f} km. Normaleta.\nLaia: I pluja forta? PWAT?\n> {pwat:.0f}mm. Sí, pot descarregar amb ganes.\n"
            elif cape.m < 2000: 
                text += f"Marc: Compte, CAPE a {cape.m:.0f}.\nLaia: Entrem en territori perillós.\nMarc: Molt. Qualsevol tempesta serà potent.\nLaia: Risc principal?\nMarc: Calamarsa >2cm i 'downbursts'.\nLaia: Ok, a vigilar els nuclis de prop.\nMarc: El cim del núvol (EL) estarà a {el_h/1000:.1f}km.\nLaia: Molt alt. Molt de recorregut per créixer.\nMarc: Exacte. Avui amb compte.\nLaia: Rebut.\n"
            else: 
                text += f"Marc: Laia, confirma. Veig {cape.m:.0f} de CAPE.\nLaia: Estàs de broma?\nMarc: Gens. El sondeig és explosiu.\nLaia: Això és perill de vida.\nMarc: Totalment. Avui no s'hi juega.\nLaia: Risc principal?\n> Pedra grossa destructiva.\nMarc: Qualsevol cosa que creixi serà una bomba.\nLaia: I si tinguéssim cisallament?\n> Seria un dia històric. Per sort és baix.\n"
                
            if shear_0_6 < 15: 
                text += f"\nMarc: El cisallament és baix ({shear_0_6:.1f} m/s).\n> Laia: Entesos. Això limita el perill. No s'organitzarà.\n"
            return text
            
        else:
            text = "--- XAT DE TEMPS (BONANÇA) ---\n"
            text += f"Laia: Tenim alguna cosa avui?\n> Marc: Negatiu. CAPE a {cape.m:.0f}.\nLaia: Totalment estable, doncs.\n> Marc: Sí, l'atmosfera està 'planxada'.\n\nLaia: Llavors, quins núvols veurem?\n"
            
            if not lcl_p: 
                text += "> Res de res. Cel serè.\n"
            elif not lfc_p or lfc_h == np.inf: 
                text += "> Humilis/fractus. Sense creixement.\n"
            elif lcl_h/1000 > 3.0: 
                text += "> Altocumulus o Cirrus.\n"
            else: 
                text += "> Estrats o boirina.\n"
                
            text += "\nMarc: Dia tranquil, vaja.\nLaia: Perfecte. Saps quin és el núvol més mandrós?\nMarc: No em diguis...\nLaia: L'estrat... perquè sempre està estirat!\nMarc: ...Tallo la comunicació.\n"
            return text

    def _get_cloud_color(self, y, base, top, b_min=0.6, b_max=0.95):
        if top <= base: 
            return (b_min,) * 3
        return (np.clip(b_min + (b_max-b_min)*((y-base)/(top-base))**0.7,0,1),)*3

    def _draw_cumulonimbus(self, ax, base_km, top_km):
        updraft_center_x = 0
        altitudes = np.linspace(base_km, top_km, 20)
        anvil_base_alt = top_km * 0.8
        
        tower_indices = np.where(altitudes < anvil_base_alt)[0]
        if len(tower_indices) == 0: 
            tower_indices = np.arange(len(altitudes))
            
        tower_alts = altitudes[tower_indices]
        widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)) + np.random.uniform(-0.05, 0.05, len(tower_indices))
        
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
            
        anvil_spread = 1.5 + random.uniform(-0.2, 0.2) 
        for _ in range(80):
            y = random.uniform(anvil_base_alt, top_km)
            height_factor = 1 + (y - anvil_base_alt) / (top_km - anvil_base_alt)
            x = updraft_center_x + random.uniform(-anvil_spread * height_factor, anvil_spread * height_factor)
            width = random.uniform(0.5, 1.2) * height_factor
            height = random.uniform(0.05, 0.15)
            brightness = random.uniform(0.95, 1.0)
            ax.add_patch(Ellipse((x, y), width, height, facecolor=(brightness,)*3, alpha=random.uniform(0.1, 0.3), lw=0, zorder=12))

    def _draw_cumulus_mediocris(self, ax, base_km, top_km, num_particles=200):
        center_x = 0
        altitudes = np.linspace(base_km, top_km, 15)
        widths = 0.3 * (1 + np.sin(np.pi * (altitudes - base_km) / (top_km - base_km + 0.01))) + np.random.uniform(-0.05, 0.05, 15)
        
        r_pts = [(center_x + widths[i], altitudes[i]) for i in range(15)]
        l_pts = [(center_x - widths[i], altitudes[i]) for i in range(15)]
        
        main_poly_pts = [(l_pts[0][0], l_pts[0][1])] + r_pts + l_pts[::-1]
        ax.add_patch(Polygon(main_poly_pts, facecolor='#e0e0e0', lw=0, zorder=10))
        
        for _ in range(num_particles):
            idx = random.randint(1, 14)
            y = altitudes[idx] + random.uniform(-0.2, 0.2)
            max_x_at_y = np.interp(y, altitudes, widths, left=widths[0], right=widths[-1])
            x = center_x + random.uniform(-max_x_at_y, max_x_at_y)
            size = random.uniform(0.1, 0.4)
            brightness = np.clip(0.8 + 0.2 * ((y - base_km) / (top_km - base_km)), 0.0, 1.0)
            ax.add_patch(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.15, 0.4), lw=0, zorder=11))

    def _draw_cumulus_fractus(self, ax, base_km, thickness, num=150):
        patches=[]
        for _ in range(num):
            y = random.uniform(base_km,base_km+thickness)
            patches.append(Ellipse(
                (random.gauss(0,0.5), y),
                random.uniform(0.2,0.4), 
                random.uniform(0.3,0.7)*random.uniform(0.2,0.4), 
                angle=random.uniform(-25,25), 
                facecolor=self._get_cloud_color(y,base_km,base_km+thickness,b_min=0.6,b_max=0.8),
                alpha=0.5,
                lw=0))
        ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))

    def _draw_clear_sky(self, ax):
        patches = [
            Ellipse(
                (random.uniform(-1.5,1.5), random.uniform(10,14)), 
                random.uniform(0.5,1.0), 
                random.uniform(0.1,0.2), 
                facecolor='white', 
                alpha=random.uniform(0.05,0.1), 
                lw=0) 
            for _ in range(15)
        ]
        ax.add_collection(PatchCollection(patches, match_original=True, zorder=5))

    def _draw_precipitation(self, ax, base_km, ground_km, p_type, center_x=0.0):
        if p_type == 'virga':
            end_y = max(base_km - random.uniform(1.0, 2.5), ground_km + 0.3)
            for _ in range(50): 
                xs = center_x + random.uniform(-0.6, 0.6)
                xe = xs + random.uniform(-0.1, 0.1)
                ax.plot([xs,xe],[base_km*0.95,end_y], color='lightblue', alpha=random.uniform(0.1,0.3), lw=1.2, zorder=5)
        elif p_type in ['rain', 'sleet']: 
            width = 1.6
            ax.add_patch(Rectangle(
                (center_x-width/2, ground_km),
                width,
                base_km-ground_km,
                facecolor='cornflowerblue',
                alpha=0.25,
                lw=0,
                zorder=5))
                
            for _ in range(100): 
                x = center_x + random.uniform(-width/2, width/2)
                ax.plot([x,x], [base_km*0.95, ground_km], color='blue', alpha=random.uniform(0.1,0.3), lw=0.8, zorder=6)
        elif p_type == 'hail': 
            ax.scatter(
                center_x + np.random.normal(0,0.3,150),
                np.random.uniform(ground_km, base_km, 150), 
                s=np.random.uniform(5,40,150),
                c='white',
                alpha=0.8,
                marker='o',
                edgecolor='gray',
                linewidth=0.5,
                zorder=8)
        elif p_type == 'snow': 
            ax.scatter(
                center_x + np.random.normal(0,0.5,300),
                np.random.uniform(ground_km, base_km, 300), 
                s=np.random.uniform(20,70,300),
                c='white',
                alpha=np.random.uniform(0.4,0.9,300),
                marker='*',
                zorder=8)

    def _calculate_dynamic_cloud_heights(self):
        _, _, lcl_p, lcl_h, _, _, _, el_h, _ = self.calculate_thermo_parameters()
        if not lcl_p: 
            return None, None
            
        cloud_base_km = lcl_h / 1000.0
        
        if self.convergence_active: 
            cloud_top_km = el_h / 1000.0 if el_h > lcl_h else cloud_base_km
        else:
            try:
                rh = mpcalc.relative_humidity_from_dewpoint(self.t_profile, self.td_profile)
                indices_above_lcl = np.where(self.p_levels <= lcl_p)[0]
                p_top = self.p_levels[-1]
                
                if len(indices_above_lcl) > 0:
                    for idx in indices_above_lcl:
                        if rh[idx] < 0.5: 
                            p_top = self.p_levels[idx]
                            break
                            
                cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
            except: 
                cloud_top_km = cloud_base_km
                
        return (cloud_base_km, cloud_top_km) if cloud_top_km > cloud_base_km else (None, None)

    def _draw_saturation_layers(self, ax):
        try:
            saturated_indices = np.where(self.t_profile.m - self.td_profile.m <= 1.5)[0]
            if not len(saturated_indices): 
                return
                
            i=0
            while i < len(saturated_indices):
                start_idx, j = saturated_indices[i], i
                while j+1 < len(saturated_indices) and saturated_indices[j+1] == saturated_indices[j]+1: 
                    j += 1
                    
                end_idx = saturated_indices[j]
                h_bottom = mpcalc.pressure_to_height_std(self.p_levels[start_idx]).to('km').m
                h_top = mpcalc.pressure_to_height_std(self.p_levels[end_idx]).to('km').m
                
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
                        facecolor=(brightness,)*3,
                        alpha=random.uniform(0.1, 0.5),
                        lw=0))
                        
                ax.add_collection(PatchCollection(patches, match_original=True, zorder=7))
                i = j + 1
                
        except Exception: 
            pass

    def draw_clouds(self):
        self.ax_cloud_drawing.cla()
        self.ax_cloud_drawing.set(
            ylim=(0,16), 
            xlim=(-1.5,1.5), 
            xticks=[], 
            facecolor='#6495ED')
        self.ax_cloud_drawing.grid(True, linestyle='dashdot', alpha=0.5)
        
        # Afegir sol
        self.ax_cloud_drawing.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
        
        # Terreny
        ground_color = 'white' if self.precipitation_type == 'snow' else '#228B22'
        self.ax_cloud_drawing.add_patch(Rectangle(
            (-1.5, 0), 
            3, 
            self.ground_height_km, 
            color=ground_color, 
            alpha=0.8, 
            zorder=3, 
            hatch='//' if ground_color == '#228B22' else ''))
            
        self._draw_saturation_layers(self.ax_cloud_drawing)
        
        real_base_km, real_top_km = self._calculate_dynamic_cloud_heights()
        if real_base_km and real_top_km:
            visual_base_km = max(real_base_km, self.ground_height_km + 0.5)
            cloud_depth = real_top_km - real_base_km
            visual_top_km = visual_base_km + cloud_depth
            
            if cloud_depth > 5.0: 
                self._draw_cumulonimbus(self.ax_cloud_drawing, visual_base_km, visual_top_km)
            elif cloud_depth > 2.0: 
                self._draw_cumulus_mediocris(self.ax_cloud_drawing, visual_base_km, visual_top_km)
            else: 
                self._draw_cumulus_fractus(self.ax_cloud_drawing, visual_base_km, cloud_depth)
                
            if self.precipitation_type: 
                self._draw_precipitation(self.ax_cloud_drawing, visual_base_km, self.ground_height_km, self.precipitation_type)
                
        elif not np.any((self.t_profile.m - self.td_profile.m) <= 1.5): 
            self._draw_clear_sky(self.ax_cloud_drawing)

    def _draw_base_feature(self, ax, f_type, base_x_left, base_x_right, base_y, ground_y):
        z = 12
        center_x = (base_x_left + base_x_right) / 2
        width = base_x_right - base_x_left
        
        if f_type == 'lowering': 
            ax.add_patch(Polygon(
                [(base_x_left, base_y), (base_x_right, base_y), 
                 (base_x_right * 0.9 + center_x * 0.1, base_y - 0.2), 
                 (base_x_left * 0.9 + center_x * 0.1, base_y - 0.2)], 
                facecolor='dimgray', 
                edgecolor='gray', 
                zorder=z))
                
        elif f_type == 'wall_cloud':
            top_w = 0.75
            bot_w = 0.55
            wall_h = 0.35
            top_l = center_x - (width * top_w / 2)
            top_r = center_x + (width * top_w / 2)
            bot_l = center_x - (width * bot_w / 2)
            bot_r = center_x + (width * bot_w / 2)
            
            ax.add_patch(Polygon(
                [(top_l, base_y), (top_r, base_y), 
                 (bot_r, base_y - wall_h), 
                 (bot_l, base_y - wall_h)], 
                facecolor='#383838', 
                edgecolor='#202020', 
                lw=0.5, 
                zorder=z))
                
            for i in range(3):
                off_x = random.uniform(-0.05, 0.05)
                off_y = random.uniform(0.0, 0.05)
                alpha = random.uniform(0.1, 0.25)
                color = random.choice(['#404040','#505050'])
                
                s_top_l = top_l + (bot_l - top_l) * 0.1 * i + off_x
                s_top_r = top_r + (bot_r - top_r) * 0.1 * i + off_x
                s_bot_l = bot_l - (bot_l - top_l) * 0.2 * i + off_x
                s_bot_r = bot_r - (bot_r - top_r) * 0.2 * i + off_x
                
                ax.add_patch(Polygon(
                    [(s_top_l, base_y - off_y), 
                     (s_top_r, base_y - off_y), 
                     (s_bot_r, base_y - wall_h + off_y), 
                     (s_bot_l, base_y - wall_h + off_y)], 
                    facecolor=color, 
                    alpha=alpha, 
                    zorder=z+1, 
                    lw=0))
                    
        elif f_type == 'funnel': 
            ax.add_patch(Polygon(
                [(center_x - 0.2, base_y), 
                 (center_x + 0.2, base_y), 
                 (center_x, max(base_y - 0.8, ground_y + 0.5))], 
                facecolor='darkgray', 
                alpha=0.8, 
                zorder=z))
                
        elif f_type == 'tornado':
            ax.add_patch(Polygon(
                [(center_x - 0.2, base_y), 
                 (center_x + 0.2, base_y), 
                 (center_x, ground_y)], 
                facecolor='#505050', 
                zorder=z))
                
            ax.add_patch(Ellipse(
                (center_x, ground_y + 0.05), 
                width=0.7, 
                height=0.25, 
                facecolor='#654321', 
                alpha=0.7, 
                zorder=z+1))
                
            for _ in range(25):
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(0.7 * 0.4, 0.7 * 0.9)
                fx = center_x + r * np.cos(angle)
                fy = ground_y + random.uniform(0, 0.3)
                ax.add_patch(Ellipse(
                    (fx, fy), 
                    random.uniform(0.02, 0.08), 
                    random.uniform(0.02, 0.05), 
                    angle=random.uniform(0, 360), 
                    facecolor='#3a2d1c', 
                    alpha=random.uniform(0.5, 0.8), 
                    zorder=z+2))
       
    def draw_cloud_structure(self):
        for a in [self.ax_cloud_structure, self.ax_shear_barbs, self.ax_cloud_label]: 
            a.cla()
            
        self.ax_cloud_label.axis('off')
        self.ax_cloud_structure.set_title("Estructura Vertical i Cisallament", fontsize=10)
        self.ax_cloud_structure.set_facecolor('skyblue')
        self.ax_cloud_structure.add_patch(Rectangle(
            (-1.5, 0), 
            3, 
            self.ground_height_km, 
            color='darkgreen', 
            alpha=0.7, 
            zorder=1, 
            hatch='//'))
            
        self.ax_cloud_structure.set(
            ylim=(0, 20), 
            xlim=(-1.5, 1.5), 
            ylabel="Altitud (km)", 
            xticks=[])
        self.ax_cloud_structure.grid(True, linestyle='--', alpha=0.3)
        
        self.ax_shear_barbs.set(xlim=(-1, 1), xticks=[])
        self.ax_shear_barbs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        for spine in self.ax_shear_barbs.spines.values(): 
            spine.set_visible(False)
            
        self.ax_shear_barbs.patch.set_alpha(0.0)
        
        base_km, top_km = self._calculate_dynamic_cloud_heights()
        cape, *_ = self.calculate_thermo_parameters()
        
        if not base_km or not top_km or cape.m < 100:
            self.ax_cloud_structure.text(
                0.5, 0.5, 
                "Sense Estructura Convectiva", 
                ha='center', 
                va='center', 
                transform=self.ax_cloud_structure.transAxes, 
                fontsize=9, 
                color='white', 
                bbox=dict(facecolor='darkblue', alpha=0.7))
            self.ax_shear_barbs.axis('off')
            return
            
        visual_base_km = max(base_km, self.ground_height_km + 0.5)
        
        try:
            u, v = mpcalc.wind_components(self.wind_speed, self.wind_dir)
            h_km = mpcalc.pressure_to_height_std(self.p_levels).to('km').m
            
            unique_h, idx = np.unique(h_km, return_index=True)
            if len(unique_h) < 2: 
                return
                
            f_u = interp1d(unique_h, u.m[idx], bounds_error=False, fill_value="extrapolate")
            f_v = interp1d(unique_h, v.m[idx], bounds_error=False, fill_value="extrapolate")
            
            barb_h = np.arange(0, min(20, h_km.max()), 1)
            barb_u = f_u(barb_h) * units('m/s')
            barb_v = f_v(barb_h) * units('m/s')
            
            self.ax_shear_barbs.barbs(
                np.zeros_like(barb_h), 
                barb_h, 
                barb_u.to('knots').m, 
                barb_v.to('knots').m, 
                length=7, 
                pivot='middle', 
                color='k')
                
            alts = np.linspace(visual_base_km, top_km, num=50)
            u_alts = f_u(alts)
            h_offsets = u_alts * 0.02
            shear_0_6, *_ = self.calculate_storm_parameters()
            
            shear_factor = np.clip(shear_0_6 / 35, 0.4, 2.5)
            widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (alts - visual_base_km) / (top_km - visual_base_km + 0.01)) * shear_factor
            
            anvil_ext = np.zeros_like(alts)
            u_anvil_top = 0
            
            if (top_km - visual_base_km) > 4.0:
                anvil_base_alt = top_km * 0.80
                anvil_indices = np.where(alts >= anvil_base_alt)[0]
                
                if len(anvil_indices) > 0:
                    u_anvil_top = f_u(top_km)
                    wind_dir = np.sign(u_anvil_top) if u_anvil_top != 0 else 1
                    max_stretch = abs(u_anvil_top) * 0.06
                    growth = (alts[anvil_indices] - anvil_base_alt) / (top_km - anvil_base_alt)
                    anvil_ext[anvil_indices] = max_stretch * wind_dir * growth**1.5
                    
            r_pts = [(widths[i] + h_offsets[i] + anvil_ext[i], alts[i]) for i in range(len(alts))]
            l_pts = [(-widths[i] + h_offsets[i], alts[i]) for i in range(len(alts))]
            
            self.ax_cloud_structure.add_patch(Polygon(
                r_pts + l_pts[::-1], 
                facecolor='white', 
                edgecolor='lightgray', 
                alpha=0.95, 
                zorder=10))
                
            if (top_km - visual_base_km) > 3.0 and u_anvil_top != 0:
                p_start_alt = top_km * 0.85
                an_ext_start = np.interp(p_start_alt, alts, anvil_ext)
                upd_off_start = np.interp(p_start_alt, alts, h_offsets)
                
                p_center_x = upd_off_start + an_ext_start * 0.75
                p_width = widths[np.argmin(np.abs(alts - p_start_alt))]
                p_alts = np.linspace(p_start_alt, self.ground_height_km, 20)
                
                u_p_alts = f_u(p_alts)
                init_off = u_p_alts[0] * 0.01
                p_offsets = u_p_alts * 0.01 - init_off
                
                r_p_pts = [(p_center_x + p_width/2 + p_offsets[i], p_alts[i]) for i in range(len(p_alts))]
                l_p_pts = [(p_center_x - p_width/2 + p_offsets[i], p_alts[i]) for i in range(len(p_alts))]
                
                self.ax_cloud_structure.add_patch(Polygon(
                    r_p_pts + l_p_pts[::-1], 
                    facecolor='cornflowerblue', 
                    alpha=0.3, 
                    lw=0, 
                    zorder=8))
                    
            base_l = l_pts[0]
            base_r = (widths[0] + h_offsets[0], alts[0])
            self.ax_cloud_structure.add_patch(Polygon(
                [base_l, base_r, (base_r[0], base_r[1] - 0.1), (base_l[0], base_l[1] - 0.1)], 
                facecolor='dimgray', 
                edgecolor='gray', 
                alpha=0.9, 
                zorder=11))
                
            cape, _, _, lcl_h, _, _, _, _, _ = self.calculate_thermo_parameters()
            shear_0_6, shear_0_1, srh_0_3, srh_0_1 = self.calculate_storm_parameters()
            
            label, l_col, bg_col, feature = "Base Plana", 'black', 'lightgray', None
            
            if top_km - base_km > 4.0 and cape.m > 500:
                if (srh_0_1 >= 150 and lcl_h <= 1000 and shear_0_1 >= 15): 
                    label, bg_col, l_col, feature = "TORNADO POSSIBLE", 'darkred', 'white', 'tornado'
                elif (srh_0_1 > 100 and lcl_h < 1200 and shear_0_1 > 12): 
                    label, bg_col, l_col, feature = "Núvol d'Embut/Tuba", 'red', 'white', 'funnel'
                elif srh_0_3 > 150 and shear_0_6 > 18 and cape.m > 1000: 
                    label, bg_col, l_col, feature = "Núvol Paret (Wall Cloud)", 'orange', 'white', 'wall_cloud'
                elif shear_0_1 > 8 and lcl_h < 1500: 
                    label, bg_col, l_col, feature = "Base Rebaixada (Lowering)", 'gold', 'black', 'lowering'
                    
            if feature: 
                self._draw_base_feature(
                    self.ax_cloud_structure, 
                    feature, 
                    l_pts[0][0], 
                    r_pts[0][0], 
                    visual_base_km, 
                    self.ground_height_km)
                    
            self.ax_cloud_label.text(
                0.5, 0.5, 
                label, 
                ha='center', 
                va='center', 
                fontsize=10, 
                weight='bold', 
                color=l_col, 
                bbox=dict(
                    facecolor=bg_col,
                    alpha=0.8,
                    edgecolor='black',
                    boxstyle='round,pad=0.4'))
                    
        except Exception: 
            pass

    def generate_public_warning(self):
        cape, _, _, _, _, _, _, _, fz_h = self.calculate_thermo_parameters()
        sfc_temp = self.t_profile[0]
        
        if fz_h < 1500 or sfc_temp.m < 5:
            if sfc_temp.m <= 0.5: 
                return "AVÍS PER NEU", "Es preveu nevada a cotes baixes amb acumulacions significatives. Preveu problemes de circulació i temperatures baixes.", "navy"
            else:
                p_low = self.p_levels[self.p_levels > (self.p_levels[0].m - 300) * units.hPa]
                if np.any(self.t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5: 
                    return "AVÍS PER PLUJA GEBRADORA", "Pluja gelant o aiguaneu que pot crear glaçades. Extremi les precaucions.", "dodgerblue"
                else: 
                    return "CEL ENNUVOLAT", "Cel tancat amb possibilitat de pluja feble o boira. Temperatures baixes.", "steelblue"
                    
        elif cape.m >= 1000:
            _, shear_0_1, _, srh_0_1 = self.calculate_storm_parameters()
            
            if srh_0_1 > 150 and shear_0_1 > 15: 
                return "AVÍS PER TORNADO", "Condicions favorables per a la formació de tornados. Vigileu el cel i estigueu atents a alertes.", "darkred"
            elif cape.m > 2000: 
                return "AVÍS PER PEDRA", "Tempestes violentes amb pedra grossa possible. Protegiu vehicles i propietats.", "purple"
            else: 
                return "AVÍS PER TEMPESTES", "Tempestes fortes amb llamp, pluja intensa i possible calamarsa.", "darkorange"
                
        else: 
            return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius. Cel variable.", "green"

    def update_plot(self):
        try:
            self.ax_public_warning.cla()
            self.ax_public_warning.axis('off')
            
            title, message, color = self.generate_public_warning()
            self.ax_public_warning.add_patch(Rectangle(
                (0, 0), 
                1, 1, 
                facecolor=color, 
                transform=self.ax_public_warning.transAxes))
                
            self.ax_public_warning.text(
                0.5, 0.85, 
                title, 
                color='white', 
                ha='center', 
                va='center', 
                fontsize=16, 
                weight='bold', 
                transform=self.ax_public_warning.transAxes, 
                wrap=True)
                
            self.ax_public_warning.text(
                0.5, 0.40, 
                message, 
                color='white', 
                ha='center', 
                va='center', 
                fontsize=11, 
                transform=self.ax_public_warning.transAxes, 
                wrap=True, 
                bbox=dict(
                    facecolor='black',
                    alpha=0.2,
                    boxstyle='round,pad=0.4'))

            risk_text, risk_color = self.calculate_flood_risk()
            self.fig.suptitle(
                risk_text, 
                fontsize=16, 
                color='white', 
                backgroundcolor=risk_color, 
                weight='bold')
                
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
                if p_val: 
                    line.set_data(xlims, [p_val.m, p_val.m]) 
                else: 
                    line.set_data([], [])
            
            if self.wind_barbs: 
                self.wind_barbs.remove()
                self.wind_barbs = None
                
            self.draw_parameters_box()
            
            for coll in self.ax.collections:
                if hasattr(coll, "is_cape_cin_patch"): 
                    coll.remove()
                    
            cape_patch = self.skew.shade_cape(self.p_levels, self.t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
            cin_patch = self.skew.shade_cin(self.p_levels, self.t_profile, parcel_prof, facecolor='black', alpha=0.3)
            
            if cape_patch: 
                cape_patch.is_cape_cin_patch = True
            if cin_patch: 
                cin_patch.is_cape_cin_patch = True
            
            self.draw_clouds()
            self.draw_cloud_structure()
            self.draw_static_radar_echo()
            
        except Exception as e: 
            st.error(f"Error fatal actualitzant el gràfic: {str(e)}")

    def load_new_data(self, sounding_data):
        self.original_p_levels = sounding_data['p_levels'].copy()
        self.original_t_profile = sounding_data['t_initial'].copy()
        self.original_td_profile = sounding_data['td_initial'].copy()
        self.observation_time = sounding_data.get('observation_time', 'Hora no disponible')
        self.time_text_obj.set_text(self.observation_time)
        
        try:
            if 'wind_speed_kmh' in sounding_data and sounding_data['wind_speed_kmh'] is not None: 
                self.original_wind_speed = sounding_data['wind_speed_kmh'].to('m/s')
            else: 
                self.original_wind_speed = np.zeros_like(self.original_p_levels.magnitude) * units('m/s')
                
            if 'wind_dir_deg' in sounding_data and sounding_data['wind_dir_deg'] is not None: 
                self.original_wind_dir = sounding_data['wind_dir_deg'].copy()
            else: 
                self.original_wind_dir = np.zeros_like(self.original_p_levels.magnitude) * units.degrees
                
            self.original_u, self.original_v = mpcalc.wind_components(self.original_wind_speed, self.original_wind_dir)
            
        except Exception as e:
            st.warning(f"Error processant dades de vent: {e}")
            self.original_wind_speed = np.zeros_like(self.original_p_levels.magnitude) * units('m/s')
            self.original_wind_dir = np.zeros_like(self.original_p_levels.magnitude) * units.degrees
            self.original_u, self.original_v = self.original_wind_speed.copy(), self.original_wind_speed.copy()
            
        self.reset_profiles()
        self._force_wind_update()
        self.update_plot()

    def _force_wind_update(self):
        if not hasattr(self, 'original_u') or self.original_u is None: 
            self.original_u, self.original_v = mpcalc.wind_components(self.original_wind_speed, self.original_wind_dir)
            
        valid_orig_mask = ~np.isnan(self.original_p_levels.m) & ~np.isnan(self.original_u.m) & ~np.isnan(self.original_v.m)
        
        if np.count_nonzero(valid_orig_mask) < 2:
             self.u = np.full_like(self.p_levels.magnitude, np.nan) * units('m/s')
             self.v = np.full_like(self.p_levels.magnitude, np.nan) * units('m/s')
        else:
            p_orig_valid = self.original_p_levels.m[valid_orig_mask]
            u_orig_valid = self.original_u.m[valid_orig_mask]
            v_orig_valid = self.original_v.m[valid_orig_mask]
            
            unique_p, idx = np.unique(p_orig_valid, return_index=True)
            
            interp_u = interp1d(unique_p, u_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            interp_v = interp1d(unique_p, v_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            
            self.u = interp_u(self.p_levels.magnitude) * units('m/s')
            self.v = interp_v(self.p_levels.magnitude) * units('m/s')
            
        self.wind_speed = mpcalc.wind_speed(self.u, self.v)
        self.wind_dir = mpcalc.wind_direction(self.u, self.v, convention='from')


# ==============================================================================
# 3. LÒGICA DE L'APLICACIÓ STREAMLIT (Nou disseny)
# ==============================================================================
def main():
    st.set_page_config(page_title="Anàlisi de Sondejos - Barcelona", layout="wide")
    st.title("Anàlisi de Sondejos - Barcelona")

    # --- LÒGICA DE CÀRREGA DE DADES ---
    base_files = ["sondeig.txt", "sondeig1.txt", "sondeig2.txt", "sondeig4.txt", "sondeig5.txt"]
    existing_files = [file for file in base_files if os.path.exists(file)]

    # Inicialització de l'estat de la sessió si és la primera execució
    if 'skew_instance' not in st.session_state:
        st.session_state.skew_instance = None
        st.session_state.current_index = 0
        st.session_state.uploaded_file_name = None
        
        if existing_files:
            try:
                with open(existing_files[0], 'r', encoding='utf-8') as f:
                    file_content = f.read()
                soundings = parse_all_soundings(file_content)
                if soundings:
                    st.session_state.skew_instance = AdvancedSkewT(**soundings[0])
            except Exception as e:
                st.error(f"No s'ha pogut carregar el sondeig inicial '{existing_files[0]}': {e}")
    
    # --- MOSTRAR EL GRÀFIC PRINCIPAL ---
    if st.session_state.skew_instance:
        st.pyplot(st.session_state.skew_instance.fig)
    else:
        st.warning("No s'ha pogut carregar cap sondeig. Si us plau, puja un fitxer a continuació.")
        # Creem una figura buida per evitar errors
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sense dades de sondeig", ha='center', va='center')
        st.pyplot(fig)

    st.divider()

    # --- ZONA DE CONTROLS I ANÀLISI (A SOTA DEL GRÀFIC) ---
    col1, col2, col3 = st.columns([2.5, 2, 2])

    # --- Columna 1: Càrrega i Navegació ---
    with col1:
        st.subheader("📁 Dades del Sondeig")
        
        uploaded_file = st.file_uploader(
            "Puja el teu propi fitxer de sondeig (.txt)",
            type="txt",
            key="file_uploader"
        )
        
        # Lògica de càrrega per arxiu pujat
        if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
            st.session_state.uploaded_file_name = uploaded_file.name
            try:
                file_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                soundings = parse_all_soundings(file_content)
                if soundings:
                    st.session_state.skew_instance = AdvancedSkewT(**soundings[0])
                    st.success(f"Carregat: {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error("L'arxiu pujat no conté dades vàlides.")
            except Exception as e:
                st.error(f"Error llegint l'arxiu: {e}")

        if existing_files and not uploaded_file:
            # Botons de navegació
            c1, c2 = st.columns(2)
            if c1.button("← Anterior", use_container_width=True):
                st.session_state.current_index = (st.session_state.current_index - 1 + len(existing_files)) % len(existing_files)
                filepath = existing_files[st.session_state.current_index]
                with open(filepath, 'r', encoding='utf-8') as f:
                    soundings = parse_all_soundings(f.read())
                    if soundings: st.session_state.skew_instance = AdvancedSkewT(**soundings[0])
                st.rerun()

            if c2.button("Següent →", use_container_width=True):
                st.session_state.current_index = (st.session_state.current_index + 1) % len(existing_files)
                filepath = existing_files[st.session_state.current_index]
                with open(filepath, 'r', encoding='utf-8') as f:
                    soundings = parse_all_soundings(f.read())
                    if soundings: st.session_state.skew_instance = AdvancedSkewT(**soundings[0])
                st.rerun()

            current_file_info = f"Visualitzant: **{existing_files[st.session_state.current_index]}** ({st.session_state.current_index + 1}/{len(existing_files)})"
            st.markdown(current_file_info)
            
    # --- Columna 2: Paràmetres d'Ajust ---
    with col2:
        st.subheader("🛠️ Paràmetres d'Ajust")
        if st.session_state.skew_instance:
            skew_instance = st.session_state.skew_instance
            
            # Control de convergència
            convergence_on = st.toggle(
                "Activar convergència", 
                value=skew_instance.convergence_active, 
                help="Simula la convergència a nivells baixos, maximitzant el desenvolupament vertical."
            )
            if convergence_on != skew_instance.convergence_active:
                skew_instance.convergence_active = convergence_on
                skew_instance.update_plot()
                st.rerun()

            # Control de pressió en superfície
            current_p_val = int(skew_instance.current_surface_pressure.magnitude)
            new_pressure = st.number_input(
                "Pressió en superfície (hPa)",
                min_value=int(skew_instance.original_p_levels[-1].m),
                max_value=int(skew_instance.original_p_levels[0].m),
                value=current_p_val,
                step=1
            )
            if new_pressure != current_p_val:
                skew_instance.change_surface_pressure(new_pressure)
                st.rerun()

    # --- Columna 3: Informació Addicional (placeholder) ---
    with col3:
        st.subheader("ℹ️ Informació")
        st.info("""
        Utilitza els controls per navegar entre els sondejos pre-carregats o puja el teu propi fitxer. 
        Ajusta els paràmetres per veure com afecten l'anàlisi en temps real.
        """)

    st.divider()
    
    # --- XAT D'ANÀLISI (Ocupant tota l'amplada) ---
    st.subheader("💬 Xat d'Anàlisi (Laia i Marc)")
    if st.session_state.skew_instance:
        analysis_text = st.session_state.skew_instance.generate_detailed_analysis()
        st.text_area(
            "Transcripció de l'anàlisi:",
            value=analysis_text,
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )

if __name__ == '__main__':
    main()
