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
# 1. FUNCIÓ DE PARSEIG DE DADES (Sense canvis)
# ==============================================================================
def parse_all_soundings(file_content):
    """
    Llegeix el contingut d'un fitxer de sondeig (en format text) que pot contenir
    múltiples sondejos i els retorna com una llista de diccionaris.
    Tradueix automàticament el text de l'hora del francès al català.
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


# ==============================================================================
# 2. CLASSE DE VISUALITZACIÓ PER A STREAMLIT (CORREGIDA I AMPLIADA)
# ==============================================================================
class StreamlitSkewT:
    def __init__(self, sounding_data):
        self.load_new_data(sounding_data)
        
        # Configuració de la figura
        self.fig = plt.figure(figsize=(18, 15))
        self.fig.subplots_adjust(left=0.08, right=0.75, top=0.93, bottom=0.1)
        
        self.skew = SkewT(self.fig, rotation=45)
        self.ax = self.skew.ax
        
        # Eixos per als panells addicionals
        self.ax_radar_sim = self.fig.add_axes([0.77, 0.74, 0.21, 0.19])
        self.ax_cloud_drawing = self.fig.add_axes([0.77, 0.50, 0.21, 0.22])
        self.ax_cloud_structure = self.fig.add_axes([0.77, 0.10, 0.17, 0.35])
        self.ax_shear_barbs = self.fig.add_axes([0.94, 0.10, 0.03, 0.35], sharey=self.ax_cloud_structure)
        
        self.setup_radar_sim()
        self.setup_plot()
        self.update_plot()

    def load_new_data(self, sounding_data):
        self.original_p_levels = sounding_data['p_levels'].copy()
        self.original_t_profile = sounding_data['t_initial'].copy()
        self.original_td_profile = sounding_data['td_initial'].copy()
        self.observation_time = sounding_data.get('observation_time', 'Hora no disponible')
        
        ws = sounding_data.get('wind_speed_kmh')
        wd = sounding_data.get('wind_dir_deg')

        self.original_wind_speed = ws.to('m/s') if ws is not None else np.zeros_like(self.original_p_levels.magnitude) * units('m/s')
        self.original_wind_dir = wd.copy() if wd is not None else np.zeros_like(self.original_p_levels.magnitude) * units.degrees
        
        self.params_text_box = None
        
        self.reset_profiles()
        self._force_wind_update()

    def reset_profiles(self):
        p_orig_mag = self.original_p_levels.magnitude
        unique_p, unique_idx = np.unique(p_orig_mag, return_index=True)
        self.p_levels = self.original_p_levels[unique_idx]
        self.t_profile = self.original_t_profile[unique_idx]
        self.td_profile = self.original_td_profile[unique_idx]
        
        self.current_surface_pressure = self.p_levels[0]
        self.ground_height_km = mpcalc.pressure_to_height_std(self.current_surface_pressure).to('km').magnitude

    def _force_wind_update(self):
        self.original_u, self.original_v = mpcalc.wind_components(self.original_wind_speed, self.original_wind_dir)
        valid_orig_mask = ~np.isnan(self.original_p_levels.m) & ~np.isnan(self.original_u.m) & ~np.isnan(self.original_v.m)
        if np.count_nonzero(valid_orig_mask) < 2:
            self.u, self.v = (np.full_like(self.p_levels.magnitude, np.nan) * units('m/s'),)*2
        else:
            p_orig_valid, u_orig_valid, v_orig_valid = self.original_p_levels.m[valid_orig_mask], self.original_u.m[valid_orig_mask], self.original_v.m[valid_orig_mask]
            unique_p, idx = np.unique(p_orig_valid, return_index=True)
            interp_u = interp1d(unique_p, u_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            interp_v = interp1d(unique_p, v_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            self.u, self.v = interp_u(self.p_levels.magnitude) * units('m/s'), interp_v(self.p_levels.magnitude) * units('m/s')
        self.wind_speed = mpcalc.wind_speed(self.u, self.v)
        self.wind_dir = mpcalc.wind_direction(self.u, self.v, convention='from')
    
    def change_surface_pressure(self, new_pressure_hpa):
        new_p_sfc = new_pressure_hpa * units.hPa
        self.current_surface_pressure = new_p_sfc
        
        # Crea una nova graella de pressió des de la nova superfície fins a 100 hPa
        p_new_mag = np.arange(new_p_sfc.m, 99.0, -10.0)
        self.p_levels = p_new_mag * units.hPa

        # Interpola els perfils originals (T, Td, u, v) a la nova graella de pressió
        p_orig_mag = self.original_p_levels.magnitude
        unique_p_orig, unique_idx_orig = np.unique(p_orig_mag, return_index=True)
        
        f_t = interp1d(unique_p_orig, self.original_t_profile.magnitude[unique_idx_orig], bounds_error=False, fill_value="extrapolate")
        f_td = interp1d(unique_p_orig, self.original_td_profile.magnitude[unique_idx_orig], bounds_error=False, fill_value="extrapolate")
        f_u = interp1d(unique_p_orig, self.original_u.magnitude[unique_idx_orig], bounds_error=False, fill_value="extrapolate")
        f_v = interp1d(unique_p_orig, self.original_v.magnitude[unique_idx_orig], bounds_error=False, fill_value="extrapolate")

        self.t_profile = f_t(p_new_mag) * units.degC
        self.td_profile = f_td(p_new_mag) * units.degC
        self.u = f_u(p_new_mag) * units('m/s')
        self.v = f_v(p_new_mag) * units('m/s')
        self.wind_speed = mpcalc.wind_speed(self.u, self.v)
        self.wind_dir = mpcalc.wind_direction(self.u, self.v, convention='from')

        self.ground_height_km = mpcalc.pressure_to_height_std(self.current_surface_pressure).to('km').magnitude
        self.update_ground_patch()
    
    # --- Funcions de Càlcul ---
    def calculate_thermo_parameters(self):
        try:
            p, t, td = self.p_levels, self.t_profile, self.td_profile
            p_sfc, t_sfc, td_sfc = p[0], t[0], td[0]
            
            # Versió simplificada sense l'argument integrator
            parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
            
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
            lcl_p, _ = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
            lfc_p, _ = mpcalc.lfc(p, t, parcel_prof)
            el_p, _ = mpcalc.el(p, t, parcel_prof)

            # Càlcul més robust del nivell de 0ºC
            try:
                heights_std = mpcalc.pressure_to_height_std(p)
                t_interp_func = interp1d(heights_std.m, t.m, bounds_error=False, fill_value="extrapolate")
                h_interp = np.arange(0, 10000, 10)
                t_on_h = t_interp_func(h_interp)
                fz_indices = np.where(t_on_h <= 0)[0]
                fz_h = h_interp[fz_indices[0]] if len(fz_indices) > 0 else np.nan
            except Exception: 
                fz_h = np.nan

            if el_p is None and cape.magnitude > 0: el_p = p[-1]
            ground_h = mpcalc.pressure_to_height_std(p_sfc).to('m').m
            lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m - ground_h if lcl_p else 0
            lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m - ground_h if lfc_p else np.inf
            el_h = mpcalc.pressure_to_height_std(el_p).to('m').m - ground_h if el_p else lfc_h
            fz_h_agl = fz_h - ground_h if not np.isnan(fz_h) else 0

            return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h_agl
        except Exception: 
            return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0)

    def calculate_storm_parameters(self):
        try:
            p, u, v = self.p_levels, self.u, self.v
            heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
            
            sfc_h = heights_raw[0].m
            h_agl = heights_raw - sfc_h

            s_0_6 = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=h_agl, depth=6000 * units.meter)).to('m/s').m
            s_0_1 = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=h_agl, depth=1000 * units.meter)).to('m/s').m
            
            srh_0_3 = mpcalc.storm_relative_helicity(h_agl, u, v, depth=3000*units.meter)[0].m
            srh_0_1 = mpcalc.storm_relative_helicity(h_agl, u, v, depth=1000*units.meter)[0].m
            return s_0_6, s_0_1, srh_0_3, srh_0_1
        except Exception: return 0.0, 0.0, 0.0, 0.0

    def calculate_steering_wind(self):
        try:
            p, u, v = self.p_levels, self.u, self.v
            heights = mpcalc.pressure_to_height_std(p).to('km')
            ground_h = heights[0].m
            mask = (heights.m >= ground_h + 2) & (heights.m <= ground_h + 6) # steering 2-6 km AGL
            if np.sum(mask) < 2: return np.mean(u), np.mean(v)
            return np.mean(u[mask]), np.mean(v[mask])
        except Exception: return 0 * units('m/s'), 0 * units('m/s')

    # --- Funcions de Dibuix ---
    def _draw_humidity_layers(self, ax):
        try:
            rh = mpcalc.relative_humidity_from_dewpoint(self.t_profile, self.td_profile)
            heights_km = mpcalc.pressure_to_height_std(self.p_levels).to('km').m
            in_layer = False
            layer_start_h = 0
            for i in range(len(rh)):
                is_humid = rh[i] >= 0.70
                if is_humid and not in_layer:
                    in_layer = True
                    layer_start_h = heights_km[i]
                if (not is_humid and in_layer) or (is_humid and i == len(rh) - 1):
                    in_layer = False
                    layer_end_h = heights_km[i] if (is_humid and i == len(rh) - 1) else heights_km[i-1]
                    if layer_end_h > layer_start_h:
                        avg_h = (layer_start_h + layer_end_h) / 2
                        color = 'white' if avg_h > (self.fz_h/1000 + self.ground_height_km) else 'silver'
                        thickness = layer_end_h - layer_start_h
                        rect = Rectangle((-1.5, layer_start_h), 3, thickness, color=color, alpha=0.6, zorder=2)
                        ax.add_patch(rect)
        except Exception: pass

    def _calculate_dynamic_cloud_heights(self):
        base_km = (self.lcl_h / 1000) + self.ground_height_km if self.lcl_h else self.ground_height_km
        top_km = (self.el_h / 1000) + self.ground_height_km if self.el_h > self.lcl_h else base_km + 0.5
        
        # Limita l'alçada per al dibuix
        top_km = min(top_km, 15.5)
        base_km = min(base_km, top_km)
        
        return base_km, top_km

    def draw_clouds(self):
        self.ax_cloud_drawing.cla()
        self.ax_cloud_drawing.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], yticks=[], facecolor='#6495ED')
        self.ax_cloud_drawing.grid(True, linestyle='dashdot', alpha=0.5)
        self.ax_cloud_drawing.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
        self._draw_humidity_layers(self.ax_cloud_drawing)
        
        ground_color = 'white' if self.precipitation_type == 'snow' else '#228B22'
        self.ax_cloud_drawing.add_patch(Rectangle((-1.5, 0), 3, self.ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//'))
        
        real_base_km, real_top_km = self._calculate_dynamic_cloud_heights()

        def draw_precipitation(base_km, p_type, intensity):
            y = np.linspace(0, base_km, 20)
            for _ in range(int(20 * intensity)):
                x_start = random.uniform(-1.2, 1.2)
                if p_type == 'rain': self.ax_cloud_drawing.plot([x_start, x_start], y, color='blue', lw=0.5, alpha=0.6, zorder=3)
                elif p_type == 'hail': self.ax_cloud_drawing.plot([x_start], [random.uniform(0, base_km)], 'o', color='cyan', ms=3, alpha=0.7, zorder=3)
                elif p_type == 'snow': self.ax_cloud_drawing.plot([x_start], [random.uniform(0, base_km)], '*', color='white', ms=4, alpha=0.7, zorder=3)

        if self.precipitation_type in ['supercell', 'multicell', 'storm', 'cumulus'] and self.lfc_h / 1000 < 4:
            width_base = 1.8
            width_top = 2.8
            cloud_points = [
                (-width_base/2, real_base_km), (width_base/2, real_base_km),
                (width_top/2, real_top_km), (-width_top/2, real_top_km)
            ]
            self.ax_cloud_drawing.add_patch(Polygon(cloud_points, color='darkgrey', alpha=0.8, zorder=4))
            # Dibuixar enclusa
            if real_top_km > 8:
                enclusa_points = [
                    (-width_top/2, real_top_km), (width_top/2, real_top_km),
                    (width_top/2 + 0.5, real_top_km - 1), (-width_top/2 - 0.5, real_top_km - 1.5)
                ]
                self.ax_cloud_drawing.add_patch(Polygon(enclusa_points, color='lightgrey', alpha=0.8, zorder=5))
            
            p_type = 'hail' if self.cape.m > 1500 else 'rain'
            draw_precipitation(real_base_km, p_type, self.cape.m/500)

        elif self.precipitation_type in ['snow', 'sleet']:
            p_type = 'snow' if self.precipitation_type == 'snow' else 'rain'
            draw_precipitation(self.ground_height_km + 1, p_type, 1.5)

    def draw_cloud_structure(self):
        self.ax_cloud_structure.cla(); self.ax_shear_barbs.cla()
        self.ax_cloud_structure.set_facecolor('lightcyan')
        self.ax_cloud_structure.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], ylabel="Altitud (km)", title="Estructura i Cisallament")
        self.ax_shear_barbs.set(ylim=(0, 16), xticks=[], yticklabels=[])
        self.ax_cloud_structure.grid(True, linestyle=':', alpha=0.7)
        self._draw_humidity_layers(self.ax_cloud_structure)
        self.ax_cloud_structure.add_patch(Rectangle((-1.5, 0), 3, self.ground_height_km, color='saddlebrown', alpha=0.6, zorder=1))
        
        # Dibuixar línies de nivells importants
        levels_to_plot = {
            'LCL': self.lcl_h / 1000 + self.ground_height_km,
            'LFC': self.lfc_h / 1000 + self.ground_height_km,
            'EL': self.el_h / 1000 + self.ground_height_km,
            '0°C': self.fz_h / 1000 + self.ground_height_km
        }
        for name, h_km in levels_to_plot.items():
            if h_km > self.ground_height_km and h_km < 16:
                self.ax_cloud_structure.axhline(h_km, color='red', linestyle='--', lw=1, xmin=0.05, xmax=0.95)
                self.ax_cloud_structure.text(1.4, h_km, name, color='red', ha='right', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=1))
        
        # Dibuixar barbes de vent
        heights_km = mpcalc.pressure_to_height_std(self.p_levels).to('km').m
        mask = (heights_km >= 0) & (heights_km <= 16)
        step = max(1, len(self.p_levels[mask]) // 15) # Dibuixar unes 15 barbes
        self.ax_shear_barbs.barbs(
            np.zeros_like(heights_km[mask][::step]), 
            heights_km[mask][::step],
            self.u[mask][::step].to('kt').m, 
            self.v[mask][::step].to('kt').m,
            length=7,
            pivot='middle'
        )

    def draw_static_radar_echo(self):
        self.ax_radar_sim.cla()
        self.setup_radar_sim() # Reinicia la configuració base de l'eix
        
        if self.cape.m < 100: return # No hi ha eco si no hi ha convecció

        # Coordenades
        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        xx, yy = np.meshgrid(x, y)

        # Intensitat i mida de la tempesta
        intensity = min(70, 20 + self.cape.m / 40)
        size = 10 + self.cape.m / 200
        
        # Posició inicial del centre de la tempesta
        cx, cy = 0, 0
        
        # Simula diferents tipus de tempesta
        if self.precipitation_type == 'supercell':
            # Eco principal
            cell = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * size**2))
            # Eco del ganxo (hook echo)
            hx, hy = cx - size*0.8, cy - size*0.8
            hook = (intensity * 0.8) * np.exp(-((xx - hx)**2 + (yy - hy)**2) / (2 * (size*0.5)**2))
            storm_echo = gaussian_filter(cell + hook, sigma=2)
        elif self.precipitation_type == 'multicell':
            cell1 = intensity * np.exp(-((xx - 5)**2 + (yy - 10)**2) / (2 * size**2))
            cell2 = (intensity*0.8) * np.exp(-((xx + 10)**2 + (yy + 5)**2) / (2 * (size*0.8)**2))
            cell3 = (intensity*0.6) * np.exp(-((xx - 10)**2 + (yy + 15)**2) / (2 * (size*0.7)**2))
            storm_echo = gaussian_filter(cell1 + cell2 + cell3, sigma=3)
        else: # Tempesta aïllada ('storm' o 'cumulus')
            cell = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * size**2))
            storm_echo = gaussian_filter(cell, sigma=2)
        
        self.ax_radar_sim.contourf(xx, yy, storm_echo, levels=self.radar_levels, cmap=self.radar_cmap, norm=self.radar_norm)
        
        # Afegir fletxa de moviment (steering wind)
        u_steer, v_steer = self.calculate_steering_wind()
        if u_steer.m != 0 or v_steer.m != 0:
            self.ax_radar_sim.arrow(cx - 30, cy - 30, u_steer.m*2, v_steer.m*2, 
                                     color='white', width=1.5, head_width=5, length_includes_head=True)

    def draw_parameters_box(self):
        param_text = (
            f"CAPE: {self.cape.m:.0f} J/kg\n"
            f"CIN: {self.cin.m:.0f} J/kg\n"
            f"LCL: {self.lcl_h:.0f} m\n"
            f"LFC: {self.lfc_h:.0f} m\n"
            f"EL: {self.el_h:.0f} m\n"
            f"0°C Level: {self.fz_h:.0f} m\n---\n"
            f"Shear 0-1km: {self.shear_0_1:.1f} m/s\n"
            f"Shear 0-6km: {self.shear_0_6:.1f} m/s\n"
            f"SRH 0-1km: {self.srh_0_1:.0f} m²/s²\n"
            f"SRH 0-3km: {self.srh_0_3:.0f} m²/s²"
        )
        self.params_text_box = self.ax.text(0.98, 0.98, param_text, transform=self.ax.transAxes,
                                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                                            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

    # --- Anàlisi i Avisos ---
    def generate_detailed_analysis(self):
        self.precipitation_type = None
        sfc_rh = mpcalc.relative_humidity_from_dewpoint(self.t_profile[0], self.td_profile[0]).m * 100
        
        if self.fz_h < 1500 or self.t_profile[0].m < 5:
            text = f"--- XAT D'HIVERN ---\nMarc: Iso 0°C?\n> {self.fz_h:.0f}m. Molt baixa.\nLaia: Llavors neu o gel.\nMarc: Humitat en superfície?\n> {sfc_rh:.0f}%. Saturat.\n"
            if self.t_profile[0].m <= 0.5: 
                self.precipitation_type = 'snow'
                text += "Laia: El perfil és 100% nival?\n> Sí. Fred a tots els nivells.\nMarc: Conclusió?\n> Nevada segura. Prepara les cadenes.\n"
            else: 
                self.precipitation_type = 'sleet'
                text += "Laia: Compte, veig una capa càlida.\n> Correcte, a mitja altura.\nMarc: Llavors?\n> Risc alt de pluja gelant. Molt perillós.\n"
            return text
        elif self.cape.m > 800 and self.shear_0_6 > 15 and self.lfc_h < 2500:
            is_supercell = self.shear_0_6 > 18 and self.srh_0_3 > 150
            self.precipitation_type = 'supercell' if is_supercell else 'multicell'
            text = f"--- XAT DE CAÇA (SEVER) ---\n"
            text += f"Anna: Tenim CAPE ({self.cape.m:.0f}) i Shear 0-6km ({self.shear_0_6:.1f}). Potent.\n"
            text += f"Pau: LFC baix ({self.lfc_h:.0f}m), la ignició és fàcil.\n"
            if is_supercell:
                text += f"Anna: SRH alt ({self.srh_0_3:.0f}), això gira. Supercèl·lula probable.\n"
                text += "Pau: Risc de pedra grossa i/o tornados. Màxima alerta.\n"
            else:
                text += "Anna: SRH més baix. Anem per multicèl·lules o línies de tempesta.\n"
                text += "Pau: Correcte. Ventades fortes i pluges intenses.\n"
            return text
        elif self.cape.m >= 100:
            self.precipitation_type = 'storm'
            text = f"--- XAT DE TARDA (CAPE: {int(self.cape.m)}) ---\n"
            text += f"Jordi: Tenim energia, però poc organitzada (Shear: {self.shear_0_6:.1f}).\n"
            text += f"Clara: Exacte. Tempestes de pols únic, poca durada.\n"
            text += f"Jordi: Algun xàfec intens local, potser amb calamarsa petita.\n"
            text += "Clara: Res de què preocupar-se gaire, el típic de l'estiu.\n"
            return text
        else:
            self.precipitation_type = 'fair'
            text = "--- XAT DE TEMPS (BONANÇA) ---\n"
            text += "Eva: Gens de CAPE. Estabilitat total.\n"
            text += "Nil: Exacte. El dia serà tranquil, potser amb alguns núvols alts.\n"
            text += "Eva: Perfecte per fer una excursió a la muntanya.\n"
            text += "Nil: Confirmat. Bon temps assegurat.\n"
            return text
    
    def generate_public_warning(self):
        sfc_temp = self.t_profile[0].m
        if self.fz_h < 1500 or sfc_temp < 5:
            if sfc_temp <= 0.5: return "AVÍS PER NEU", "Es preveu nevada a cotes baixes. Possibles gruixos importants.", "navy"
            else:
                p_low_indices = self.p_levels > (self.p_levels[0] - 300 * units.hPa)
                if np.any(self.t_profile[p_low_indices].m > 0.5) and sfc_temp < 2.5: 
                    return "AVÍS PER PLUJA GEBRADORA", "Risc alt de pluja que es congela en contacte amb el terra. Condicions molt perilloses.", "purple"
                else: 
                    return "CEL ENNUVOLAT AMB PLUJA", "Cel tancat amb possibilitat de pluja feble o plugim. Ambient fred.", "steelblue"
        elif self.cape.m >= 1500 and self.lfc_h < 2500:
            if self.srh_0_1 > 100 and self.shear_0_1 > 10: 
                return "AVÍS PER TORNADO", "Condicions molt favorables per a la formació de supercèl·lules i tornados. Risc extrem.", "darkred"
            elif self.cape.m > 2500: 
                return "AVÍS PER PEDRA GROSSA", "Tempestes violentes amb alta probabilitat de calamarsa o pedra de gran mida.", "darkmagenta"
            else: 
                return "AVÍS PER TEMPESTES FORTES", "Es preveuen tempestes fortes amb pluja torrencial, ratxes de vent i possible calamarsa.", "darkorange"
        elif self.cape.m >= 400:
             return "POSSIBLES XÀFECS I TRONADES", "Possibilitat de xàfecs o tronades disperses, especialment a la tarda.", "goldenrod"
        else: 
            return "SENSE AVISOS", "Condicions meteorològiques estables i sense riscos significatius.", "green"

    # --- Configuració i Actualització del Gràfic ---
    def setup_radar_sim(self):
        self.ax_radar_sim.set_facecolor('darkslategray'); self.ax_radar_sim.set_title("Eco Radar Simulat", fontsize=10)
        self.ax_radar_sim.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
        self.ax_radar_sim.set_xlim(-50, 50); self.ax_radar_sim.set_ylim(-50, 50); self.ax_radar_sim.grid(True, linestyle=':', alpha=0.3, color='white')
        self.radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
        self.radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
        self.radar_cmap = ListedColormap(self.radar_colors); self.radar_norm = BoundaryNorm(self.radar_levels, self.radar_cmap.N)
    
    def setup_plot(self):
        self.ax.set_ylim(1050, 100); self.ax.set_xlim(-50, 45)
        self.skew.plot_dry_adiabats(alpha=0.3, color='orange'); self.skew.plot_moist_adiabats(alpha=0.3, color='green'); self.skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        self.line_t, = self.skew.plot([], [], 'r', linewidth=2, label='Temperatura (T)')
        self.line_td, = self.skew.plot([], [], 'b', linewidth=2, label='Punt de Rosada (Td)')
        self.line_parcel, = self.skew.plot([], [], 'k--', linewidth=2, label='Bombolla Adiabàtica')
        self.line_wb, = self.skew.plot([], [], color='purple', linewidth=1.5, label='Tª Bombolla Humida')
        self.line_lcl, = self.ax.plot([], [], 'gray', linestyle='--', label='LCL'); self.line_lfc, = self.ax.plot([], [], 'purple', linestyle='--', label='LFC'); self.line_el, = self.ax.plot([], [], 'red', linestyle='--', label='EL')
        self.ground_patch = Rectangle((0, 0), 1, 1, color='darkgreen', alpha=0.7); self.ax.add_patch(self.ground_patch)
        self.update_ground_patch()

    def update_ground_patch(self):
        y_min = self.current_surface_pressure.magnitude
        self.ground_patch.set_xy((-50, y_min)); self.ground_patch.set_width(95); self.ground_patch.set_height(max(20, self.p_levels[0].m - self.p_levels[1].m))
        self.ground_patch.set_zorder(-1)

    def update_plot(self):
        if self.params_text_box: self.params_text_box.remove()
        for coll in self.ax.collections[:]:
            if hasattr(coll, "is_cape_cin_patch"): coll.remove()

        # Càlcul de paràmetres
        self.cape, self.cin, self.lcl_p, self.lcl_h, self.lfc_p, self.lfc_h, self.el_p, self.el_h, self.fz_h = self.calculate_thermo_parameters()
        self.shear_0_6, self.shear_0_1, self.srh_0_3, self.srh_0_1 = self.calculate_storm_parameters()
        
        # Dibuix principal
        self.td_profile = np.minimum(self.t_profile, self.td_profile)
        self.line_t.set_data(self.t_profile, self.p_levels)
        self.line_td.set_data(self.td_profile, self.p_levels)
        
        # Versió simplificada sense l'argument integrator
        parcel_prof = mpcalc.parcel_profile(self.p_levels, self.t_profile[0], self.td_profile[0]).to('degC')
        self.line_parcel.set_data(parcel_prof, self.p_levels)
        self.line_wb.set_data(mpcalc.wet_bulb_temperature(self.p_levels, self.t_profile, self.td_profile), self.p_levels)
        
        xlims = self.ax.get_xlim()
        for line, p_val in [(self.line_lcl, self.lcl_p), (self.line_lfc, self.lfc_p), (self.line_el, self.el_p)]:
            if p_val: line.set_data(xlims, [p_val.m, p_val.m]) 
            else: line.set_data([], [])

        cape_patch = self.skew.shade_cape(self.p_levels, self.t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
        cin_patch = self.skew.shade_cin(self.p_levels, self.t_profile, parcel_prof, facecolor='black', alpha=0.3)
        if cape_patch: cape_patch.is_cape_cin_patch = True
        if cin_patch: cin_patch.is_cape_cin_patch = True

        # Crida a les funcions de dibuix i anàlisi
        self.draw_parameters_box()
        self.generate_detailed_analysis() # Aquesta crida defineix `self.precipitation_type`
        self.draw_clouds()
        self.draw_cloud_structure()
        self.draw_static_radar_echo()


# ==============================================================================
# 3. LÒGICA DE L'APLICACIÓ STREAMLIT
# ==============================================================================
def main():
    st.set_page_config(page_title="Visor de Sondejos", layout="wide")
    st.title("Visor de Sondejos Meteorològics Interactiu")

    st.sidebar.header("Configuració")
    
    # Llista de fitxers disponibles (crea'ls si no existeixen per a proves)
    base_files = [f"{h}{p}.txt" for h in range(1, 13) for p in ['am', 'pm']]
    existing_files = []
    for file in base_files:
        if os.path.exists(file):
            existing_files.append(file)
    
    if not existing_files:
        st.error("No s'ha trobat cap fitxer de sondeig al directori. Assegura't que els fitxers (ex: '1am.txt') existeixen.")
        st.info("Pots crear arxius de text buits amb aquests noms per evitar aquest error.")
        return

    selected_file = st.sidebar.selectbox("Selecciona un sondeig:", options=existing_files)

    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        all_soundings = parse_all_soundings(file_content)
        if not all_soundings:
            st.error(f"L'arxiu '{selected_file}' no conté dades de sondeig vàlides o està buit.")
            return
        
        current_data = all_soundings[0]

        if 'skew_instance' not in st.session_state or st.session_state.get('current_file') != selected_file:
            st.session_state.skew_instance = StreamlitSkewT(current_data)
            st.session_state.current_file = selected_file
        
        skew_instance = st.session_state.skew_instance

    except Exception as e:
        st.error(f"Error en carregar o processar '{selected_file}': {e}")
        st.exception(e)
        return

    # --- PANTALLA PRINCIPAL ---
    st.subheader(f"Dades per a: {skew_instance.observation_time.replace(chr(10), ' | ')}")
    
    title, message, color = skew_instance.generate_public_warning()
    st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; color:white;"><h4 style="color:white;">{title}</h4>{message}</div>', unsafe_allow_html=True)
    
    # --- CONTROLS A LA BARRA LATERAL ---
    st.sidebar.subheader("Ajustos en temps real")
    
    current_p_val = int(skew_instance.current_surface_pressure.magnitude)
    new_pressure = st.sidebar.slider(
        "Pressió en superfície (hPa)",
        min_value=900,
        max_value=int(skew_instance.original_p_levels[0].m),
        value=current_p_val,
        step=1,
        key=f"pressure_slider_{selected_file}"
    )
    
    if new_pressure != current_p_val:
        skew_instance.change_surface_pressure(new_pressure)
        skew_instance.update_plot()
        st.rerun()

    st.sidebar.toggle("Activar convergència (Visual)", value=True, help="Aquesta opció és només demostrativa.")

    st.pyplot(skew_instance.fig)

    with st.expander("Veure Anàlisi Detallada (Xat Simulat)"):
        analysis_text = skew_instance.generate_detailed_analysis()
        st.code(analysis_text)

if __name__ == '__main__':
    main()
