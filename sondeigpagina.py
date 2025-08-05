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
# SECCIÓ 1: LÒGICA DE CÀLCUL I DADES
# ==============================================================================

@st.cache_data(show_spinner="Processant arxiu de sondeig...")
def parse_all_soundings(filepath):
    """
    Llegeix un fitxer de text que pot contenir múltiples sondejos i els retorna
    com una llista de diccionaris.
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
        cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
        if not cleaned_text or cleaned_text == '-':
            return None
        try:
            return float(cleaned_text)
        except ValueError:
            return None

    def process_sounding_block(block_lines):
        if not block_lines:
            return None

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
        self.observation_time = sounding_data.get('observation_time', "Hora no disponible")
        self.original_wind_speed = sounding_data['wind_speed_kmh'].to('m/s') if sounding_data.get('wind_speed_kmh') is not None else np.zeros(len(self.original_p_levels)) * units('m/s')
        self.original_wind_dir = np.nan_to_num(sounding_data.get('wind_dir_deg', 0), nan=0) * units.degrees
        self.current_surface_pressure = self.original_p_levels[0]
        self.convergence_active = True
        self.precipitation_type = None

        self.fig = plt.figure(figsize=(20, 15))
        self.fig.subplots_adjust(left=0.25, right=0.75, top=0.93, bottom=0.1)
        
        self.skew = SkewT(self.fig, rotation=45)
        self.ax = self.skew.ax
        
        self.ax_public_warning = self.fig.add_axes([-0.015, 0.78, 0.22, 0.19])
        self.ax_info_panel = self.fig.add_axes([-0.015, 0.1, 0.22, 0.66])
        self.ax_radar_sim = self.fig.add_axes([0.25, 0.74, 0.18, 0.20])
        self.ax_cloud_drawing = self.fig.add_axes([0.77, 0.58, 0.22, 0.39])
        self.ax_cloud_structure = self.fig.add_axes([0.77, 0.15, 0.18, 0.38])
        self.ax_shear_barbs = self.fig.add_axes([0.95, 0.15, 0.03, 0.38], sharey=self.ax_cloud_structure)
        self.ax_cloud_label = self.fig.add_axes([0.77, 0.10, 0.18, 0.05])
        self.time_text_ax = self.fig.add_axes([0.35, 0.03, 0.3, 0.04])
        self.time_text_ax.axis('off')
        
        self.setup_plot()
        self.reset_profiles()
        self._force_wind_update()

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
            self.t_profile = np.concatenate(([f_t(new_p.m) * units.degC], t_masked[p_masked < new_p]))
            self.td_profile = np.concatenate(([f_td(new_p.m) * units.degC], td_masked[p_masked < new_p]))
            self.wind_speed = np.concatenate(([f_ws(new_p.m) * units('m/s')], ws_masked[p_masked < new_p]))
            self.wind_dir = np.concatenate(([f_wd(new_p.m) * units.degrees], wd_masked[p_masked < new_p]))
        except Exception: self.reset_profiles()

    def _force_wind_update(self):
        if not hasattr(self, 'original_u') or self.original_u is None:
            self.original_u, self.original_v = mpcalc.wind_components(self.original_wind_speed, self.original_wind_dir)
        valid_orig_mask = ~np.isnan(self.original_p_levels.m) & ~np.isnan(self.original_u.m) & ~np.isnan(self.original_v.m)
        if np.count_nonzero(valid_orig_mask) < 2:
            self.u = np.full_like(self.p_levels.magnitude, np.nan) * units('m/s')
            self.v = np.full_like(self.p_levels.magnitude, np.nan) * units('m/s')
        else:
            p_orig_valid = self.original_p_levels.m[valid_orig_mask]; u_orig_valid = self.original_u.m[valid_orig_mask]; v_orig_valid = self.original_v.m[valid_orig_mask]
            unique_p, idx = np.unique(p_orig_valid, return_index=True)
            interp_u = interp1d(unique_p, u_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            interp_v = interp1d(unique_p, v_orig_valid[idx], bounds_error=False, fill_value="extrapolate")
            self.u = interp_u(self.p_levels.magnitude) * units('m/s')
            self.v = interp_v(self.p_levels.magnitude) * units('m/s')
        self.wind_speed = mpcalc.wind_speed(self.u, self.v); self.wind_dir = mpcalc.wind_direction(self.u, self.v, convention='from')

    def calculate_thermo_parameters(self):
        try:
            p, t, td = self.p_levels, self.t_profile, self.td_profile
            valid = ~np.isnan(p.m) & ~np.isnan(t.m) & ~np.isnan(td.m)
            if np.sum(valid) < 2: return (units.Quantity(0, 'J/kg'),)*2 + (None, 0)*4 + (0,)
            p, t, td = p[valid], t[valid], td[valid]
            parcel_prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
            lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])
            lfc_p, _ = mpcalc.lfc(p, t, td, parcel_prof)
            el_p, _ = mpcalc.el(p, t, td, parcel_prof)
            t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")
            p_range = np.arange(p.m.min(), p.m.max())
            t_range = t_interp(p_range)
            fz_idx = np.where(t_range < 0)[0]
            fz_lvl = p_range[fz_idx[0]] * units.hPa if fz_idx.size > 0 else np.nan * units.hPa
            lcl_h = mpcalc.pressure_to_height_std(lcl_p).m if lcl_p else 0
            lfc_h = mpcalc.pressure_to_height_std(lfc_p).m if lfc_p else np.inf
            el_h = mpcalc.pressure_to_height_std(el_p).m if el_p else 0
            fz_h = mpcalc.pressure_to_height_std(fz_lvl).m if not np.isnan(fz_lvl.m) else 0
            return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
        except: return (units.Quantity(0, 'J/kg'),)*2 + (None, 0, None, np.inf, None, 0, 0)
    
    def calculate_storm_parameters(self):
        try:
            p, u, v = self.p_levels, self.u, self.v
            h_raw = mpcalc.pressure_to_height_std(p).to('meter')
            valid = ~np.isnan(h_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
            if np.sum(valid) < 2: return 0.0, 0.0, 0.0, 0.0
            p_c, u_c, v_c, h_c = p[valid], u[valid], v[valid], h_raw[valid]
            _, idx = np.unique(h_c.m, return_index=True)
            if len(idx) < 2: return 0.0, 0.0, 0.0, 0.0
            p_u, u_u, v_u, h_u = p_c[idx], u_c[idx], v_c[idx], h_c[idx]
            h_min, h_max = h_u.m.min(), min(h_u.m.max(), 16000)
            if h_max <= h_min: return 0.0, 0.0, 0.0, 0.0
            h_i = np.arange(h_min, h_max, 50)*units.meter
            u_i = np.interp(h_i.m, h_u.m, u_u.m)*units('m/s')
            v_i = np.interp(h_i.m, h_u.m, v_u.m)*units('m/s')
            p_i = np.interp(h_i.m, h_u.m, p_u.m)*units.hPa
            u6, v6 = mpcalc.bulk_shear(p_i, u_i, v_i, height=h_i, depth=6000*units.meter)
            s06 = mpcalc.wind_speed(u6, v6).m
            u1, v1 = mpcalc.bulk_shear(p_i, u_i, v_i, height=h_i, depth=1000*units.meter)
            s01 = mpcalc.wind_speed(u1, v1).m
            srh3 = mpcalc.storm_relative_helicity(h_i, u_i, v_i, depth=3000*units.meter)[0].m
            srh1 = mpcalc.storm_relative_helicity(h_i, u_i, v_i, depth=1000*units.meter)[0].m
            return s06, s01, srh3, srh1
        except: return 0.0, 0.0, 0.0, 0.0

    def calculate_flood_risk(self):
        try:
            pwat = mpcalc.precipitable_water(self.p_levels, self.td_profile).to('mm').m
            if pwat > 45: return f"RISC EXTREM D'INUNDACIONS ({pwat:.0f} mm)", "maroon"
            if pwat > 35: return f"RISC ALT D'INUNDACIONS ({pwat:.0f} mm)", "darkred"
            if pwat > 25: return f"RISC MODERAT ({pwat:.0f} mm)", "#DAA520"
            return f"RISC BAIX ({pwat:.0f} mm)", "darkgreen"
        except: return "RISC INDETERMINAT", "darkgray"

    def generate_detailed_analysis(self):
        self.precipitation_type = None; p, t, td = self.p_levels, self.t_profile, self.td_profile; cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = self.calculate_thermo_parameters(); shear_0_6, shear_0_1, srh_0_3, srh_0_1 = self.calculate_storm_parameters(); pwat = mpcalc.precipitable_water(p, td).to('mm').m
        if fz_h < 1500 or t[0].m < 5: text = "--- XAT D'HIVERN ---\n"; text += f"Marc: Iso 0°C?\n> {fz_h:.0f}m. Molt baixa.\n"; text += "Laia: Llavors neu o gel.\n"; text += f"Marc: Humitat en superfície?\n> {mpcalc.relative_humidity_from_dewpoint(t[0], td[0]).m*100:.0f}%. Saturat.\n"; self.precipitation_type = 'snow' if t[0].m <= 0.5 else 'sleet'; text += "Laia: Perfil nival?\n> Sí.\n" if t[0].m <= 0.5 else "Laia: Compte, capa càlida.\n> Risc de pluja gelant.\n"; return text
        elif cape.m > 2000 and shear_0_6 > 15: text = f"--- XAT DE CAÇA (SEVER) ---\nLaia: CAPE?\n> {cape.m:.0f} J/kg. Extrem.\n"; text += f"Marc: CIN?\n> {cin.m:.0f}. Tapa feble.\n"; text += f"Laia: LCL?\n> {lcl_h:.0f} m. Baix. Perfecte.\n"; text += f"Marc: Shear 0-6km?\n> {shear_0_6:.0f} m/s. Excel·lent.\n"; text += f"Laia: SRH 0-3km?\n> {srh_0_3:.0f}. Supercèl·lula probable.\n"; return text
        elif cape.m >= 100: text = f"--- XAT DE TARDA (CAPE: {int(cape.m)}) ---\n"; text += f"Marc: CAPE a {cape.m:.0f}. Moderat.\n"; text += "Laia: Risc principal?\n"; text += "Marc: Calamarsa i ràfegues fortes.\n" if cape.m > 1000 else "Marc: Xàfecs de tarda.\n"; return text
        else: text = "--- XAT DE TEMPS (BONANÇA) ---\n"; text += f"Laia: Alguna cosa?\n> Marc: Res. CAPE a {cape.m:.0f}. Estable.\n"; return text

    def generate_public_warning(self):
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = self.calculate_thermo_parameters(); sfc_temp = self.t_profile[0]
        if fz_h < 1500 or sfc_temp.m < 5:
            if sfc_temp.m <= 0.5: return "AVÍS PER NEU", "Nevada a cotes baixes. Problemes de circulació.", "navy"
            else: return "AVÍS PER PLUJA GEBRADORA", "Risc de glaçades. Precaució.", "dodgerblue"
        elif cape.m >= 1000:
            shear_0_6, shear_0_1, srh_0_3, srh_0_1 = self.calculate_storm_parameters()
            if srh_0_1 > 150 and shear_0_1 > 15: return "AVÍS PER TORNADO", "Condicions per a tornados. Vigileu.", "darkred"
            elif cape.m > 2000: return "AVÍS PER PEDRA", "Tempestes violentes amb pedra grossa.", "purple"
            else: return "AVÍS PER TEMPESTES", "Tempestes fortes amb pluja i calamarsa.", "darkorange"
        else: return "SENSE AVISOS", "Condicions estables.", "green"

    def _draw_cloud_visuals(self):
        real_base_km, real_top_km = self._calculate_dynamic_cloud_heights()
        ground_color = 'white' if self.precipitation_type == 'snow' else '#228B22'
        
        # Dibuix a ax_cloud_drawing
        self.ax_cloud_drawing.clear()
        self.ax_cloud_drawing.add_patch(Rectangle((-1.5, 0), 3, self.ground_height_km, color=ground_color, zorder=3))
        if real_base_km and real_top_km and real_top_km > real_base_km:
            self.ax_cloud_drawing.add_patch(Rectangle((-1, real_base_km), 2, real_top_km - real_base_km, facecolor='lightgray', alpha=0.8, zorder=10))
        self.ax_cloud_drawing.set_xlim(-2, 2)
        self.ax_cloud_drawing.set_ylim(0, 15)
        self.ax_cloud_drawing.axis('off')

        # Dibuix a ax_cloud_structure
        self.ax_cloud_structure.clear()
        self.ax_cloud_structure.add_patch(Rectangle((-1.5, 0), 3, self.ground_height_km, color=ground_color, zorder=1))
        if real_base_km and real_top_km and real_top_km > real_base_km:
            self.ax_cloud_structure.add_patch(Polygon([(-0.5, real_base_km), (0.5, real_base_km), (0, real_top_km)], facecolor='lightgray', alpha=0.8, zorder=10))
        self.ax_cloud_structure.set_xlim(-2, 2)
        self.ax_cloud_structure.set_ylim(0, 15)
        self.ax_cloud_structure.axis('off')

    def _calculate_dynamic_cloud_heights(self):
        _, _, lcl_p, lcl_h, _, _, el_p, el_h, _ = self.calculate_thermo_parameters()
        if not lcl_p: return None, None
        cloud_base_km = lcl_h / 1000.0
        if self.convergence_active:
            cloud_top_km = el_h / 1000.0 if el_h > lcl_h else cloud_base_km
        else:
            rh = mpcalc.relative_humidity_from_dewpoint(self.t_profile, self.td_profile)
            indices = np.where(self.p_levels <= lcl_p)[0]
            p_top = self.p_levels[-1]
            if len(indices) > 0:
                for idx in indices:
                    if rh[idx] < 0.5:
                        p_top = self.p_levels[idx]
                        break
            cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
        return (cloud_base_km, cloud_top_km) if cloud_top_km > cloud_base_km else (None, None)

    def _clear_axes(self):
        for ax in [self.ax, self.ax_public_warning, self.ax_info_panel, self.ax_radar_sim, self.ax_cloud_drawing, self.ax_cloud_structure, self.ax_shear_barbs, self.ax_cloud_label, self.time_text_ax]:
            ax.cla()
        self.ax_public_warning.axis('off'); self.ax_info_panel.axis('off'); self.ax_cloud_label.axis('off'); self.time_text_ax.axis('off')

    def update_plot(self):
        self._clear_axes()
        try:
            self.setup_plot()
            self.td_profile = np.minimum(self.t_profile, self.td_profile)
            self.line_t.set_data(self.t_profile, self.p_levels); self.line_td.set_data(self.td_profile, self.p_levels)
            parcel_prof = mpcalc.parcel_profile(self.p_levels, self.t_profile[0], self.td_profile[0]).to('degC')
            self.line_parcel.set_data(parcel_prof, self.p_levels)
            self.line_wb.set_data(mpcalc.wet_bulb_temperature(self.p_levels, self.t_profile, self.td_profile), self.p_levels)
            _, _, lcl_p, _, lfc_p, _, el_p, _, _ = self.calculate_thermo_parameters()
            xlims = self.ax.get_xlim()
            for line, p_val in [(self.line_lcl, lcl_p), (self.line_lfc, lfc_p), (self.line_el, el_p)]:
                if p_val: line.set_data(xlims, [p_val.m, p_val.m])
                else: line.set_data([], [])
            
            self.skew.shade_cape(self.p_levels, self.t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
            self.skew.shade_cin(self.p_levels, self.t_profile, parcel_prof, facecolor='black', alpha=0.3)
            
            self.update_ground_patch()
            self._draw_cloud_visuals()

            # Mostrar avís públic
            warning_title, warning_msg, warning_color = self.generate_public_warning()
            self.ax_public_warning.text(0.5, 0.7, warning_title, ha='center', va='center', fontsize=14, weight='bold', color=warning_color, transform=self.ax_public_warning.transAxes)
            self.ax_public_warning.text(0.5, 0.3, warning_msg, ha='center', va='center', fontsize=10, color='black', transform=self.ax_public_warning.transAxes)
            self.ax_public_warning.set_facecolor('lightgray')

            # Mostrar temps d'observació
            self.time_text_ax.text(0.5, 0.5, self.observation_time, ha='center', va='center', fontsize=10, color='black')

        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7), color='white', transform=self.ax.transAxes)

# ==============================================================================
# SECCIÓ 2: INTERFÍCIE D'USUARI AMB STREAMLIT
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
    selected_file = st.selectbox("Selecciona un arxiu:", existing_files, key="file_selector")
    
    # Llegir dades i gestionar la sessió
    if 'current_file' not in st.session_state or st.session_state.current_file != selected_file:
        st.session_state.current_file = selected_file
        sounding_data = parse_all_soundings(selected_file)
        if not sounding_data:
            st.error(f"L'arxiu '{selected_file}' és invàlid."); st.stop()
        st.session_state.sounding_data = sounding_data[0]
        st.session_state.skewt_instance = AdvancedSkewT(st.session_state.sounding_data)
    
    skewt_instance = st.session_state.skewt_instance
    default_pressure = int(st.session_state.sounding_data['p_levels'][0].magnitude)
    
    surface_p = st.number_input("Pressió en superfície (hPa):", 850, 1050, default_pressure, 1)
    convergence = st.toggle("Activar convergència (tempestes)", value=True)

# Actualitzar la instància amb els valors dels controls
skewt_instance.convergence_active = convergence
if surface_p != int(skewt_instance.current_surface_pressure.magnitude):
    skewt_instance.current_surface_pressure = surface_p * units.hPa
    skewt_instance.adjust_profiles_to_new_surface()
    skewt_instance._force_wind_update()

# Executar tota la lògica de dibuix i anàlisi
with st.spinner("Generant visualització detallada..."):
    skewt_instance.update_plot()
    
    risk_text, risk_color = skewt_instance.calculate_flood_risk()
    public_warning_title, public_warning_msg, public_warning_color = skewt_instance.generate_public_warning()
    analysis_text = skewt_instance.generate_detailed_analysis()

# Presentació dels resultats a la pàgina principal
st.markdown(f"<h2 style='text-align: center; color: white; background-color:{risk_color}; padding: 10px; border-radius: 5px;'>{risk_text}</h2>", unsafe_allow_html=True)
st.markdown(f"**Font:** `{selected_file}` | **Hora:** `{skewt_instance.observation_time}`")

col1, col2 = st.columns([3, 1])
with col1:
    st.pyplot(skewt_instance.fig, use_container_width=True)
with col2:
    st.markdown(f"<div style='background-color:{public_warning_color}; color:white; padding:15px; border-radius:10px;'><h3 style='margin-top:0;'>{public_warning_title}</h3><p>{public_warning_msg}</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("🕵️‍♂️ Xat d'Anàlisi Tècnica", expanded=True):
        st.code(analysis_text)
