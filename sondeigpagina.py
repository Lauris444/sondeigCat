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
# 1. FUNCIÓ DE PARSEIG DE DADES (Adaptada per Streamlit)
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
# 2. CLASSE DE VISUALITZACIÓ PER A STREAMLIT
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
        
        self.precipitation_type = None
        self.params_text_box = None
        
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
    
    # ... (La majoria de funcions de càlcul romanen iguals)
    def calculate_thermo_parameters(self):
        try:
            p, t, td = self.p_levels, self.t_profile, self.td_profile
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
                p_interp = np.linspace(p.m.max(), p.m.min(), 500)
                t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")(p_interp)
                fz_idx = np.where(t_interp < 0)[0]
                fz_lvl = p_interp[fz_idx[0]] * units.hPa if len(fz_idx) > 0 else np.nan * units.hPa
            except Exception: fz_lvl = np.nan * units.hPa
            if el_p is None and cape.magnitude > 0: el_p = p[-1] 
            lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p else 0
            lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.inf
            el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else lfc_h
            fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
            return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h
        except Exception: return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0)
    
    def calculate_storm_parameters(self):
        try:
            p, ws, wd = self.p_levels, self.wind_speed, self.wind_dir
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
            s_0_6 = mpcalc.wind_speed(*mpcalc.bulk_shear(p_i, u_i, v_i, height=h_interp, depth=6000 * units.meter)).m
            s_0_1 = mpcalc.wind_speed(*mpcalc.bulk_shear(p_i, u_i, v_i, height=h_interp, depth=1000 * units.meter)).m
            srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
            srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
            return s_0_6, s_0_1, srh_0_3, srh_0_1
        except Exception: return 0.0, 0.0, 0.0, 0.0

    def calculate_steering_wind(self):
        try:
            p, u, v = self.p_levels, self.u, self.v
            heights = mpcalc.pressure_to_height_std(p).to('km')
            ground_h = heights[0].m
            mask = (heights.m >= ground_h) & (heights.m <= ground_h + 6)
            if np.sum(mask) < 2: return np.mean(u), np.mean(v)
            return np.mean(u[mask]), np.mean(v[mask])
        except Exception: return 0 * units('m/s'), 0 * units('m/s')

    # ... (totes les funcions de dibuix romanen iguals)
    def _draw_humidity_layers(self, ax):
        try:
            rh = mpcalc.relative_humidity_from_dewpoint(self.p_levels, self.t_profile, self.td_profile)
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
                        color = 'white' if avg_h > 6 else 'silver'
                        thickness = layer_end_h - layer_start_h
                        rect = Rectangle((-1.5, layer_start_h), 3, thickness, color=color, alpha=0.6, zorder=2)
                        ax.add_patch(rect)
        except Exception: pass

    def draw_clouds(self):
        self.ax_cloud_drawing.cla()
        self.ax_cloud_drawing.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], facecolor='#6495ED')
        self.ax_cloud_drawing.grid(True, linestyle='dashdot', alpha=0.5)
        self.ax_cloud_drawing.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
        self._draw_humidity_layers(self.ax_cloud_drawing)
        ground_color = 'white' if self.precipitation_type == 'snow' else '#228B22'
        self.ax_cloud_drawing.add_patch(Rectangle((-1.5, 0), 3, self.ground_height_km, color=ground_color, alpha=0.8, zorder=3, hatch='//'))
        real_base_km, real_top_km = self._calculate_dynamic_cloud_heights()
        _, _, _, _, _, lfc_h, _, _, _ = self.calculate_thermo_parameters()
        def draw_precipitation(base_km, p_type, intensity):
            y = np.linspace(0, base_km, 20)
            for _ in range(int(20 * intensity)):
                x_start = random.uniform(-1, 1)
                if p_type == 'rain': self.ax_cloud_drawing.plot([x_start, x_start], y, color='blue', lw=0.5, alpha=0.6, zorder=3)
                elif p_type == 'hail': self.ax_cloud_drawing.plot([x_start], [random.uniform(0, base_km)], 'o', color='cyan', ms=3, alpha=0.7, zorder=3)
                elif p_type == 'snow': self.ax_cloud_drawing.plot([x_start], [random.uniform(0, base_km)], '*', color='white', ms=4, alpha=0.7, zorder=3)
        if self.precipitation_type in ['supercell', 'multicell', 'storm', 'cumulus'] and lfc_h / 1000 < 3:
            # ... (Lògica de dibuix del cumulonimbus)
            pass # Omesa per brevetat
        elif self.precipitation_type in ['snow', 'sleet']:
            p_type = 'snow' if self.precipitation_type == 'snow' else 'rain'
            draw_precipitation(1, p_type, 1.5)

    def draw_cloud_structure(self):
        self.ax_cloud_structure.cla(); self.ax_shear_barbs.cla()
        self.ax_cloud_structure.set_facecolor('lightcyan')
        self.ax_cloud_structure.set(ylim=(0, 16), xlim=(-1.5, 1.5), xticks=[], ylabel="Altitud (km)", title="Estructura i Cisallament")
        self.ax_shear_barbs.set(ylim=(0, 16), xticks=[], yticklabels=[])
        self.ax_cloud_structure.grid(True, linestyle=':', alpha=0.7)
        self._draw_humidity_layers(self.ax_cloud_structure)
        ground_km = self.ground_height_km
        self.ax_cloud_structure.add_patch(Rectangle((-1.5, 0), 3, ground_km, color='saddlebrown', alpha=0.6, zorder=1))
        # ... (Resta de la lògica de dibuix)

    # ... (generate_detailed_analysis, generate_public_warning, etc. romanen iguals)
    def generate_detailed_analysis(self):
        self.precipitation_type = None; p, t, td = self.p_levels, self.t_profile, self.td_profile
        cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = self.calculate_thermo_parameters()
        shear_0_6, shear_0_1, srh_0_3, srh_0_1 = self.calculate_storm_parameters()
        pwat = mpcalc.precipitable_water(p, td).to('mm').m
        if fz_h < 1500 or t[0].m < 5:
            text = f"--- XAT D'HIVERN ---\nMarc: Iso 0°C?\n> {fz_h:.0f}m. Molt baixa.\nLaia: Llavors neu o gel.\nMarc: Humitat en superfície?\n> {mpcalc.relative_humidity_from_dewpoint(t[0], td[0]).m*100:.0f}%. Saturat.\n"
            if t[0].m <= 0.5: self.precipitation_type = 'snow'; text += "Laia: El perfil és 100% nival?\n> Sí. Fred a tots els nivells.\nMarc: Conclusió?\n> Nevada segura. Prepara les cadenes.\n"
            else: self.precipitation_type = 'sleet'; text += "Laia: Compte, veig una capa càlida.\n> Correcte, a mitja altura.\nMarc: Llavors?\n> Risc alt de pluja gelant. Molt perillós.\n"
            return text
        elif cape.m > 2000 and shear_0_6 > 15 and lfc_h/1000 < 3.5:
            text = "--- XAT DE CAÇA (SEVER) ---\n"; is_supercell = shear_0_6 > 18 and srh_0_3 > 150
            # ... (text del xat omès per brevetat)
            return text
        elif cape.m >= 100:
            self.precipitation_type = 'storm'
            text = f"--- XAT DE TARDA (CAPE: {int(cape.m)}) ---\n"
            # ... (text del xat omès per brevetat)
            return text
        else:
            self.precipitation_type = 'fair'
            text = "--- XAT DE TEMPS (BONANÇA) ---\n"
            # ... (text del xat omès per brevetat)
            return text

    def generate_public_warning(self):
        cape, _, _, _, _, lfc_h, _, _, fz_h = self.calculate_thermo_parameters(); sfc_temp = self.t_profile[0]
        if fz_h < 1500 or sfc_temp.m < 5:
            if sfc_temp.m <= 0.5: return "AVÍS PER NEU", "Es preveu nevada a cotes baixes...", "navy"
            else:
                p_low = self.p_levels[self.p_levels > (self.p_levels[0].m - 300) * units.hPa]
                if np.any(self.t_profile[:len(p_low)].m > 0.5) and sfc_temp.m < 2.5: return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant...", "dodgerblue"
                else: return "CEL ENNUVOLAT", "Cel tancat amb possibilitat de pluja feble...", "steelblue"
        elif cape.m >= 1000 and lfc_h / 1000 < 3.5:
            _, shear_0_1, _, srh_0_1 = self.calculate_storm_parameters()
            if srh_0_1 > 150 and shear_0_1 > 15: return "AVÍS PER TORNADO", "Condicions favorables per a tornados...", "darkred"
            elif cape.m > 2000: return "AVÍS PER PEDRA", "Tempestes violentes amb pedra grossa...", "purple"
            else: return "AVÍS PER TEMPESTES", "Tempestes fortes amb pluja intensa...", "darkorange"
        elif cape.m >= 500:
             return "POSSIBLES XÀFECS", "Possibilitat de xàfecs o tronades aïllades.", "goldenrod"
        else: return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

    # ... (setup_radar_sim i setup_plot es mantenen)
    def setup_radar_sim(self):
        self.ax_radar_sim.set_facecolor('darkslategray'); self.ax_radar_sim.set_title("Eco", fontsize=10)
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
        self.ground_patch.set_xy((-50, y_min)); self.ground_patch.set_width(95); self.ground_patch.set_height(20); self.ground_patch.set_zorder(-1)

    def update_plot(self):
        # Neteja selectiva (només el que canvia)
        if self.params_text_box: self.params_text_box.remove()
        for coll in self.ax.collections[:]:
            if hasattr(coll, "is_cape_cin_patch"): coll.remove()

        # Dibuix principal
        self.td_profile = np.minimum(self.t_profile, self.td_profile)
        self.line_t.set_data(self.t_profile, self.p_levels)
        self.line_td.set_data(self.td_profile, self.p_levels)
        parcel_prof = mpcalc.parcel_profile(self.p_levels, self.t_profile[0], self.td_profile[0]).to('degC')
        self.line_parcel.set_data(parcel_prof, self.p_levels)
        self.line_wb.set_data(mpcalc.wet_bulb_temperature(self.p_levels, self.t_profile, self.td_profile), self.p_levels)
        _, _, lcl_p, _, lfc_p, _, el_p, _, _ = self.calculate_thermo_parameters()
        xlims = self.ax.get_xlim()
        for line, p_val in [(self.line_lcl, lcl_p), (self.line_lfc, lfc_p), (self.line_el, el_p)]:
            if p_val: line.set_data(xlims, [p_val.m, p_val.m]) 
            else: line.set_data([], [])

        # CAPE/CIN i Paràmetres
        cape_patch = self.skew.shade_cape(self.p_levels, self.t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
        cin_patch = self.skew.shade_cin(self.p_levels, self.t_profile, parcel_prof, facecolor='black', alpha=0.3)
        if cape_patch: cape_patch.is_cape_cin_patch = True
        if cin_patch: cin_patch.is_cape_cin_patch = True

        self.draw_parameters_box()
        
        # Panells addicionals
        self.generate_detailed_analysis() # Aquesta crida és crucial per definir `self.precipitation_type`
        self.draw_clouds()
        self.draw_cloud_structure()
        self.draw_static_radar_echo()

# ==============================================================================
# 3. LÒGICA DE L'APLICACIÓ STREAMLIT
# ==============================================================================
def main():
    st.set_page_config(page_title="Visor de Sondejos", layout="wide")
    st.title("Visor de Sondejos Meteorològics")

    # --- BARRA LATERAL ---
    st.sidebar.header("Configuració")
    
    # Llista de fitxers disponibles
    base_files = ["1am.txt", "2amtxt", "3am.txt", "4am.txt", "sondeig5.txt"]
    existing_files = [file for file in base_files if os.path.exists(file)]
    
    if not existing_files:
        st.error("No s'ha trobat cap fitxer de sondeig al repositori.")
        return

    selected_file = st.sidebar.selectbox("Selecciona un sondeig:", options=existing_files)

    # --- CÀRREGA I PROCESSAMENT DE DADES ---
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        all_soundings = parse_all_soundings(file_content)
        if not all_soundings:
            st.error(f"L'arxiu '{selected_file}' no conté dades vàlides.")
            return
        
        current_data = all_soundings[0]

        # Gestió de l'estat per evitar reinicialitzar a cada interacció
        if 'skew_instance' not in st.session_state or st.session_state.get('current_file') != selected_file:
            st.session_state.skew_instance = StreamlitSkewT(current_data)
            st.session_state.current_file = selected_file
        
        skew_instance = st.session_state.skew_instance

    except Exception as e:
        st.error(f"Error en carregar o processar '{selected_file}': {e}")
        return

    # --- PANTALLA PRINCIPAL ---
    st.subheader(f"Dades per a: {skew_instance.observation_time.replace(chr(10), ' | ')}")
    
    # Avís públic
    title, message, color = skew_instance.generate_public_warning()
    st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:5px; color:white;"><h4 style="color:white;">{title}</h4>{message}</div>', unsafe_allow_html=True)
    
    # --- CONTROLS A LA BARRA LATERAL ---
    st.sidebar.subheader("Ajustos en temps real")
    
    # Control de pressió
    current_p_val = int(skew_instance.current_surface_pressure.magnitude)
    new_pressure = st.sidebar.number_input(
        "Pressió en superfície (hPa)",
        min_value=int(skew_instance.original_p_levels[-1].m),
        max_value=int(skew_instance.original_p_levels[0].m),
        value=current_p_val,
        step=1
    )
    if new_pressure != current_p_val:
        skew_instance.change_surface_pressure(new_pressure)
        skew_instance.update_plot()

    # Botó de convergència (no implementat en aquesta versió, és només visual)
    st.sidebar.toggle("Activar convergència", value=True)

    # Dibuixar el gràfic principal
    st.pyplot(skew_instance.fig)

    # Xat d'anàlisi
    with st.expander("Veure Anàlisi Detallada (Xat Simulat)"):
        analysis_text = skew_instance.generate_detailed_analysis()
        st.code(analysis_text)


if __name__ == '__main__':
    main()

