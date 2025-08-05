# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from scipy.interpolate import interp1d
import os
import re

# ==============================================================================
# SECCIÓ 1: CÀRREGA DE DADES (Sense canvis)
# ==============================================================================

@st.cache_data(show_spinner="Llegint arxiu de sondeig...")
def parse_sounding_file(filepath):
    """
    Llegeix un fitxer de text que pot contenir múltiples sondejos i retorna
    el PRIMER sondeig vàlid que trobi.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    def clean_and_convert(text):
        cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
        if not cleaned_text or cleaned_text == '-': return None
        try: return float(cleaned_text)
        except ValueError: return None
    
    current_sounding_lines = []
    for line in lines:
        if 'Pression' in line and (line.strip().startswith('Nivell') or line.strip().startswith('# Nivell')):
            if current_sounding_lines:
                break # Hem trobat el final del primer sondeig
        current_sounding_lines.append(line)

    if not current_sounding_lines: return None

    p_list, t_list, td_list, wdir_list, wspd_list, time_lines = [], [], [], [], [], []
    time_keywords = ['observació', 'hora', 'time', 'locale', 'run', 'z', 'date']

    for line in current_sounding_lines:
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
            wind_str = parts[6].strip()
            if '/' in wind_str:
                wind_parts = wind_str.split('/')
                wdir, wspd = (clean_and_convert(wind_parts[0]) or 0.0), (clean_and_convert(wind_parts[1]) or 0.0)
            else:
                wdir, wspd = 0.0, 0.0
            wdir_list.append(wdir); wspd_list.append(wspd)
        except Exception: continue
    
    if not p_list or len(p_list) < 2: return None
    
    observation_time = "\n".join(time_lines) if time_lines else "Hora no disponible"
    sorted_indices = np.argsort(p_list)[::-1]
    
    return {
        'p_levels': np.array(p_list)[sorted_indices] * units.hPa,
        't_initial': np.array(t_list)[sorted_indices] * units.degC,
        'td_initial': np.array(td_list)[sorted_indices] * units.degC,
        'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph,
        'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees,
        'observation_time': observation_time
    }

# ==============================================================================
# SECCIÓ 2: MOTOR D'ANÀLISI (Lògica de càlcul sense canvis)
# ==============================================================================

class SoundingAnalyzer:
    def __init__(self, sounding_data):
        self.original_p = sounding_data['p_levels'].copy()
        self.original_t = sounding_data['t_initial'].copy()
        self.original_td = sounding_data['td_initial'].copy()
        self.original_ws = sounding_data['wind_speed_kmh'].to('m/s') if sounding_data.get('wind_speed_kmh') is not None else np.zeros_like(self.original_p.magnitude) * units('m/s')
        self.original_wd = sounding_data['wind_dir_deg'] if sounding_data.get('wind_dir_deg') is not None else np.zeros_like(self.original_p.magnitude) * units.degrees
        self.observation_time = sounding_data.get('observation_time', 'Hora no disponible')
        self.params = {}

    def run_analysis(self, surface_p_hpa, convergence):
        """Ajusta perfils i calcula tots els paràmetres meteorològics."""
        # 1. Ajustar perfils a la pressió de superfície
        p_surf = surface_p_hpa * units.hPa
        mask = self.original_p <= p_surf
        p_masked, t_masked, td_masked, ws_masked, wd_masked = (self.original_p[mask], self.original_t[mask], self.original_td[mask], self.original_ws[mask], self.original_wd[mask])
        
        self.p = np.concatenate(([p_surf], p_masked[p_masked < p_surf]))
        
        f_t = interp1d(self.original_p.m, self.original_t.m, fill_value="extrapolate")
        self.t = np.concatenate(([f_t(p_surf.m) * units.degC], t_masked[p_masked < p_surf]))
        
        f_td = interp1d(self.original_p.m, self.original_td.m, fill_value="extrapolate")
        self.td = np.concatenate(([f_td(p_surf.m) * units.degC], td_masked[p_masked < p_surf]))
        self.td = np.minimum(self.t, self.td) # Assegurem consistència física
        
        f_ws = interp1d(self.original_p.m, self.original_ws.to('m/s').m, fill_value="extrapolate")
        ws_new = np.concatenate(([f_ws(p_surf.m) * units('m/s')], ws_masked[p_masked < p_surf]))
        
        f_wd = interp1d(self.original_p.m, self.original_wd.m, fill_value="extrapolate")
        wd_new = np.concatenate(([f_wd(p_surf.m) * units.degrees], wd_masked[p_masked < p_surf]))

        self.u, self.v = mpcalc.wind_components(ws_new, wd_new)

        # 2. Calcular paràmetres
        self.parcel_prof = mpcalc.parcel_profile(self.p, self.t[0], self.td[0]).to('degC')
        self.params['cape'], self.params['cin'] = mpcalc.cape_cin(self.p, self.t, self.td, self.parcel_prof)
        self.params['lcl_p'], self.params['lcl_t'] = mpcalc.lcl(self.p[0], self.t[0], self.td[0])
        self.params['lfc_p'], self.params['lfc_t'] = mpcalc.lfc(self.p, self.t, self.td, self.parcel_prof)
        self.params['el_p'], self.params['el_t'] = mpcalc.el(self.p, self.t, self.td, self.parcel_prof)
        self.params['pwat'] = mpcalc.precipitable_water(self.p, self.td)

        # Altures
        try:
            # Utilitzem l'array complet d'altures per a més precisió
            all_heights = mpcalc.pressure_to_height_std(self.p)
            self.params['lcl_h'] = mpcalc.pressure_to_height_std(self.params['lcl_p']).to('m') if self.params.get('lcl_p') else 0 * units.m
            self.params['el_h'] = mpcalc.pressure_to_height_std(self.params['el_p']).to('m') if self.params.get('el_p') else self.params['lcl_h']
        except Exception:
            self.params['lcl_h'] = 0 * units.m
            self.params['el_h'] = 0 * units.m

        if convergence:
            self.params['cloud_base'] = self.params['lcl_h']
            self.params['cloud_top'] = self.params['el_h']
        else:
            saturated_mask = (self.t - self.td).m < 2.0
            if np.any(saturated_mask):
                p_saturated = self.p[saturated_mask]
                h_saturated = mpcalc.pressure_to_height_std(p_saturated)
                self.params['cloud_base'] = np.max(h_saturated)
                self.params['cloud_top'] = np.min(h_saturated)
            else:
                self.params['cloud_base'] = self.params['cloud_top'] = 0 * units.m

        # Paràmetres de cisallament
        try:
            # Assegurem que les altures cobreixen la capa 0-6km
            height_for_shear = mpcalc.pressure_to_height_std(self.p) - mpcalc.pressure_to_height_std(self.p[0])
            shear_u, shear_v = mpcalc.bulk_shear(self.p, self.u, self.v, height=height_for_shear, depth=6000 * units.meter)
            self.params['shear_0_6_kt'] = mpcalc.wind_speed(shear_u, shear_v).to('knots')
        except Exception:
            self.params['shear_0_6_kt'] = 0 * units.knots

    def plot_skewt_and_structure(self):
        """
        Dibuixa el Skew-T i una representació de l'estructura de núvols
        que s'inclina segons la cisalladura del vent.
        """
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0.1)
        
        # Gràfic Skew-T
        skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
        ax = skew.ax
        ax.set_ylim(1050, 150)
        ax.set_xlim(-40, 50)
        skew.plot_dry_adiabats(alpha=0.3)
        skew.plot_moist_adiabats(alpha=0.3)
        skew.plot_mixing_lines(alpha=0.4)
        skew.plot(self.p, self.t, 'r', linewidth=2, label='Temperatura')
        skew.plot(self.p, self.td, 'b', linewidth=2, label='Punt de Rosada')
        if hasattr(self, 'parcel_prof'):
            skew.plot(self.p, self.parcel_prof, 'k--', linewidth=2, label='Parcel·la')
            skew.shade_cape(self.p, self.t, self.parcel_prof, facecolor='orange', alpha=0.4)
            skew.shade_cin(self.p, self.t, self.parcel_prof, facecolor='lightblue', alpha=0.4)
        ax.axvline(0, color='c', linestyle='--', linewidth=1) # Isoterma 0ºC
        skew.plot_barbs(self.p, self.u, self.v, xloc=1.05, length=7, linewidth=0.8) # Barbes de vent
        ax.legend()
        
        # Gràfic d'estructura de núvols
        ax_structure = fig.add_subplot(gs[0, 1]) # No compartim 'y' per evitar problemes d'escala
        ax_structure.set_facecolor('skyblue')
        ax_structure.set_xticks([])
        ax_structure.set_title("Estructura Inclinada", fontsize=10)
        ax_structure.set_xlabel("Cisalladura", fontsize=9)
        ax_structure.set_xlim(-0.5, 1.5) # Eix X més ample per a la inclinació
        ax_structure.tick_params(axis='y', labelleft=False, labelright=True) # Etiquetes d'altura a la dreta
        ax_structure.yaxis.set_label_position("right")
        ax_structure.set_ylabel("Altura (km)")

        # Convertim pressió de l'eix Y a altura en km per al nou gràfic
        p_ticks_hpa = np.array([1000, 850, 700, 500, 400, 300, 200, 150])
        h_ticks_km = mpcalc.pressure_to_height_std(p_ticks_hpa * units.hPa).to('km').m
        ax_structure.set_yticks(h_ticks_km)
        ax_structure.set_ylim(h_ticks_km.min(), h_ticks_km.max())
        ax_structure.grid(axis='y', linestyle='--', alpha=0.5, color='white')

        # Dibuixar terra
        h_surf_km = mpcalc.pressure_to_height_std(self.p[0]).to('km').m
        ax_structure.add_patch(Rectangle((-0.5, 0), 2, h_surf_km, color='saddlebrown', zorder=2))

        # Dibuixar núvol inclinat
        base_h_km = self.params.get('cloud_base', 0 * units.m).to('km').m
        top_h_km = self.params.get('cloud_top', 0 * units.m).to('km').m
        shear_kt = self.params.get('shear_0_6_kt', 0 * units.knots).m
        
        if top_h_km > base_h_km:
            # Càlcul del desplaçament: més cisalladura = més inclinació
            # Aquest factor és empíric, pots ajustar-lo per a l'efecte visual desitjat
            shear_factor = 0.02 
            shear_offset = shear_kt * shear_factor

            # Definim els vèrtexs del polígon (un trapezi)
            # (x, y)
            cloud_verts = [
                (0.0, base_h_km),                      # Base esquerra
                (1.0, base_h_km),                      # Base dreta
                (1.0 + shear_offset, top_h_km),        # Cim dreta (desplaçat)
                (0.0 + shear_offset, top_h_km)         # Cim esquerra (desplaçat)
            ]
            
            cloud_patch = Polygon(cloud_verts, color='white', alpha=0.8, edgecolor='gray', zorder=3)
            ax_structure.add_patch(cloud_patch)
            
        return fig


# ==============================================================================
# SECCIÓ 3: INTERFÍCIE D'USUARI AMB STREAMLIT (Sense canvis)
# ==============================================================================

st.set_page_config(layout="wide", page_title="SondeigCat Pro")

st.title("SondeigCat Pro (Versió Millorada)")
st.markdown("Anàlisi ràpida de sondejos atmosfèrics amb visualització de cisalladura.")

# --- Càrrega i selecció d'arxius ---
AVAILABLE_FILES = [f"sondeig{i}.txt" for i in ["", "1", "2", "3", "4", "5"]]
existing_files = [file for file in AVAILABLE_FILES if os.path.exists(file)]

if not existing_files:
    st.error("Error: No s'han trobat arxius de sondeig (`sondeig.txt`, etc.). Assegura't que estiguin al repositori de GitHub.")
    st.stop()

# --- Controls a la barra lateral ---
with st.sidebar:
    st.header("⚙️ Controls")
    selected_file = st.selectbox("Selecciona un sondeig:", existing_files)
    
    # Llegir dades del fitxer
    sounding_data = parse_sounding_file(selected_file)
    if not sounding_data:
        st.error(f"L'arxiu '{selected_file}' no s'ha pogut processar.")
        st.stop()

    # Valors per defecte basats en el fitxer
    default_pressure = int(sounding_data['p_levels'][0].magnitude)

    surface_p = st.number_input("Pressió en superfície (hPa):", min_value=850, max_value=1050, value=default_pressure, step=1)
    convergence = st.toggle("Activar convergència (per tempestes)", value=True)

# --- Anàlisi i presentació ---
analyzer = SoundingAnalyzer(sounding_data)
analyzer.run_analysis(surface_p, convergence)
params = analyzer.params

st.markdown(f"**Font:** `{selected_file}` | **Hora:** `{analyzer.observation_time}`")

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.pyplot(analyzer.plot_skewt_and_structure(), use_container_width=True)

with col2:
    st.subheader("Paràmetres Clau")
    
    m1, m2 = st.columns(2)
    m1.metric("CAPE (J/kg)", f"{params['cape'].m:.0f}")
    m2.metric("CIN (J/kg)", f"{params['cin'].m:.0f}")
    
    m3, m4 = st.columns(2)
    m3.metric("PWAT (mm)", f"{params['pwat'].to('mm').m:.1f}", help="Aigua Precipitable")
    m4.metric("Shear 0-6km (kt)", f"{params['shear_0_6_kt'].m:.1f}", help="Cisallament del vent")

    with st.expander("Tots els Paràmetres", expanded=False):
        st.table({
            "Paràmetre": ["CAPE", "CIN", "PWAT", "LCL", "LFC", "EL", "Shear 0-6km"],
            "Valor": [
                f"{params['cape'].m:.0f} J/kg",
                f"{params['cin'].m:.0f} J/kg",
                f"{params['pwat'].to('mm').m:.1f} mm",
                f"{params['lcl_p'].m:.0f} hPa" if params.get('lcl_p') else "N/A",
                f"{params['lfc_p'].m:.0f} hPa" if params.get('lfc_p') else "N/A",
                f"{params['el_p'].m:.0f} hPa" if params.get('el_p') else "N/A",
                f"{params['shear_0_6_kt'].m:.1f} kt"
            ]
        })

    # Diagnòstic simplificat en lloc del xat
    st.subheader("Diagnòstic Ràpid")
    cape_val = params['cape'].m
    if cape_val > 2500:
        st.error("🔴 **EXTREM:** Condicions per a tempestes molt severes (possible pedra grossa, esclafits).")
    elif cape_val > 1500:
        st.warning("🟠 **ALT:** Condicions per a tempestes fortes (calamarsa, fortes ratxes de vent).")
    elif cape_val > 500:
        st.info("🟡 **MODERAT:** Potencial per a tronades i xàfecs forts.")
    elif cape_val > 100:
        st.success("🟢 **BAIX:** Possibilitat de xàfecs aïllats i febles.")
    else:
        st.success("✅ **ESTABLE:** No s'espera convecció significativa.")

st.sidebar.markdown("---")
st.sidebar.info("Versió millorada amb visualització de cisalladura.")
