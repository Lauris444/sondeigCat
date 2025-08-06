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
import time
import threading
import base64
import io

integrator_lock = threading.Lock()

# =============================================================================
# === 1. PROCESSAMENT DE DADES ==============================================
# =============================================================================
def parse_all_soundings(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: return []
    
    all_soundings, current_lines = [], []
    
    def clean_and_convert(text):
        cleaned = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
        try: return float(cleaned)
        except (ValueError, TypeError): return None

    for line in lines:
        if 'Pression' in line and (line.strip().startswith(('Nivell', '# Nivell'))):
            if current_lines:
                data = process_sounding_block(current_lines)
                if data: all_soundings.append(data)
            current_lines = []
        current_lines.append(line)
    if current_lines:
        data = process_sounding_block(current_lines)
        if data: all_soundings.append(data)
    return all_soundings

def process_sounding_block(block_lines):
    p_list, t_list, td_list, wdir_list, wspd_list, time_lines = [], [], [], [], [], []
    time_keywords = ['observació', 'hora', 'time', 'locale', 'run', 'z', 'date']
    fr_to_ca_days = {'Lundi': 'Dilluns', 'Mardi': 'Dimarts', 'Mercredi': 'Dimecres', 'Jeudi': 'Dijous', 'Vendredi': 'Divendres', 'Samedi': 'Dissabte', 'Dimanche': 'Diumenge'}
    fr_to_ca_months = {'janvier': 'de gener', 'février': 'de febrer', 'mars': 'de març', 'avril': 'd\'abril', 'mai': 'de maig', 'juin': 'de juny', 'juillet': 'de juliol', 'août': 'd\'agost', 'septembre': 'de setembre', 'octobre': 'd\'octubre', 'novembre': 'de novembre', 'décembre': 'de desembre'}
    
    def clean_and_convert(text):
        cleaned = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
        try: return float(cleaned)
        except (ValueError, TypeError): return None

    for line in block_lines:
        line_strip = line.strip()
        if any(kw in line_strip.lower() for kw in time_keywords) and not (line_strip and line_strip[0].isdigit()):
            time_lines.append(line_strip)
            continue
        if not line_strip or line_strip.startswith(('#', 'Pression')): continue
        try:
            parts = re.split(r'\s{2,}|[\t]', line_strip)
            if len(parts) < 7: continue
            p, t, td = clean_and_convert(parts[1]), clean_and_convert(parts[2]), clean_and_convert(parts[4])
            if p is None or t is None or td is None: continue
            p_list.append(p); t_list.append(t); td_list.append(td)
            wdir_val, wspd_val = 0.0, 0.0
            if '/' in parts[6]:
                wind_parts = parts[6].strip().split('/')
                if len(wind_parts) == 2:
                    wdir_val = clean_and_convert(wind_parts[0]) or 0.0
                    wspd_val = clean_and_convert(wind_parts[1]) or 0.0
            wdir_list.append(wdir_val); wspd_list.append(wspd_val)
        except Exception: continue
    if not p_list or len(p_list) < 2: return None
    
    translated_lines = []
    for line in time_lines:
        translated = line.replace('Run', 'Model').replace('locale', 'local').replace('du', 'del')
        for fr, ca in fr_to_ca_days.items(): translated = translated.replace(fr, ca)
        for fr, ca in fr_to_ca_months.items(): translated = re.sub(fr, ca, translated, flags=re.IGNORECASE)
        translated_lines.append(translated)
    
    sorted_indices = np.argsort(p_list)[::-1]
    return {
        'p_levels': np.array(p_list)[sorted_indices] * units.hPa,
        't_initial': np.array(t_list)[sorted_indices] * units.degC,
        'td_initial': np.array(td_list)[sorted_indices] * units.degC,
        'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph,
        'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees,
        'observation_time': "\n".join(translated_lines) or "Hora no disponible"
    }
# =========================================================================
# === 2. CÀLCULS ==========================================================
# =========================================================================
def calculate_all_parameters(p, t, td, ws, wd):
    params = {}
    try:
        valid = ~np.isnan(p.m) & ~np.isnan(t.m) & ~np.isnan(td.m)
        p, t, td = p[valid], t[valid], td[valid]
        p_sfc, t_sfc, td_sfc = p[0], t[0], td[0]
        
        parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
        params['cape'], params['cin'] = mpcalc.cape_cin(p, t, td, parcel_prof)
        params['lcl_p'], _ = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
        params['lfc_p'], _ = mpcalc.lfc(p, t, td, parcel_prof)
        params['el_p'], _ = mpcalc.el(p, t, td, parcel_prof)
        params['pwat_total'] = mpcalc.precipitable_water(p, td).to('mm')
        
        t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value="extrapolate")
        p_range = np.arange(p.m.min(), p.m.max())
        fz_idx = np.where(t_interp(p_range) < 0)[0]
        fz_lvl = p_range[fz_idx[0]] * units.hPa if fz_idx.size > 0 else np.nan * units.hPa
        
        params['lcl_h'] = mpcalc.pressure_to_height_std(params['lcl_p']).to('m').m if params['lcl_p'] else 0
        params['lfc_h'] = mpcalc.pressure_to_height_std(params['lfc_p']).to('m').m if params['lfc_p'] else np.inf
        params['el_h'] = mpcalc.pressure_to_height_std(params['el_p']).to('m').m if params['el_p'] else params['lfc_h']
        params['fz_h'] = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
        
        heights_agl = (mpcalc.pressure_to_height_std(p) - mpcalc.pressure_to_height_std(p_sfc)).to('km')
        mask_0_4 = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(mask_0_4) > 2:
            params['rh_0_4'] = np.mean(mpcalc.relative_humidity_from_dewpoint(t[mask_0_4], td[mask_0_4]))
            params['pwat_0_4'] = mpcalc.precipitable_water(p[mask_0_4], td[mask_0_4]).to('mm')
        else: params['rh_0_4'], params['pwat_0_4'] = 0.0, 0 * units.mm
            
        u, v = mpcalc.wind_components(ws, wd)
        h_raw = mpcalc.pressure_to_height_std(p)
        valid_wind = ~np.isnan(h_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_wind) > 2:
            h_clean, u_clean, v_clean = h_raw[valid_wind], u[valid_wind], v[valid_wind]
            _, unique_idx = np.unique(h_clean.m, return_index=True)
            if len(unique_idx) > 2:
                h_u, u_u, v_u = h_clean[unique_idx], u_clean[unique_idx], v_clean[unique_idx]
                h_interp = np.arange(h_u.m.min(), min(h_u.m.max(), 16000), 50) * units.meter
                u_i, v_i = np.interp(h_interp.m, h_u.m, u_u.m)*units('m/s'), np.interp(h_interp.m, h_u.m, v_u.m)*units('m/s')
                u6, v6 = mpcalc.bulk_shear(p, u_i, v_i, height=h_interp, depth=6000*units.meter)
                params['shear_0_6'] = mpcalc.wind_speed(u6, v6).m
                params['srh_0_1'] = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000*units.meter)[0].m
                params['srh_0_3'] = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000*units.meter)[0].m
    except Exception: pass
    
    # Valors per defecte si el càlcul falla
    default_params = {'cape': 0*units('J/kg'), 'cin': 0*units('J/kg'), 'lcl_p': None, 'lfc_p': None, 'el_p': None, 'pwat_total': 0*units.mm, 'lcl_h': 0, 'lfc_h': np.inf, 'el_h': 0, 'fz_h': 0, 'rh_0_4': 0.0, 'pwat_0_4': 0*units.mm, 'shear_0_6': 0.0, 'srh_0_1': 0.0, 'srh_0_3': 0.0}
    for key, val in default_params.items():
        if key not in params: params[key] = val
    return params

# =========================================================================
# === 3. ANÀLISI I GENERACIÓ DE TEXT ======================================
# =========================================================================
def generate_detailed_analysis(params, cloud_type):
    initial_summary = f"Hola! Mira el que veig: una situació compatible amb la formació de núvols de tipus **{cloud_type}**."
    conversation = [("Tempestes.cat", initial_summary)]
    
    if cloud_type == "Hivernal":
        conversation.extend([
            ("Yo", f"La isoterma 0°C és a {params['fz_h']:.0f}m. Això és molt baix, no?"),
            ("Tempestes.cat", "Exacte. És el factor clau per a la neu."),
            ("Yo", f"I amb una Tª en superfície de {params['t_sfc']:.1f}°C, què podem esperar?"),
            ("Tempestes.cat", "Amb Tª sota zero, neu segura. Si és lleugerament positiva, compte amb la pluja gelant!")
        ])
    # ... (Altres condicions de conversa es poden afegir aquí de manera similar) ...
    else:
        conversation.extend([
            ("Yo", "Sembla un dia tranquil, oi?"),
            ("Tempestes.cat", f"Sí. Amb un CAPE de només {params['cape'].m:.0f} J/kg, l'atmosfera és molt estable."),
        ])
    return conversation

def generate_public_warning(params):
    if params['fz_h'] < 1500 or params['t_sfc'] < 5:
        if params['t_sfc'] <= 0.5: return "AVÍS PER NEU", "Es preveu nevada a cotes baixes.", "navy"
        return "AVÍS PER PLUJA GEBRADORA", "Risc de pluja gelant o glaçades.", "dodgerblue"
    if params['rh_0_4'] > 0.85 and params['cape'].m < 350:
        if params['pwat_0_4'].m > 25: return "AVÍS PER PLUGES INTENSES", "Risc de pluges fortes i persistents.", "darkblue"
        elif params['pwat_0_4'].m > 15: return "AVÍS PER PLUJA MODERADA", "Pluja contínua i moderada.", "steelblue"
        return "PREVISIÓ DE PLUJA FEBLE", "S'esperen plugims o ruixats febles.", "cadetblue"
    if params['cape'].m >= 1000:
        if params['srh_0_1'] > 150 and params['shear_0_6'] > 15: return "AVÍS PER TORNADO", "Condicions favorables per a tornados.", "darkred"
        if params['lfc_h'] > 3000: return "AVÍS PER TEMPESTES DE BASE ALTA", "Risc de ratxes de vent fortes (downbursts).", "darkorange"
        if params['cape'].m > 2000: return "AVÍS PER PEDRA", "Tempestes violentes amb risc de pedra grossa.", "purple"
        return "AVÍS PER TEMPESTES", "Tempestes fortes amb pluja intensa i possible calamarsa.", "darkorange"
    return "SENSE AVISOS", "Condicions meteorològiques sense riscos significatius.", "green"

# =========================================================================
# === 4. FUNCIONS DE DIBUIX ===============================================
# =========================================================================
def create_logo_figure():
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')
    ax.add_patch(Circle((5, 5), 5, facecolor='#F5F1E9'))
    ax.add_patch(Polygon([(2, 5.8), (1.5, 6.8), (2.5, 7.8), (4, 8.3), (6, 8.3), (7.5, 7.8), (8.5, 6.8), (8, 5.8), (7, 5.3), (3, 5.3)], facecolor='#4B2A4B', zorder=10))
    ax.text(5, 6.6, 'tempestes.cat', ha='center', va='center', fontsize=3.3, color='white', weight='bold', fontfamily='sans-serif', zorder=20)
    bar_data = {'heights': [0.8, 1.0, 0.9, 0.7, 0.95, 0.85, 0.6, 0.75, 0.5], 'start_x': 3.0, 'width': 0.4, 'start_y': 5.3}
    for i, h in enumerate(bar_data['heights']):
        x, color = bar_data['start_x'] + i * bar_data['width'], '#DA121A' if i % 2 == 0 else '#FCDD09'
        height = h * 4.0
        ax.add_patch(Rectangle((x + 0.05, bar_data['start_y'] - height - 0.05), bar_data['width'], height, facecolor='black', alpha=0.3, lw=0, zorder=4))
        ax.add_patch(Rectangle((x, bar_data['start_y'] - height), bar_data['width'], height, facecolor=color, lw=0, zorder=5))
    return fig

def _draw_cumulus_mediocris(ax, base_km, top_km):
    center_x, cloud_height = 0, top_km - base_km
    altitudes = np.linspace(base_km, top_km, 20)
    widths = 0.4 * (1 + 0.8 * np.sin(np.pi * (altitudes - base_km) / (cloud_height + 0.01))) + np.random.uniform(-0.1, 0.1, 20)
    widths[0] = max(widths[0], 0.3)
    r_pts, l_pts = [(center_x + w, alt) for w, alt in zip(widths, altitudes)], [(center_x - w, alt) for w, alt in zip(widths, altitudes)]
    ax.add_patch(Polygon([l_pts[0]] + r_pts + l_pts[::-1], facecolor='#d0d0d0', lw=0, zorder=10))
    patches = []
    for _ in range(250):
        y_progress = random.betavariate(2, 2)
        y = base_km + y_progress * cloud_height
        x = center_x + random.uniform(-1, 1) * np.interp(y, altitudes, widths) * 0.95
        size = random.uniform(0.15, 0.5) * (1 + y_progress * 0.5)
        brightness = 0.8 + 0.2 * (y_progress ** 0.7)
        patches.append(Circle((x, y), size, facecolor=(brightness,)*3, alpha=random.uniform(0.15, 0.45), lw=0))
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=11))

# ... Altres funcions de dibuix (_draw_cumulonimbus, etc.) es poden simplificar de manera similar ...

def create_skewt_figure(p, t, td, params):
    fig, skew = plt.figure(figsize=(10, 10)), SkewT(plt.figure(figsize=(10, 10)), rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100); ax.set_xlim(-50, 45)
    with integrator_lock: skew.plot_dry_adiabats(alpha=0.3, color='orange'); skew.plot_moist_adiabats(alpha=0.3, color='green')
    skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
    skew.plot(p, t, 'r', linewidth=2, label='Temperatura (T)')
    skew.plot(p, np.minimum(t, td), 'b', linewidth=2, label='Punt de Rosada (Td)')
    parcel_prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')
    skew.plot(p, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
    skew.shade_cape(p, t, parcel_prof, facecolor='yellow', alpha=0.3); skew.shade_cin(p, t, parcel_prof, facecolor='black', alpha=0.3)
    xlims = ax.get_xlim()
    if params['lcl_p']: ax.plot(xlims, [params['lcl_p'].m]*2, 'gray', ls='--', label='LCL')
    if params['lfc_p']: ax.plot(xlims, [params['lfc_p'].m]*2, 'purple', ls='--', label='LFC')
    if params['el_p']: ax.plot(xlims, [params['el_p'].m]*2, 'red', ls='--', label='EL')
    ax.legend(); plt.tight_layout()
    return skew.fig

# ... La resta de funcions de dibuix es mantindrien aquí ...

# =========================================================================
# === 5. APP STREAMLIT ====================================================
# =========================================================================
def main():
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")

    if 'initialized' not in st.session_state:
        st.session_state.existing_files = [f for f in [f"{h}{p}.txt" for h in range(1, 13) for p in ['am', 'pm']] if os.path.exists(f)]
        if not st.session_state.existing_files: st.stop()
        st.session_state.sounding_index, st.session_state.loaded_sounding_index = 0, -1
        st.session_state.convergence_active, st.session_state.initialized = True, True
        st.session_state.chat_open, st.session_state.chat_progress = False, 0

    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        st.session_state.selected_file = st.session_state.existing_files[st.session_state.sounding_index]
        soundings = parse_all_soundings(st.session_state.selected_file)
        if soundings:
            st.session_state.original_data = soundings[0]
            reset_working_profiles()
            st.session_state.loaded_sounding_index = st.session_state.sounding_index
        else:
            st.error(f"Error carregant {st.session_state.selected_file}")
            st.session_state.sounding_index = st.session_state.loaded_sounding_index

    logo_fig = create_logo_figure()
    logo_buffer = io.BytesIO(); logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    logo_b64 = base64.b64encode(logo_buffer.getvalue()).decode()
    
    with st.sidebar:
        st.image(logo_buffer)
        st.title("Controls")
        st.selectbox("Selecciona una hora:", st.session_state.existing_files, index=st.session_state.sounding_index, key='selectbox_widget', on_change=sync_index_from_selectbox)
        st.toggle("Activar convergència", st.session_state.convergence_active, key='convergence_active')
        if st.button("🔄 Reiniciar Perfils"): reset_working_profiles(); st.success("Perfils reiniciats.")
        with st.expander("🔬 Modificació Avançada"):
            sfc_temp_val = st.session_state.t_profile[0].m
            new_sfc_temp = st.slider("Tª Superfície (°C)", sfc_temp_val-20, sfc_temp_val+20, sfc_temp_val, 0.5)
            if new_sfc_temp != sfc_temp_val: st.session_state.t_profile[0] = new_sfc_temp * units.degC

    st.title("Visor de Sondejos Atmosfèrics")
    time_parts = st.session_state.observation_time.split('\n')
    cleaned_time_str = next((p.strip() for p in time_parts if 'local' in p.lower()), time_parts[0].strip() if time_parts else "")
    st.markdown(f"#### {cleaned_time_str}")

    p, t, td, ws, wd = st.session_state.p_levels, st.session_state.t_profile, st.session_state.td_profile, st.session_state.wind_speed, st.session_state.wind_dir
    
    params = calculate_all_parameters(p, t, td, ws, wd)
    params['t_sfc'] = t[0].m # Afegir per a anàlisi
    
    title, message, color = generate_public_warning(params)
    st.markdown(f'<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>', unsafe_allow_html=True)

    sub_cols = st.columns([2, 8, 2])
    with sub_cols[0]: st.button('← Anterior', on_click=decrement_index, disabled=(st.session_state.sounding_index==0), use_container_width=True)
    with sub_cols[1]: st.subheader("Diagrama Skew-T", anchor=False)
    with sub_cols[2]: st.button('Següent →', on_click=increment_index, disabled=(st.session_state.sounding_index>=len(st.session_state.existing_files)-1), use_container_width=True)
    
    st.pyplot(create_skewt_figure(p, t, td, params), use_container_width=True)
    st.divider()

    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, st.session_state.convergence_active)
    
    cloud_type = "Cel Serè"
    if params['t_sfc'] < 5 or params['fz_h'] < 1500: cloud_type = "Hivernal"
    elif params['rh_0_4'] > 0.85 and params['cape'].m < 350:
        if params['pwat_0_4'].m > 25: cloud_type = "Nimbostratus (Intens)"
        elif params['pwat_0_4'].m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif params['cape'].m > 2000 and params['shear_0_6'] > 18 and params['srh_0_3'] > 150: cloud_type = "Supercèl·lula"
    elif params['cape'].m > 500: cloud_type = "Cumulonimbus (Multicèl·lula)" if params['lfc_h'] < 3000 else "Castellanus"
    elif base_km and top_km:
        if (top_km - base_km) > 2.0 and params['lfc_h'] < 3000: cloud_type = "Cumulus Mediocris"
        elif (top_km - base_km) > 0: cloud_type = "Cumulus Fractus"

    conversation, precip_type = generate_detailed_analysis(params, cloud_type)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi", "📊 Paràmetres", "☁️ Visualització", "📡 Radar"])

    with tab1:
        if 'chat_open' not in st.session_state: st.session_state.chat_open = False
        if not st.session_state.chat_open:
            if st.button("Obrir Anàlisi de Tempestes.cat", key="open_chat_btn", use_container_width=True):
                st.session_state.chat_open = True; st.rerun()
        else: display_whatsapp_chat(conversation, logo_b64)

    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        cols = st.columns(4)
        cols[0].metric("CAPE", f"{params['cape'].m:.0f} J/kg"); cols[1].metric("CIN", f"{params['cin'].m:.0f} J/kg")
        cols[2].metric("PWAT Total", f"{params['pwat_total'].m:.1f} mm"); cols[3].metric("0°C", f"{params['fz_h']/1000:.2f} km")
        cols[0].metric("LCL", f"{params['lcl_p'].m:.0f} hPa" if params['lcl_p'] else "N/A"); cols[1].metric("LFC", f"{params['lfc_p'].m:.0f} hPa" if params['lfc_p'] else "N/A")
        # CORRECCIÓ: Accedir al valor .m de el_p només si existeix
        cols[2].metric("EL", f"{params['el_p'].m:.0f} hPa" if params['el_p'] else "N/A"); cols[3].metric("Shear 0-6", f"{params['shear_0_6']:.1f} m/s")
        cols[0].metric("SRH 0-1", f"{params['srh_0_1']:.1f} m²/s²"); cols[1].metric("SRH 0-3", f"{params['srh_0_3']:.1f} m²/s²")
        cols[2].metric("PWAT 0-4km", f"{params['pwat_0_4'].m:.1f} mm"); cols[3].metric("RH Mitja 0-4km", f"{params['rh_0_4']*100:.0f}%")
        
    with tab3: # Les funcions de dibuix s'haurien de refactoritzar per acceptar `params`
        pass # Ocult per brevetat, però aquí anirien les crides a create_cloud_drawing_figure, etc.

    with tab4: # Igual que a la pestanya 3
        pass

# Funcions auxiliars per a l'app
def reset_working_profiles(): st.session_state.t_profile, st.session_state.td_profile = st.session_state.original_data['t_initial'].copy(), st.session_state.original_data['td_initial'].copy()
def increment_index(): st.session_state.sounding_index = min(st.session_state.sounding_index + 1, len(st.session_state.existing_files) - 1); st.session_state.chat_open = False
def decrement_index(): st.session_state.sounding_index = max(st.session_state.sounding_index - 1, 0); st.session_state.chat_open = False
def sync_index_from_selectbox(): st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget); st.session_state.chat_open = False

if __name__ == '__main__':
    main()
