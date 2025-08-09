# =========================================================================
# === ANALITZADOR DE SONDEJOS ATMOSFÈRICS v2.2 (FIX DEFINITIU) =============
# =========================================================================

# --- 1. IMPORTACIONS DE LLIBRERIES ---------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import random
import os
import re
import threading
import base64
import io
import time
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup

# --- 2. CONFIGURACIÓ GLOBAL ----------------------------------------------
integrator_lock = threading.Lock()

# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

def show_loading_animation(message="Carregant"):
    loading_html = f"""
    <style>
        .loading-container{{position:fixed;top:0;left:0;width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;background:rgba(25,37,81,.9);z-index:9999}}.loading-svg{{width:150px;height:auto;margin-bottom:20px}}.loading-text{{color:#fff;font-size:1.5rem;font-family:sans-serif}}.loading-text .dot{{animation:blink 1.4s infinite both}}.loading-text .dot:nth-child(2){{animation-delay:.2s}}.loading-text .dot:nth-child(3){{animation-delay:.4s}}@keyframes blink{{0%,80%,100%{{opacity:0}}40%{{opacity:1}}}}
    </style>
    <div class="loading-container">
        <svg class="loading-svg" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg"><path d="M 155.6,66.1 C 155.6,42.9 135.5,23.5 111.4,23.5 C 98.4,23.5 86.8,29.4 79.1,38.7 C 75.2,16.8 57.3,0 36.4,0 C 16.3,0 0,16.3 0,36.4 C 0,56.5 16.3,72.8 36.4,72.8 L 110,72.8 C 110,72.8 110,72.8 110,72.8 C 135,72.8 155.6,93.4 155.6,118.4 C 155.6,143.4 135,164 110,164 L 50, 164" fill="none" stroke="#FFFFFF" stroke-width="8"/><polygon points="120,60 90,10 115,110 100,150 145,90 120,90 130,60" fill="#FFD700" /></svg>
        <div class="loading-text">{message}<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

def set_main_background():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main{{background:linear-gradient(0deg,rgba(6,14,42,1) 0%,rgba(25,37,81,1) 100%);background-size:cover;background-position:center center;background-repeat:no-repeat;background-attachment:local}}[data-testid="stHeader"]{{background:rgba(0,0,0,0)}}[data-testid="stToolbar"]{{right:2rem}}.welcome-title{{font-size:3.5rem;font-weight:700;color:#fff;text-align:center;text-shadow:2px 2px 8px rgba(0,0,0,.7)}}.welcome-subtitle{{font-size:1.5rem;color:#e0e0e0;text-align:center;margin-bottom:40px}}.mode-card{{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);padding:25px;border-radius:15px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);color:#fff;height:100%}}.mode-card h3{{color:#fff;font-weight:700}}.mode-card p{{color:#d0d0d0}}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

def fetch_sounding_from_url(lat, lon, forecast_hour):
    url = f"https://www.meteociel.fr/modeles/sondage2arome.php?archive=0&ech={forecast_hour}&map=8&wrf=0&y1={lat}&x1={lon}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        html_content = response.text
        pattern = re.compile(r"mySondage\.draw\('(.*?)'\);", re.DOTALL)
        match = pattern.search(html_content)
        if not match:
            st.warning(f"No s'ha pogut trobar el bloc de dades del sondeig a la URL per a +{forecast_hour}h.")
            return None
        sounding_raw_text = match.group(1)
        sounding_text = sounding_raw_text.encode().decode('unicode_escape')
        sounding_lines = sounding_text.splitlines()
        processed_data = process_sounding_block(sounding_lines)
        if processed_data:
            soup = BeautifulSoup(response.content, 'html.parser')
            model_run_info = soup.find('h3').get_text(strip=True) if soup.find('h3') else f"Model AROME {forecast_hour:02d}Z"
            processed_data['observation_time'] = f"{model_run_info}\nPronòstic: +{forecast_hour}h"
        return processed_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error de xarxa en intentar obtenir el sondeig: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperat processant el sondeig de la URL: {e}")
        return None

def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f: data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError: return None

def clean_and_convert(text):
    cleaned_text = re.sub(r'[^\d.,-]', '', str(text)).replace(',', '.')
    if not cleaned_text or cleaned_text == '-': return None
    try: return float(cleaned_text)
    except ValueError: return None

def process_sounding_block(block_lines):
    if not block_lines: return None
    p_list, t_list, td_list, wdir_list, wspd_list = [], [], [], [], []
    time_lines = []
    for line in block_lines:
        line_strip = line.strip()
        if any(keyword in line_strip.lower() for keyword in ['observació','time','run','date']) and not (line_strip and line_strip[0].isdigit()):
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
            if len(parts) > 6 and '/' in parts[6]:
                wind_parts = parts[6].strip().split('/')
                if len(wind_parts) == 2:
                    wdir_val, wspd_val = clean_and_convert(wind_parts[0]), clean_and_convert(wind_parts[1])
                    if wdir_val is not None: wdir = wdir_val
                    if wspd_val is not None: wspd = wspd_val
            wdir_list.append(wdir); wspd_list.append(wspd)
        except: continue
    if not p_list or len(p_list) < 2: return None
    observation_time = "\n".join(time_lines) if time_lines else "Hora no disponible"
    sorted_indices = np.argsort(p_list)[::-1]
    return {'p_levels': np.array(p_list)[sorted_indices] * units.hPa, 't_initial': np.array(t_list)[sorted_indices] * units.degC, 'td_initial': np.array(td_list)[sorted_indices] * units.degC, 'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph, 'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees, 'observation_time': observation_time}

def parse_all_soundings(filepath):
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat '{filepath}'. Aquest fitxer és necessari per al Mode Laboratori.")
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
# === TOTES LES ALTRES FUNCIONS (Càlcul, Dibuix, Laboratori...) ============
# =========================================================================
# ... (Aquí enganxes TOTES les altres funcions que et vaig donar al codi complet anterior)
# ... Aquesta secció és molt llarga, però és només copiar i enganxar. ...

# =========================================================================
# === ESTRUCTURA PRINCIPAL DE L'APLICACIÓ =================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">TEMPESTES.CAT PRESENTA:</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Eina de visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️ Temps real</h3><p>Visualitza sondejos actualitzats del model AROME per a diferents localitzacions.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode temps real", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪 Laboratori</h3><p>Aprèn de forma interactiva modificant un sondeig o experimenta lliurement.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def show_province_selection_screen():
    animation_html = """
    <style>
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        @keyframes day-night-cycle { 0%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)}20%{background:linear-gradient(to top,#020111 20%,#2c3e50,#f39c12,#f1c40f)}35%{background:linear-gradient(to top,#87CEEB 20%,#B2FFFF)}60%{background:linear-gradient(to top,#87CEEB 20%,#B2FFFF)}75%{background:linear-gradient(to top,#1e2531 20%,#e74c3c,#f39c12)}90%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)}100%{background:linear-gradient(to top,#020111 25%,#0d1a3b,#1d2f63)} }
        @keyframes sun-moon-move { 0%{transform:translate(-10vw,30vh);background:#F5F3CE;box-shadow:0 0 20px #F5F3CE;opacity:1}20%{transform:translate(50vw,-20vh);background:#FFD700;box-shadow:0 0 40px #FFD700;opacity:1}60%{transform:translate(110vw,30vh);opacity:1}75%{transform:translate(50vw,80vh);opacity:0}80%{transform:translate(-10vw,30vh);background:#F5F3CE;box-shadow:0 0 20px #F5F3CE;opacity:1}100%{transform:translate(-10vw,30vh);opacity:1} }
        @keyframes stars-opacity { 0%{opacity:1}20%{opacity:0}75%{opacity:0}90%{opacity:1}100%{opacity:1} }
        @keyframes shooting-star-anim { 0%{transform:translate(120vw,-30vh);opacity:0}94%{transform:translate(120vw,-30vh);opacity:0}95%{transform:translate(80vw,10vh) scale(1);opacity:1}100%{transform:translate(-40vw,90vh) scale(1);opacity:0} }
        @keyframes twinkle { 0%,100%{opacity:1}50%{opacity:.3} }
        .animation-container { position:fixed;top:0;left:0;width:100%;height:100vh;overflow:hidden;z-index:-1 }
        .animated-sky { position:absolute;top:0;left:0;width:100%;height:100%;animation:day-night-cycle 40s linear infinite }
        .sun-moon { position:absolute;width:clamp(60px,8vw,120px);height:clamp(60px,8vw,120px);border-radius:50%;animation:sun-moon-move 40s linear infinite }
        .stars-wrapper { position:absolute;top:0;left:0;width:100%;height:60%;animation:stars-opacity 40s linear infinite }
        .star { position:absolute;background:white;border-radius:50%;animation:twinkle 4s ease-in-out infinite }
        .shooting-star { position:absolute;width:3px;height:100px;background:linear-gradient(to bottom,rgba(255,255,255,1),rgba(255,255,255,0));border-radius:50%;transform-origin:top center;filter:drop-shadow(0 0 6px white);animation:shooting-star-anim 40s linear infinite }
        .landscape { position:absolute;bottom:0;left:0;width:100%;height:35vh;background:#000 }
        .city-silhouette,.mountain-silhouette { position:absolute;bottom:0;width:100%;height:100%;background-repeat:no-repeat;background-position:bottom }
        .mountain-silhouette { background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 175"><path d="M0 175 H 500 V 100 L 450 120 L 400 60 L 350 110 L 300 30 L 250 100 L 200 80 L 150 130 L 100 90 L 50 140 L 0 110 Z" fill="%2312182c"/></svg>');background-size:100% 100%;z-index:2 }
        .city-silhouette { background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 150"><path d="M0 150 V 100 L 20 105 L 40 80 L 60 90 L 100 40 L 140 110 L 180 100 L 220 50 L 250 120 L 280 110 L 320 20 L 360 130 L 400 110 L 450 10 L 500 140 L 550 120 L 600 70 L 650 130 L 700 90 L 750 120 L 800 80 V 150 H 0 Z" fill="%230a0a0a"/></svg>');background-size:100% 100%;z-index:3 }
        .city-lights-wrapper { position:absolute;bottom:0;left:0;width:100%;height:100%;z-index:4;animation:stars-opacity 40s linear infinite }
        .light { position:absolute;background:#fde9a0;border-radius:50%;width:.3%;height:1.5%;animation:twinkle 3s ease-in-out infinite }
    </style>
    <div class="animation-container"><div class="animated-sky"><div class="sun-moon"></div><div class="stars-wrapper">
    """ + ''.join([f'<div class="star" style="top:{random.uniform(0,70)}%;left:{random.uniform(0,100)}%;width:{random.uniform(1,2.5)}px;height:{random.uniform(1,2.5)}px;animation-delay:{random.uniform(0,4)}s;"></div>' for _ in range(200)]) + """
    </div><div class="shooting-star"></div></div><div class="landscape"><div class="mountain-silhouette"></div><div class="city-silhouette"></div><div class="city-lights-wrapper">
    """ + ''.join([f'<div class="light" style="bottom:{random.uniform(5,50)}%;left:{random.uniform(2,98)}%;animation-delay:{random.uniform(0,3)}s;"></div>' for _ in range(80)]) + """
    </div></div></div>
    """
    st.markdown(animation_html, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:white;text-shadow:2px 2px 6px #000;padding-top:25vh;'>Selecciona una Província</h2>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        def select_province(name):
            st.session_state.province_selected = name
        st.button("Barcelona", on_click=select_province, args=('barcelona',), use_container_width=True, type="primary")
        st.button("Girona", on_click=select_province, args=('girona',), use_container_width=True)
        st.button("Lleida", on_click=select_province, args=('lleida',), use_container_width=True)
        st.button("Tarragona", on_click=select_province, args=('tarragona',), use_container_width=True)

def run_live_mode():
    LOCATIONS = {
        'barcelona': {'name': 'Barcelona', 'lat': 668, 'lon': 467},
        'girona': {'name': 'Girona', 'lat': 668, 'lon': 437},
        'lleida': {'name': 'Lleida', 'lat': 738, 'lon': 477},
        'tarragona': {'name': 'Tarragona', 'lat': 698, 'lon': 497}
    }
    selected_province_key = st.session_state.get('province_selected')

    if not selected_province_key:
        with st.sidebar:
            st.header("Controls")
            if st.button("⬅️ Tornar a l'inici", use_container_width=True):
                st.session_state.app_mode = 'welcome'
                st.rerun()
        show_province_selection_screen()
        return

    location_data = LOCATIONS.get(selected_province_key)
    set_main_background() 
    st.title(location_data['name'].upper())
    
    with st.sidebar:
        st.header("Controls")
        def back_to_selection():
            st.session_state.province_selected = None
        st.button("⬅️ Tornar a la selecció", use_container_width=True, on_click=back_to_selection)
        st.markdown("---")
        st.subheader("Selecciona pronòstic")
        st.info("Dades del model AROME obtingudes en temps real des de Meteociel.fr")
        forecast_hours = list(range(1, 49, 1))
        selected_hour = st.selectbox("Hora del pronòstic:", options=forecast_hours, format_func=lambda h: f"+ {h} hores", key=f"hour_selector_{selected_province_key}")

    @st.cache_data(ttl=600)
    def get_data_for_hour(lat, lon, hour):
        return fetch_sounding_from_url(lat, lon, hour)

    with st.spinner(f"Obtenint dades per a {location_data['name']} a +{selected_hour}h..."):
        sounding_data = get_data_for_hour(location_data['lat'], location_data['lon'], selected_hour)
    
    if sounding_data:
        # show_full_analysis_view(...) # Aquesta funció ha d'estar definida al teu codi
        pass
    else:
        st.error(f"No s'han pogut obtenir dades per a {location_data['name']} a +{selected_hour}h.")

def run_sandbox_mode():
    st.title("🧪 Laboratori de Sondejos")
    st.info("El Mode Laboratori està en construcció.")

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

