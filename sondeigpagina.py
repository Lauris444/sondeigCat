# =========================================================================
# === ANALITZADOR DE SONDEJOS ATMOSFÈRICS v2.4 (FIX DEFINITIU) =============
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
    # ... (Codi complet de la funció)
    pass

def set_main_background():
    # ... (Codi complet de la funció)
    pass

# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

def fetch_sounding_from_url(station_code, forecast_hour):
    """
    VERSIÓ ROBUSTA: Obté i processa un sondeig del model GFS des de la
    Universitat de Wyoming, amb una lògica de temps corregida.
    """
    now_utc = datetime.utcnow()
    
    # LÒGICA CORREGIDA: Determinar la passada del model més recent que JA estigui disponible
    # S'aplica un marge de seguretat de 5 hores.
    adjusted_time = now_utc - timedelta(hours=5)
    
    if adjusted_time.hour >= 18: model_run_hour = 18
    elif adjusted_time.hour >= 12: model_run_hour = 12
    elif adjusted_time.hour >= 6: model_run_hour = 6
    else: model_run_hour = 0
    
    model_run_date = adjusted_time.date()

    base_url = "http://weather.uwyo.edu/cgi-bin/sounding"
    params = {
        'region': 'europe', 'TYPE': 'TEXT:LIST', 'YEAR': model_run_date.strftime('%Y'),
        'MONTH': model_run_date.strftime('%m'), 'DAY': model_run_date.strftime('%d'),
        'TIME': f'{model_run_hour:02d}', 'STNM': station_code, 'MODEL': 'gfs', 'FHOUR': forecast_hour
    }
    
    try:
        response = requests.get(base_url, params=params, headers={'User-Agent': 'MyCoolWeatherApp/1.0'})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        pre_tag = soup.find('pre')

        if not pre_tag:
            st.warning(f"No s'ha pogut trobar el bloc de dades per a l'estació {station_code} a +{forecast_hour}h.")
            return None

        sounding_text = pre_tag.get_text()
        sounding_lines = sounding_text.splitlines()
        processed_data = process_wyoming_sounding_block(sounding_lines)
        
        if processed_data:
            processed_data['observation_time'] = f"Model GFS | Passada: {model_run_date.strftime('%d/%m/%Y')} {model_run_hour:02d}Z\nPronòstic: +{forecast_hour}h"
            
        return processed_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error de xarxa en contactar amb la Universitat de Wyoming: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperat processant el sondeig de Wyoming: {e}")
        return None

def process_wyoming_sounding_block(block_lines):
    p_list, t_list, td_list, wdir_list, wspd_list = [], [], [], [], []
    data_started = False
    for line in block_lines:
        line_strip = line.strip()
        if "-----------------" in line_strip:
            data_started = True
            continue
        if not data_started or not line_strip: continue
        try:
            parts = line_strip.split()
            if len(parts) < 11: continue
            
            p, t, td, wdir = clean_and_convert(parts[0]), clean_and_convert(parts[2]), clean_and_convert(parts[3]), clean_and_convert(parts[6])
            wspd_knots = clean_and_convert(parts[7])
            
            if any(v is None for v in [p, t, td, wdir, wspd_knots]): continue

            p_list.append(p); t_list.append(t); td_list.append(td); wdir_list.append(wdir)
            wspd_list.append(wspd_knots * 1.852)

        except (ValueError, IndexError): continue
            
    if not p_list: return None
    
    sorted_indices = np.argsort(p_list)[::-1]
    return {
        'p_levels': np.array(p_list)[sorted_indices] * units.hPa, 't_initial': np.array(t_list)[sorted_indices] * units.degC,
        'td_initial': np.array(td_list)[sorted_indices] * units.degC, 'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph,
        'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees
    }

# ... (Aquí anirien les altres funcions auxiliars com get_image_as_base64, parse_all_soundings, etc.)

# =========================================================================
# === 5. MODES DE L'APLICACIÓ COMPLETES ===================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    st.markdown('<p class="welcome-title">TEMPESTES.CAT PRESENTA:</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Eina de visualització i experimentació amb perfils atmosfèrics.</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="mode-card"><h3>🛰️ Temps real</h3><p>Visualitza sondejos del model GFS per a diferents localitzacions de Catalunya.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Mode temps real", use_container_width=True):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown("""<div class="mode-card"><h3>🧪 Laboratori</h3><p>Aprèn de forma interactiva modificant un sondeig o experimenta lliurement.</p></div>""", unsafe_allow_html=True)
        if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
            st.session_state.app_mode = 'sandbox'
            st.rerun()

def show_province_selection_screen():
    # ... (Codi de l'animació complet aquí)
    pass

def run_live_mode():
    LOCATIONS = {
        'barcelona': {'name': 'Barcelona', 'station': '08181'},
        'girona': {'name': 'Girona', 'station': '08181'},
        'lleida': {'name': 'Lleida', 'station': '08170'},
        'tarragona': {'name': 'Tarragona (Reus)', 'station': '08175'}
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
        st.info("Dades del model GFS obtingudes de la Universitat de Wyoming.")
        forecast_hours = list(range(0, 181, 3)) 
        selected_hour = st.selectbox(
            "Hora del pronòstic:",
            options=forecast_hours,
            format_func=lambda h: f"+ {h} hores",
            key=f"hour_selector_{selected_province_key}"
        )

    @st.cache_data(ttl=1800)
    def get_data_for_hour(station_code, hour):
        return fetch_sounding_from_url(station_code, hour)

    with st.spinner(f"Obtenint dades del GFS per a {location_data['name']} a +{selected_hour}h..."):
        sounding_data = get_data_for_hour(location_data['station'], selected_hour)
    
    if sounding_data:
        # show_full_analysis_view(...) # Aquesta funció ha d'estar definida al teu codi complet
        pass
    else:
        st.error(f"No s'han pogut obtenir dades per a {location_data['name']} a +{selected_hour}h.")

def run_sandbox_mode():
    # ... (Codi complet del mode laboratori)
    pass

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
