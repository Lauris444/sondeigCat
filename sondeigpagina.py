# =========================================================================
# === 4. FUNCIONS PER A L'ESTRUCTURA DE L'APP ============================
# =========================================================================

def show_welcome_screen_animated():
    """
    Mostra una pantalla de benvinguda animada amb un fons negre i llamps.
    La pàgina es refresca cada segon per crear l'animació.
    """
    # CSS per a l'efecte de fons negre, llamps i centrat de contingut
    st.markdown("""
    <style>
        /* Forçar el fons negre a tota la pàgina */
        body {
            background-color: #000000 !important;
        }
        .main > div {
            background-color: #000000 !important;
        }

        /* Contenidor principal per centrar el contingut */
        .welcome-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 90vh;
            color: white;
            text-align: center;
            position: relative;
            z-index: 10;
        }

        /* Contenidor per a la capa de llamps */
        .lightning-layer {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; /* No interfereix amb els botons */
            z-index: 5;
            overflow: hidden;
        }

        /* L'element SVG del llamp */
        .lightning-svg {
            position: absolute;
            opacity: 0;
            animation: flash 0.6s ease-out;
        }
        
        /* Animació del flaix */
        @keyframes flash {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 0; }
        }
        
        /* Estil per als botons i text */
        .welcome-container h1, .welcome-container h3 {
            text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

    lightning_html = ""
    # Probabilitat d'un 35% que aparegui un llamp a cada refresc (cada segon)
    if random.random() < 0.35:
        # Generar propietats aleatòries per al llamp
        top = random.randint(0, 50)
        left = random.randint(5, 95)
        size = random.randint(150, 400)
        rotation = random.randint(-25, 25)
        # Paleta de colors per als llamps
        color = random.choice(['#FFFFFF', '#f0f8ff', '#f5f5dc']) # Blanc, AliceBlue, Beix
        
        # Codi SVG d'un llamp
        svg_path = "M13 1L1 14H9L7 23L23 10H15L13 1Z"
        style = (f"top: {top}%; left: {left}%; width: {size}px; height: {size}px; "
                 f"transform: rotate({rotation}deg) translateX(-50%);")

        lightning_html = f"""
        <div class="lightning-layer">
            <div class="lightning-svg" style="{style}">
                <svg viewBox="0 0 24 24" fill="{color}">
                    <path d="{svg_path}"></path>
                </svg>
            </div>
        </div>
        """
    
    # Injectar l'HTML del llamp (si s'ha generat)
    if lightning_html:
        st.markdown(lightning_html, unsafe_allow_html=True)
    
    # Contingut principal (títol i botons)
    with st.container():
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        
        st.title("Visor de Sondejos de Tempestes.cat")
        st.subheader("Tria un mode per començar")

        # Columnes per als botons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🛰️ Mode en Viu")
            st.info("Visualitza els sondejos atmosfèrics basats en dades reals i la teva hora local. Navega entre les diferents hores disponibles.")
            if st.button("Accedir al Mode en Viu", use_container_width=True):
                st.session_state.app_mode = 'live'
                st.rerun()
        with col2:
            st.markdown("### 🧪 Laboratori de Sondejos")
            st.info("Experimenta amb un sondeig de proves. Modifica paràmetres com la temperatura i la humitat o carrega escenaris predefinits per entendre com afecten el temps.")
            if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
                st.session_state.app_mode = 'sandbox'
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

    # Aquesta línia és la clau: fa que el navegador recarregui la pàgina cada segon.
    # Això executa el script de nou, permetent que la lògica del llamp s'executi repetidament.
    st.html('<meta http-equiv="refresh" content="1">')


# def show_welcome_screen(): # Aquesta és la funció original, la deixo comentada.
#     st.title("Benvingut al Visor de Sondejos de Tempestes.cat")
#     logo_fig = create_logo_figure()
#     st.pyplot(logo_fig)
#     st.subheader("Tria un mode per començar")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("### 🛰️ Mode en Viu")
#         st.info("Visualitza els sondejos atmosfèrics basats en dades reals i la teva hora local. Navega entre les diferents hores disponibles.")
#         if st.button("Accedir al Mode en Viu", use_container_width=True):
#             st.session_state.app_mode = 'live'
#             st.rerun()
#     with col2:
#         st.markdown("### 🧪 Laboratori de Sondejos")
#         st.info("Experimenta amb un sondeig de proves. Modifica paràmetres com la temperatura i la humitat o carrega escenaris predefinits per entendre com afecten el temps.")
#         if st.button("Accedir al Laboratori", use_container_width=True, type="primary"):
#             st.session_state.app_mode = 'sandbox'
#             st.rerun()

def apply_preset(preset_name):
    original_data = st.session_state.sandbox_original_data
    t_new = original_data['t_initial'].to('degC').magnitude.copy()
    td_new = original_data['td_initial'].to('degC').magnitude.copy()
    ws_new = original_data['wind_speed_kmh'].to('m/s').magnitude.copy()
    wd_new = original_data['wind_dir_deg'].magnitude.copy()
    if preset_name == 'neu':
        t_new -= 10
        td_new = t_new - np.random.uniform(0.5, 2, len(td_new))
    elif preset_name == 'calor':
        t_new += 15
        td_new = t_new - np.random.uniform(15, 25, len(td_new))
    elif preset_name == 'supercel':
        t_new[0] += 5
        td_new[0] = t_new[0] - 4
        inversion_mask = (st.session_state.sandbox_p_levels.magnitude > 800) & (st.session_state.sandbox_p_levels.magnitude < 900)
        t_new[inversion_mask] += 3
        ws_new += np.linspace(0, 30, len(ws_new))
        wd_new = (wd_new + np.linspace(0, 90, len(wd_new))) % 360
    elif preset_name == 'pluja':
        td_new = t_new - np.random.uniform(1, 3, len(td_new))
    td_new = np.minimum(t_new, td_new)
    st.session_state.sandbox_t_profile = t_new * units.degC
    st.session_state.sandbox_td_profile = td_new * units.degC
    st.session_state.sandbox_ws = ws_new * units('m/s')
    st.session_state.sandbox_wd = wd_new * units.degrees

def run_display_logic(p, t, td, ws, wd, obs_time):
    logo_fig = create_logo_figure()
    st.markdown(f"#### {obs_time}")
    convergence_active = st.session_state.get('convergence_active', True)
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h = calculate_thermo_parameters(p, t, td)
    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    base_km, top_km = _calculate_dynamic_cloud_heights(p, t, td, convergence_active)
    cloud_type = "Cel Serè"
    pwat_0_4, rh_0_4 = units.Quantity(0, 'mm'), 0.0
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_profile_layer = mpcalc.relative_humidity_from_dewpoint(t[layer_mask], td[layer_mask])
            rh_0_4 = np.mean(rh_profile_layer)
            pwat_0_4 = mpcalc.precipitable_water(p[layer_mask], td[layer_mask]).to('mm')
    except Exception: pass
    sfc_temp = t[0]
    if sfc_temp.m < 5 or fz_h < 1500: cloud_type = "Hivernal"
    elif rh_0_4 > 0.85 and cape.m < 350:
        if pwat_0_4.m > 25: cloud_type = "Nimbostratus (Intens)"
        elif pwat_0_4.m > 15: cloud_type = "Nimbostratus (Moderat)"
        else: cloud_type = "Nimbostratus (Fluix)"
    elif cape.m > 2000 and shear_0_6 > 18 and srh_0_3 > 150: cloud_type = "Supercèl·lula"
    elif cape.m > 500:
        cloud_type = "Cumulonimbus (Multicèl·lula)"
        if lfc_h >= 3000: cloud_type = "Castellanus"
    elif base_km and top_km:
        if (top_km - base_km) > 2.0 and lfc_h < 3000: cloud_type = "Cumulus Mediocris"
        elif (top_km - base_km) > 0: cloud_type = "Cumulus Fractus"
    title, message, color = generate_public_warning(p, t, td, ws, wd)
    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h3 style="color:white; text-align:center;">{title}</h3><p style="color:white; text-align:center; font-size:16px;">{message}</p></div>""", unsafe_allow_html=True)
    st.subheader("Diagrama Skew-T", anchor=False)
    fig_skewt = create_skewt_figure(p, t, td, ws, wd)
    st.pyplot(fig_skewt, use_container_width=True)
    st.divider()
    chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type, base_km, top_km, pwat_0_4)
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Anàlisi Detallada", "📊 Paràmetres Detallats", "☁️ Visualització de Núvols", "📡 Simulació Radar"])
    with tab1:
        st.subheader("Anàlisi conversacional")
        logo_buffer = io.BytesIO()
        logo_fig.savefig(logo_buffer, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        logo_base64 = base64.b64encode(logo_buffer.getvalue()).decode()
        css_styles = f"""<style>.chat-container {{ background-color: #f0f2f5; padding: 15px; border-radius: 10px; font-family: Arial, sans-serif; max-height: 450px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }}.message-row {{ display: flex; align-items: flex-end; gap: 10px; }}.message-row-right {{ justify-content: flex-end; }}.message {{ padding: 8px 14px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); position: relative; color: black; }}.yo {{ background-color: #0078D4; color: white; }}.tempestes-cat {{ background-color: #FFFFFF; border: 1px solid #e0e0e0; }}.sistema {{ background-color: #E1F2FB; align-self: center; text-align: center; font-style: italic; font-size: 0.9em; color: #555; width: auto; max-width: 90%; }}.message strong {{ display: block; margin-bottom: 3px; font-weight: bold; }}.yo strong {{color: #FFFFFF;}}.tempestes-cat strong {{ color: #075E54; }}.profile-pic {{ width: 40px; height: 40px; border-radius: 50%; object-fit: cover; }}.online-status {{ text-align: center; font-size: 0.9em; color: #666; padding: 5px; }}</style>"""
        html_chat = "<div class='online-status'>Tempestes.cat • en línia</div><div class='chat-container'>"
        for speaker, message in chat_log:
            css_class = speaker.lower().replace('.', '-')
            if speaker == "Tempestes.cat":
                html_chat += f"""<div class="message-row"><img src="data:image/png;base64,{logo_base64}" class="profile-pic"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            elif speaker == "Yo":
                html_chat += f"""<div class="message-row message-row-right"><div class="message {css_class}"><strong>{speaker}</strong>{message}</div></div>"""
            else:
                html_chat += f"<div class='message sistema'>{message}</div>"
        html_chat += "</div>"
        st.markdown(css_styles + html_chat, unsafe_allow_html=True)
    with tab2:
        st.subheader("Paràmetres Termodinàmics i de Cisallament")
        param_cols = st.columns(4)
        param_cols[0].metric("CAPE", f"{cape.m:.0f} J/kg"); param_cols[1].metric("CIN", f"{cin.m:.0f} J/kg")
        param_cols[2].metric("PWAT Total", f"{pwat_total.m:.1f} mm"); param_cols[3].metric("0°C", f"{fz_h/1000:.2f} km")
        param_cols[0].metric("LCL", f"{lcl_p.m:.0f} hPa" if lcl_p else "N/A"); param_cols[1].metric("LFC", f"{lfc_p.m:.0f} hPa" if lfc_p else "N/A")
        param_cols[2].metric("EL", f"{el_p.m:.0f} hPa" if el_p else "N/A"); param_cols[3].metric("Shear 0-6", f"{shear_0_6:.1f} m/s")
        param_cols[0].metric("SRH 0-1", f"{srh_0_1:.1f} m²/s²"); param_cols[1].metric("SRH 0-3", f"{srh_0_3:.1f} m²/s²")
        param_cols[2].metric("PWAT 0-4km", f"{pwat_0_4.m:.1f} mm")
        rh_display = "N/A"
        try:
            rh_display = f"{rh_0_4.m*100:.0f}%" if hasattr(rh_0_4, 'm') else f"{rh_0_4*100:.0f}%"
        except: pass
        param_cols[3].metric("RH Mitja 0-4km", rh_display)
    with tab3:
        st.subheader("Representacions Gràfiques del Núvol")
        cloud_cols = st.columns(2)
        with cloud_cols[0]:
            fig_clouds = create_cloud_drawing_figure(p, t, td, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type)
            st.pyplot(fig_clouds, use_container_width=True)
        with cloud_cols[1]:
            fig_structure = create_cloud_structure_figure(p, t, td, ws, wd, convergence_active)
            st.pyplot(fig_structure, use_container_width=True)
    with tab4:
        st.subheader("Simulació de Reflectivitat Radar")
        fig_radar = create_radar_figure(p, t, td, ws, wd)
        st.pyplot(fig_radar, use_container_width=True)

def run_live_mode():
    st.title("🛰️ Mode en Viu: Sondejos Reals")
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Mode Viu)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    if 'live_initialized' not in st.session_state:
        base_files = ['12am.txt'] + [f'{i}am.txt' for i in range(1, 12)] + ['12pm.txt'] + [f'{i}pm.txt' for i in range(1, 12)]
        st.session_state.existing_files = [f for f in base_files if os.path.exists(f)]
        if not st.session_state.existing_files:
            st.error("No s'ha trobat cap arxiu de sondeig per al mode en viu."); return
        now = datetime.now()
        hour_12 = now.hour % 12 if now.hour % 12 != 0 else 12
        am_pm = 'am' if now.hour < 12 else 'pm'
        current_hour_file = f"{hour_12}{am_pm}.txt"
        initial_index = 0
        if current_hour_file in st.session_state.existing_files:
            initial_index = st.session_state.existing_files.index(current_hour_file)
        st.session_state.sounding_index = initial_index
        st.session_state.loaded_sounding_index = -1
        st.session_state.live_initialized = True
    if st.session_state.sounding_index != st.session_state.loaded_sounding_index:
        selected_file = st.session_state.existing_files[st.session_state.sounding_index]
        soundings = parse_all_soundings(selected_file)
        if soundings:
            st.session_state.live_data = soundings[0]
            st.session_state.loaded_sounding_index = st.session_state.sounding_index
        else:
            st.error(f"No s'han pogut carregar dades de {selected_file}"); st.session_state.sounding_index = st.session_state.loaded_sounding_index; return
    with st.sidebar:
        def sync_index_from_selectbox():
            st.session_state.sounding_index = st.session_state.existing_files.index(st.session_state.selectbox_widget)
        st.selectbox("Selecciona una hora:", options=st.session_state.existing_files, index=st.session_state.sounding_index, key='selectbox_widget', on_change=sync_index_from_selectbox)
    main_cols = st.columns([1, 10, 1])
    with main_cols[0]:
        if st.button('←', use_container_width=True, disabled=(st.session_state.sounding_index == 0)):
            st.session_state.sounding_index -= 1; st.rerun()
    with main_cols[2]:
        if st.button('→', use_container_width=True, disabled=(st.session_state.sounding_index >= len(st.session_state.existing_files) - 1)):
            st.session_state.sounding_index += 1; st.rerun()
    data = st.session_state.live_data
    run_display_logic(p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], obs_time=data.get('observation_time', 'Hora no disponible'))

def run_sandbox_mode():
    st.title("🧪 Laboratori de Sondejos")
    with st.sidebar:
        logo_fig = create_logo_figure()
        st.pyplot(logo_fig)
        st.header("Controls (Laboratori)")
        if st.button("⬅️ Tornar a l'inici", use_container_width=True):
            st.session_state.app_mode = 'welcome'; st.rerun()
        st.toggle("Activar convergència", value=st.session_state.get('convergence_active', True), key='convergence_active')
    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings:
            st.error("No s'ha trobat o no s'ha pogut llegir 'sondeigproves.txt'. Aquest mode no pot funcionar."); return
        st.session_state.sandbox_original_data = soundings[0]
        st.session_state.sandbox_p_levels = st.session_state.sandbox_original_data['p_levels'].copy()
        st.session_state.sandbox_t_profile = st.session_state.sandbox_original_data['t_initial'].copy()
        st.session_state.sandbox_td_profile = st.session_state.sandbox_original_data['td_initial'].copy()
        st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True
    with st.sidebar:
        if st.button("🔄 Reiniciar al perfil original", use_container_width=True):
            data = st.session_state.sandbox_original_data
            st.session_state.sandbox_t_profile = data['t_initial'].copy()
            st.session_state.sandbox_td_profile = data['td_initial'].copy()
            st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
            st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
            st.rerun()
        st.markdown("---")
        st.subheader("Modificació Manual")
        sfc_t = st.session_state.sandbox_t_profile[0].magnitude
        new_sfc_t = st.slider("🌡️ Temperatura en Superfície (°C)", -20.0, 50.0, sfc_t, 0.5)
        sfc_td = st.session_state.sandbox_td_profile[0].magnitude
        new_sfc_td = st.slider("💧 Punt de Rosada en Superfície (°C)", -20.0, new_sfc_t, sfc_td, 0.5)
        st.session_state.sandbox_t_profile[0] = new_sfc_t * units.degC
        st.session_state.sandbox_td_profile[0] = new_sfc_td * units.degC
        st.markdown("---")
        st.subheader("Escenaris Predefinits")
        if st.button("❄️ Nevada Severa", use_container_width=True): apply_preset('neu'); st.rerun()
        if st.button("☀️ Calor Extrema", use_container_width=True): apply_preset('calor'); st.rerun()
        if st.button("🌪️ Supercèl·lula Clàssica", use_container_width=True): apply_preset('supercel'); st.rerun()
        if st.button("🌧️ Pluja Estratiforme", use_container_width=True): apply_preset('pluja'); st.rerun()
    run_display_logic(p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile, td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws, wd=st.session_state.sandbox_wd, obs_time="Sondeig de Prova - Mode Laboratori")

# =========================================================================
# === 6. PUNT D'ENTRADA DE L'APLICACIÓ ====================================
# =========================================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Visor de Sondejos")
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'
    if st.session_state.app_mode == 'welcome':
        # Crida a la nova funció de benvinguda animada
        show_welcome_screen_animated()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
