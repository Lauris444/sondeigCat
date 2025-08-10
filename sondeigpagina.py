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


# El pany segueix sent crucial per evitar errors de concurrència.
integrator_lock = threading.Lock()
# =============================================================================
# === 0. FUNCIONS D'ESTIL I PRESENTACIÓ ======================================
# =============================================================================

def show_loading_animation(message="Carregant"):
    """Mostra una animació de càrrega personalitzada amb HTML i CSS."""
    loading_html = f"""
    <style>
        .loading-container {{
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            background: rgba(25,37,81,0.9); z-index: 9999;
        }}
        .loading-svg {{ width: 150px; height: auto; margin-bottom: 20px; }}
        .loading-text {{ color: white; font-size: 1.5rem; font-family: sans-serif; }}
        .loading-text .dot {{ animation: blink 1.4s infinite both; }}
        .loading-text .dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .loading-text .dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{ 0%, 80%, 100% {{ opacity: 0; }} 40% {{ opacity: 1; }} }}
    </style>
    <div class="loading-container">
        <svg class="loading-svg" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
            <path d="M 155.6,66.1 C 155.6,42.9 135.5,23.5 111.4,23.5 C 98.4,23.5 86.8,29.4 79.1,38.7 C 75.2,16.8 57.3,0 36.4,0 C 16.3,0 0,16.3 0,36.4 C 0,56.5 16.3,72.8 36.4,72.8 L 110,72.8 C 110,72.8 110,72.8 110,72.8 C 135,72.8 155.6,93.4 155.6,118.4 C 155.6,143.4 135,164 110,164 L 50, 164" fill="none" stroke="#FFFFFF" stroke-width="8"/>
            <polygon points="120,60 90,110 115,110 100,150 145,90 120,90 130,60" fill="#FFD700" />
        </svg>
        <div class="loading-text">{message}<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

# =============================================================================
# === DICCIONARI D'IDIOMES COMPLET I TRADUÏT ==================================
# =============================================================================

LANGUAGES = {
    'ca': {
        # --- General UI ---
        "lang_selector_label": "Idioma", "app_title": "Analitzador de Sondejos",
        "welcome_title": "TEMPESTES.CAT PRESENTA:", "welcome_subtitle": "Una eina per a la visualització i experimentació amb perfils atmosfèrics.",
        "live_mode_title": "⚠️ Avisos d'avui", "live_mode_text": "Visualitza els sondejos atmosfèrics més recents basats en dades de models per a les zones més actives del dia.", "live_mode_button": "Accedir",
        "sandbox_mode_title": "🧪 Laboratori", "sandbox_mode_text": "Aprèn de forma interactiva com es formen els fenòmens severs modificant pas a pas un sondeig o experimenta lliurement.", "sandbox_mode_button": "Accedir al Laboratori",
        "manual_mode_title": "✍️ Mode Manual", "manual_mode_text": "Enganxa el text d'un sondeig en format estàndard i l'analitzarem a l'instant, sense necessitat d'arxius externs.", "manual_mode_button": "Analitzar el teu Sondeig",
        "coming_soon_title": "🗺️ Pròximament...", "coming_soon_subtitle": "MAPES DE VENTS",
        "back_to_start_button": "⬅️ Tornar a l'inici", "loading_message": "Carregant",
        "analysis_zones_title": "Anàlisi de Zones",
        "follow_todays_zone_button": "Segueix la zona de canvis d'avui",
        "todays_changes_title": "Canvis d'avui",
        "select_region_desc": "Selecciona la comarca que vols analitzar. Cada zona representa un perfil atmosfèric diferent basat en les previsions més recents.",
        "most_notable_zone_title": "Zona Més Destacable",
        "most_notable_zone_desc": "El perfil amb el major potencial per a fenòmens significatius.",
        "interesting_zone_title": "Zona Interessant",
        "interesting_zone_desc": "Un perfil que presenta algunes característiques d'interès.",
        "loading_region": "Carregant {region}",


"tutorial_lab_title": "🧪 Laboratori de Sondejos - Mode Tutorial",
"tutorial_title": "Tutorial: {scenario}",
"tutorial_congrats": "🎉 Enhorabona, has completat el tutorial! 🎉",
"tutorial_finish_button": "Finalitzar i Veure Resultat",
"tutorial_abandon_button": "Abandonar Tutorial",
"tutorial_sleet_step1_title": "Pas 1: La Fàbrica de Neu",
"tutorial_sleet_step1_instr": "Hem carregat un perfil d'aiguaneu. Observa a les capes altes (sobre 700 hPa). Les temperatures són negatives. Aquí es formen els flocs de neu.",
"tutorial_sleet_step1_button": "Entès, pas 1/3 →",
"tutorial_sleet_step1_expl": "Aquí és on es formen els flocs de neu inicials. De moment, tot correcte.",
"tutorial_sleet_step2_title": "Pas 2: La Capa Càlida que ho fon tot",
"tutorial_sleet_step2_instr": "Ara mira la capa mitjana (~850 hPa). La temperatura supera els 0°C. Aquest és el problema: els flocs es fonen i es converteixen en pluja.",
"tutorial_sleet_step2_button": "Ho veig, pas 2/3 →",
"tutorial_sleet_step2_expl": "Quan els flocs de neu cauen a través d'aquesta capa càlida, es fonen i es converteixen en gotes de pluja.",
"tutorial_sleet_step3_title": "Pas 3: Recongelació a Superfície",
"tutorial_sleet_step3_instr": "Finalment, a prop de terra, la temperatura torna a ser negativa. Les gotes de pluja es tornen a congelar just abans de tocar el terra.",
"tutorial_sleet_step3_button": "Entès, pas 3/3 →",
"tutorial_sleet_step3_expl": "Això és el que produeix l'aiguaneu (sleet) o la perillosa pluja gelant.",
"tutorial_sleet_step4_title": "Conclusió i Repte Final",
"tutorial_sleet_step4_instr": "Has analitzat un perfil clàssic d'aiguaneu! Ara saps que una capa càlida intermèdia és la culpable.",
"tutorial_sleet_step4_button": "Finalitzar Tutorial",
"tutorial_sleet_step4_expl": "Repte: Ara que has acabat, fes clic a 'Finalitzar'. Utilitza l'eina '❄️ Refredar Capa Mitjana-Baixa' a la barra lateral i veuràs com converteixes aquest perfil en una nevada perfecta!",
"manual_dialog_error": "Text del sondeig no vàlid o buit. Si us plau, tanca i enganxa les dades.",
"manual_dialog_success": "Sondeig processat correctament amb una elevació de {elevation} m ({pressure:.1f} hPa) i orografia de {orography} m.",
"manual_dialog_process_error": "No s'ha pogut processar el text. Assegura't que el format és correcte.",




         "caption_tornado": "Un tornado format sota una supercèl·lula.",
"caption_funnel": "Una tuba (funnel cloud) baixant de la base del núvol.",
"caption_wallcloud": "Un mur de núvols (wall cloud) ben definit.",
"caption_shelfcloud": "Un espectacular núvol de prestatge (shelf cloud).",
"caption_scud": "Base rugosa amb fragments de núvols (scud).",
"caption_supercell": "Una supercèl·lula organitzada.",
"caption_lenticularis": "Núvols lenticulars, indicant fort vent en altura.",
"caption_castellanus": "Altocumulus Castellanus, indicant inestabilitat en capes mitjanes.",
"caption_fractus": "Cumulus Fractus, núvols fragmentats.",
"caption_cumulonimbus": "Un Cumulonimbus, el núvol de tempesta per excel·lència.",
"caption_congestus": "Cumulus Congestus, amb gran desenvolupament vertical.",
"caption_humilis": "Cumulus Humilis, núvols de bon temps.",
"caption_cirrus": "Núvols alts i prims formats per cristalls de gel.",
"caption_altostratus": "Cel cobert per Altostratus, pot indicar pluja propera.",
"caption_nimbostratus": "Cel cobert per Nimbostratus, associat a pluja contínua.",
"caption_stratus": "Una capa baixa i grisa de Stratus, semblant a la boira.",
"caption_sleet": "Precipitació en forma d'aiguaneu (sleet).",
"caption_snow": "Una nevada cobrint el paisatge.",
        
        

        # --- Laboratori ---
        "sandbox_welcome_title": "🧪 Benvingut al Laboratori!", "sandbox_welcome_desc": "Tria com vols començar. Pots seguir un tutorial guiat per aprendre els conceptes clau o anar directament al mode lliure per experimentar per tu mateix.",
        "sandbox_tutorial_supercell_title": "🌪️ Tutorial: Supercèl·lula", "sandbox_tutorial_supercell_desc": "Aprèn a crear un entorn amb una inestabilitat explosiva i el cisallament necessari per a les tempestes més severes i organitzades.", "sandbox_tutorial_supercell_button": "Començar Tutorial de Supercèl·lula",
        "sandbox_tutorial_sleet_title": "💧 Tutorial: Aiguaneu", "sandbox_tutorial_sleet_desc": "Analitza una situació d'aiguaneu, identifica la capa càlida culpable i aprèn com transformar la precipitació en neu.", "sandbox_tutorial_sleet_button": "Començar Tutorial d'Aiguaneu",
        "sandbox_freemode_title": "🛠️ Mode Lliure", "sandbox_freemode_desc": "Salta directament a l'acció. Tindràs el control total sobre el perfil atmosfèric des del principi per crear els teus propis escenaris.", "sandbox_freemode_button": "Anar al Mode Lliure",
        "sandbox_toolbox_header": "Caixa d'Eines", "sandbox_thermo_header": "Modificacions Termodinàmiques",
        "sandbox_layer_sfc": "Superfície (>950hPa)", "sandbox_layer_low": "Baixes (950-800hPa)", "sandbox_layer_lowmid": "Mitjanes-Baixes (800-600hPa)",
        "sandbox_shear_header": "Cisallament del Vent", "sandbox_shear_low": "Capes Baixes", "sandbox_shear_mid": "Capes Mitjanes",
        "sandbox_reset_wind": "🚫 Reiniciar Vents", "sandbox_reset_all": "🔄 Reiniciar Tot al Perfil Original",
         "analysis_zones_title": "Anàlisi de Zones",
        "follow_todays_zone_button": "Segueix la zona de canvis d'avui",
        "todays_changes_title": "Canvis d'avui",
        "select_region_desc": "Selecciona la comarca que vols analitzar. Cada zona representa un perfil atmosfèric diferent basat en les previsions més recents.",
        "most_notable_zone_title": "Zona Més Destacable",
        "most_notable_zone_desc": "El perfil amb el major potencial per a fenòmens significatius.",
        "interesting_zone_title": "Zona Interessant",
        "interesting_zone_desc": "Un perfil que presenta algunes característiques d'interès.",
        "loading_region": "Carregant {region}",

        # --- Mode Manual ---
        "manual_mode_page_title": "✍️ Analitzador de Sondeig Manual", "manual_mode_page_desc": "Enganxa aquí el text complet del teu sondeig. L'analitzador processarà les dades i mostrarà els resultats a sota.",
        "manual_textarea_label": "Introdueix les dades del sondeig:", "manual_textarea_placeholder": "Enganxa aquí el text del sondeig...", "manual_analyze_button": "Analitzar Sondeig",
        "manual_dialog_title": "Anàlisi Inicial Personalitzada", "manual_dialog_header": "Dades del Lloc de Sondeig", "manual_dialog_desc": "Introdueix l'elevació base i l'altura de l'orografia per a una anàlisi precisa.",
        "manual_dialog_elevation_label": "**1. Altura sobre el nivell del mar (en metres):**", "manual_dialog_orography_label": "**2. Altura de les muntanyes del voltant (en metres):**", "manual_dialog_button": "Acceptar i Generar Anàlisi Completa",

        # --- Vista d'Anàlisi ---
        "analyst_assistant_tab": "💬 Assistent d'Anàlisi", "parameters_tab": "📊 Paràmetres", "hodograph_tab": "📈 Hodògraf", "orography_tab": "⛰️ Orografia",
        "visualization_tab": "☁️ Visualització", "cloud_types_tab": "📋 Tipus de Núvols", "radar_tab": "📡 Radar",
        "skewt_diagram_title": "Diagrama Skew-T", "hodograph_title": "Hodògraf del Perfil de Vents", "orography_graph_title": "Potencial d'Activació per Orografia",
        "cloud_viz_title": "Representacions Gràfiques del Núvol", "cloud_drawing_title": "Visualització del Núvol", "cloud_structure_title": "Estructura Vertical i Cisallament",
        "cloud_list_title": "Llista de Gèneres de Núvols Probables", "cloud_list_desc": "Aquesta llista es basa en el balanç entre energia (CAPE), inhibició (CIN), humitat (HR) i vent a diferents capes.",
        "representative_images_title": "Imatges Representatives", "radar_sim_title": "Simulació de Reflectivitat Radar",

        # --- Paràmetres ---
        "param_sfc_temp": "Temperatura Superficial", "param_usable_cape": "CAPE Utilitzable", "param_usable_cape_help": "CAPE brut menys la inhibició (CIN). L'energia real disponible.",
        "param_srh_01": "SRH 0-1km", "param_cin": "CIN (Fre)", "param_lcl": "LCL (AGL)", "param_srh_03": "SRH 0-3km", "param_cape_raw": "CAPE (Brut)",
        "param_lfc": "LFC (AGL)", "param_pwat": "PWAT Total", "param_shear_06": "Shear 0-6km", "param_el": "EL (MSL)", "param_rh_04": "RH Mitja 0-4km",

        # --- Avisos Públics ---
        "warn_no_significant_title": "SENSE AVISOS SIGNIFICATIUS", "warn_no_significant_desc": "Les condicions actuals no presenten riscos meteorològics destacables.",
        "warn_snow_title": "AVÍS PER NEVADA", "warn_snow_desc": "Perfil favorable per a nevades. Temperatura en superfície de {temp:.1f}°C i absència de capes càlides.",
        "warn_sleet_title": "AVÍS PER AIGUANEU", "warn_sleet_desc": "Una capa càlida en alçada ({warm_layer_temp:.1f}°C) i fred intens en superfície ({temp:.1f}°C) afavoreixen l'aiguaneu (glaçons).",
        "warn_ice_rain_title": "AVÍS PER PLUJA GELANT / AIGUANEU", "warn_ice_rain_desc": "Risc alt de pluja gelant o aiguaneu per capa càlida en alçada i Tª superficial de {temp:.1f}°C. Perill a les carreteres.",
        "warn_high_lfc_title": "SENSE AVISOS SIGNIFICATIUS (PER ARA)", "warn_high_lfc_desc": "Tot i que hi ha energia potencial, el Nivell de Convecció Lliure (LFC) és molt alt ({lfc_agl:.0f} m). Això indica una forta 'tapadera' que podria impedir la formació de tempestes.",
        "warn_tornado_title": "AVÍS PER TORNADO", "warn_tornado_desc": "Condicions extremes (Energia Neta {cape:.0f}, SRH {srh:.0f}). Risc molt alt de supercèl·lules tornàdiques.",
        "warn_extreme_severe_title": "AVÍS PER TEMPS SEVER EXTREM", "warn_extreme_severe_desc": "Potencial per a supercèl·lules destructives (Energia Neta {cape:.0f}). Risc molt alt de calamarsa gran (>5cm) i vents severs.",
        "warn_severe_title": "AVÍS PER TEMPS SEVER", "warn_severe_desc": "Atmosfera molt inestable i organitzada (Energia Neta {cape:.0f}, Shear {shear:.1f}). Risc de calamarsa gran i/o ratxes de vent molt fortes.",
        "warn_strong_storms_title": "AVÍS PER TEMPESTES FORTES", "warn_strong_storms_desc": "Inestabilitat elevada (Energia Neta {cape:.0f}). Risc de tempestes amb calamarsa i forts vents localitzats.",
        "warn_moderate_storms_title": "RISC DE TEMPESTES MODERADES", "warn_moderate_storms_desc": "Potencial per a tempestes organitzades (Energia Neta {cape:.0f}). Poden deixar ruixats forts i calamarsa petita.",
        "warn_isolated_storms_title": "RISC DE RUIXATS I TEMPESTES AÏLLADES", "warn_isolated_storms_desc": "Inestabilitat baixa (Energia Neta {cape:.0f}). Es poden formar alguns ruixats o tempestes de curta durada.",
        "warn_heavy_rain_title": "AVÍS PER PLUGES INTENSES", "warn_heavy_rain_desc": "Atmosfera molt humida ({pwat:.1f} mm) i saturada a nivells baixos ({rh:.0f}% HR). Risc de pluges eficients.",

        # --- Text del Xat ---
        "chat_analyst": "Analista", "chat_user": "Usuari",
        "chat_winter_intro": "Estem en un escenari de temps hivernal amb una temperatura en superfície de {temp:.1f}°C.",
        "chat_winter_q_snow": "Hi ha potencial per a neu?", "chat_winter_a_no_snow": "No realment. Les capes altes no són prou fredes per a formar flocs de neu de manera eficient. La precipitació, si n'hi ha, seria en forma de pluja.",
        "chat_winter_q_cold_aloft": "És prou fred a dalt per a nevar?", "chat_winter_a_cold_aloft": "Sí, les capes superiors a 700 hPa són una 'fàbrica de neu' perfecta. Els flocs de neu es formaran sin problemes.",
        "chat_winter_q_falling": "I què passa quan els flocs cauen?", "chat_winter_a_warm_layer": "Aquí ve la clau: en caure, es troben amb una capa càlida d'uns **{temp:.1f}°C**. Això fondrà els flocs i els convertirà en gotes de pluja.",
        "chat_winter_q_surface": "Llavors, què arribarà a terra?", "chat_winter_a_sleet": "Com que la superfície està a 0°C o menys, aquestes gotes de pluja es tornaran a congelar just abans de tocar el terra. El resultat serà **aiguaneu** (sleet) o la perillosa **pluja gelant**.",
        "chat_winter_a_rain": "Tot i la neu en alçada, la capa càlida i la temperatura positiva en superfície faran que la precipitació final sigui **pluja**.",
        "chat_winter_a_cold_column": "La columna atmosfèrica es manté per sota de 0°C durant tot el seu recorregut. Els flocs de neu no es fondran.", "chat_winter_q_then": "Llavors...", "chat_winter_a_snow": "Exacte. Tindrem una **nevada** a la superfície!",
        "chat_detailed_intro": "Hola! Anem a analitzar aquest perfil atmosfèric, que comença a una elevació de {height:.0f} metres.",
        "chat_detailed_energy_balance": "Primer, avaluem el balanç energètic. Tenim un CAPE (energia potencial) de **{cape:.0f} J/kg**.",
        "chat_detailed_q_cin": "I què passa amb la 'tapadora' (CIN)? Pot frenar-ho?", "chat_detailed_a_cin": "Molt bona pregunta. El CIN (inhibició) és de **{cin:.0f} J/kg**. Aquest valor actua com un fre. Si restem aquest fre a l'energia potencial, ens queda un **CAPE utilitzable de {usable_cape:.0f} J/kg**.",
        "chat_detailed_stable": "Com que l'energia neta és molt baixa, la 'tapadora' és massa forta. És **molt poc probable** que es formin tempestes significatives des de la superfície, malgrat el CAPE inicial. L'atmosfera és estable en la pràctica.",
        "chat_detailed_go_ahead": "Aquesta és l'energia realment disponible per formar tempestes. Ara que sabem que tenim 'llum verda', podem analitzar la resta d'ingredients.",
        "chat_detailed_cape_desc_extreme": "un valor extremadament alt. Això significa que hi ha un potencial explosiu per a corrents ascendents molt violents.",
        "chat_detailed_cape_desc_strong": "un valor que indica una inestabilitat forta, suficient per a tempestes intenses.",
        "chat_detailed_cape_desc_moderate": "un valor moderat. Hi ha energia per a ruixats o alguna tempesta.",
        "chat_detailed_q_orography": "I una muntanya de {oro_height} m podria ajudar a superar el CIN restant?", "chat_detailed_a_no_lfc": "En aquest cas no hi ha Nivell de Convecció Lliure (LFC), així que l'orografia no podrà iniciar convecció profunda.",
        "chat_detailed_a_orography_yes": "Sí! L'orografia de {oro_height} m **ÉS prou alta** per forçar l'aire a superar el LFC (situat a {lfc_agl:.0f} m sobre el terra). Pot actuar com a disparador definitiu!",
        "chat_detailed_a_orography_no": "En aquest cas, l'orografia de {oro_height} m **NO és prou alta** per arribar al LFC (situat a {lfc_agl:.0f} m sobre el terra). Necessitarem un altre mecanisme de tret (com un front).",
        "chat_detailed_q_moisture": "Tenim prou 'combustible' (humitat) per aprofitar aquesta energia?", "chat_detailed_a_moisture": "L'aigua precipitable és de {pwat:.1f} mm en els primers 4 km. {pwat_analysis}",
        "chat_detailed_q_organization": "Perfecte. I les tempestes, s'organitzaran o seran caòtiques?", "chat_detailed_a_organization": "Aquí entra en joc el cisallament del vent (0-6 km), que és de {shear:.1f} m/s. {shear_analysis}",
        "chat_detailed_q_tornado": "Això vol dir que hi ha risc de tornados?", "chat_detailed_a_tornado": "Per això mirem l'Helicitat Relativa a la Tempesta (SRH 0-1km), que és de {srh:.1f} m²/s². {srh_analysis}",
        "chat_detailed_summary": "**En resum:** {verdict}",
        "pwat_analysis_dry": "És un ambient relativament sec. Això podria limitar la intensitat de la precipitació.", "pwat_analysis_moist": "Hi ha humitat suficient per alimentar tempestes i generar pluja moderada o forta.", "pwat_analysis_loaded": "L'atmosfera està molt carregada d'humitat. Si es desenvolupen tempestes, tenen potencial per a ser molt eficients i deixar grans acumulacions de pluja.",
        "shear_analysis_weak": "És feble. Les tempestes que es formin seran probablement de cicle de vida curt i desorganitzades (tempestes unicel·lulars).", "shear_analysis_moderate": "És moderat. Això és suficient per organitzar les tempestes en sistemes multicel·lulars més duradors i amb més potencial.", "shear_analysis_strong": "És fort. Aquest és l'ingredient clau que ajuda les tempestes a rotar i a evolucionar cap a supercèl·lules, molt més organitzades i severes.",
        "srh_analysis_high_risk": "Sí, el risc és significatiu. Valors d'SRH per sobre de 150 m²/s² amb una base del núvol baixa (LCL a {lcl_agl:.0f} m sobre el terra) són un indicador clàssic de potencial tornàdic.",
        "srh_analysis_moderate_risk": "Indica una rotació considerable a nivells baixos. El risc de tornados no és extrem, però s'han de vigilar possibles embuts (funnels) o tubes.",
        "srh_analysis_low_risk": "La rotació a nivells baixos no és especialment forta. El risc principal serien els vents forts lineals i la calamarsa, més que no pas els tornados.",
    },
    'es': {
        # --- General UI ---
        "lang_selector_label": "Idioma", "app_title": "Analizador de Sondeos",
        "welcome_title": "TEMPESTES.CAT PRESENTA:", "welcome_subtitle": "Una herramienta para la visualización y experimentación con perfiles atmosféricos.",
        "live_mode_title": "⚠️ Avisos de hoy", "live_mode_text": "Visualiza los sondeos atmosféricos más recientes basados en datos de modelos para las zonas más activas del día.", "live_mode_button": "Acceder",
        "sandbox_mode_title": "🧪 Laboratorio", "sandbox_mode_text": "Aprende de forma interactiva cómo se forman los fenómenos severos modificando paso a paso un sondeo o experimenta libremente.", "sandbox_mode_button": "Acceder al Laboratorio",
        "manual_mode_title": "✍️ Modo Manual", "manual_mode_text": "Pega el texto de un sondeo en formato estándar y lo analizaremos al instante, sin necesidad de archivos externos.", "manual_mode_button": "Analizar tu Sondeo",
        "coming_soon_title": "🗺️ Próximamente...", "coming_soon_subtitle": "MAPAS DE VIENTOS",
        "back_to_start_button": "⬅️ Volver al inicio", "loading_message": "Cargando",
        "analysis_zones_title": "Análisis de Zonas",
        "follow_todays_zone_button": "Sigue la zona de cambios de hoy",
        "todays_changes_title": "Cambios de hoy",
        "select_region_desc": "Selecciona la comarca que quieres analizar. Cada zona representa un perfil atmosférico diferente basado en las previsiones más recientes.",
        "most_notable_zone_title": "Zona Más Destacable",
        "most_notable_zone_desc": "El perfil con el mayor potencial para fenómenos significativos.",
        "interesting_zone_title": "Zona Interesante",
        "interesting_zone_desc": "Un perfil que presenta algunas características de interés.",
        "loading_region": "Cargando {region}",

        "tutorial_lab_title": "🧪 Laboratorio de Sondeos - Modo Tutorial",
"tutorial_title": "Tutorial: {scenario}",
"tutorial_congrats": "🎉 ¡Enhorabuena, has completado el tutorial! 🎉",
"tutorial_finish_button": "Finalizar y Ver Resultado",
"tutorial_abandon_button": "Abandonar Tutorial",
"tutorial_sleet_step1_title": "Paso 1: La Fábrica de Nieve",
"tutorial_sleet_step1_instr": "Hemos cargado un perfil de aguanieve. Observa en las capas altas (sobre 700 hPa). Las temperaturas son negativas. Aquí se forman los copos de nieve.",
"tutorial_sleet_step1_button": "Entendido, paso 1/3 →",
"tutorial_sleet_step1_expl": "Aquí es donde se forman los copos de nieve iniciales. Por ahora, todo correcto.",
"tutorial_sleet_step2_title": "Paso 2: La Capa Cálida que lo derrite todo",
"tutorial_sleet_step2_instr": "Ahora mira la capa media (~850 hPa). La temperatura supera los 0°C. Este es el problema: los copos se derriten y se convierten en lluvia.",
"tutorial_sleet_step2_button": "Lo veo, paso 2/3 →",
"tutorial_sleet_step2_expl": "Cuando los copos de nieve caen a través de esta capa cálida, se derriten y se convierten en gotas de lluvia.",
"tutorial_sleet_step3_title": "Paso 3: Recongelación en Superficie",
"tutorial_sleet_step3_instr": "Finalmente, cerca del suelo, la temperatura vuelve a ser negativa. Las gotas de lluvia se vuelven a congelar justo antes de tocar el suelo.",
"tutorial_sleet_step3_button": "Entendido, paso 3/3 →",
"tutorial_sleet_step3_expl": "Esto es lo que produce el aguanieve (sleet) o la peligrosa lluvia engelante.",
"tutorial_sleet_step4_title": "Conclusión y Reto Final",
"tutorial_sleet_step4_instr": "¡Has analizado un perfil clásico de aguanieve! Ahora sabes que una capa cálida intermedia es la culpable.",
"tutorial_sleet_step4_button": "Finalizar Tutorial",
"tutorial_sleet_step4_expl": "Reto: Ahora que has terminado, haz clic en 'Finalizar'. Utiliza la herramienta '❄️ Enfriar Capa Media-Baja' en la barra lateral y verás cómo conviertes este perfil en una nevada perfecta.",
"manual_dialog_error": "Texto del sondeo no válido o vacío. Por favor, cierra y pega los datos.",
"manual_dialog_success": "Sondeo procesado correctamente con una elevación de {elevation} m ({pressure:.1f} hPa) y orografía de {orography} m.",
"manual_dialog_process_error": "No se ha podido procesar el texto. Asegúrate de que el formato es correcto.",

        

        "caption_tornado": "Un tornado formado bajo una supercélula.",
"caption_funnel": "Una tuba (funnel cloud) descendiendo de la base de la nube.",
"caption_wallcloud": "Una nube pared (wall cloud) bien definida.",
"caption_shelfcloud": "Una espectacular nube de estantería (shelf cloud).",
"caption_scud": "Base irregular con fragmentos de nubes (scud).",
"caption_supercell": "Una supercélula organizada.",
"caption_lenticularis": "Nubes lenticulares, indicando fuerte viento en altura.",
"caption_castellanus": "Altocumulus Castellanus, indicando inestabilidad en capas medias.",
"caption_fractus": "Cumulus Fractus, nubes fragmentadas.",
"caption_cumulonimbus": "Un Cumulonimbus, la nube de tormenta por excelencia.",
"caption_congestus": "Cumulus Congestus, con gran desarrollo vertical.",
"caption_humilis": "Cumulus Humilis, nubes de buen tiempo.",
"caption_cirrus": "Nubes altas y delgadas formadas por cristales de hielo.",
"caption_altostratus": "Cielo cubierto por Altostratus, puede indicar lluvia próxima.",
"caption_nimbostratus": "Cielo cubierto por Nimbostratus, asociado a lluvia continua.",
"caption_stratus": "Una capa baja y gris de Stratus, similar a la niebla.",
"caption_sleet": "Precipitación en forma de aguanieve (sleet).",
"caption_snow": "Una nevada cubriendo el paisaje.",

        # --- Laboratorio ---
        "sandbox_welcome_title": "🧪 ¡Bienvenido al Laboratorio!", "sandbox_welcome_desc": "Elige cómo quieres empezar. Puedes seguir un tutorial guiado para aprender los conceptos clave o ir directamente al modo libre para experimentar por ti mismo.",
        "sandbox_tutorial_supercell_title": "🌪️ Tutorial: Supercélula", "sandbox_tutorial_supercell_desc": "Aprende a crear un entorno con una inestabilidad explosiva y la cizalladura necesaria para las tormentas más severas y organizadas.", "sandbox_tutorial_supercell_button": "Empezar Tutorial de Supercélula",
        "sandbox_tutorial_sleet_title": "💧 Tutorial: Aguanieve", "sandbox_tutorial_sleet_desc": "Analiza una situación de aguanieve, identifica la capa cálida culpable y aprende cómo transformar la precipitación en nieve.", "sandbox_tutorial_sleet_button": "Empezar Tutorial de Aguanieve",
        "sandbox_freemode_title": "🛠️ Modo Libre", "sandbox_freemode_desc": "Salta directamente a la acción. Tendrás el control total sobre el perfil atmosférico desde el principio para crear tus propios escenarios.", "sandbox_freemode_button": "Ir al Modo Libre",
        "sandbox_toolbox_header": "Caja de Herramientas", "sandbox_thermo_header": "Modificaciones Termodinámicas",
        "sandbox_layer_sfc": "Superficie (>950hPa)", "sandbox_layer_low": "Bajas (950-800hPa)", "sandbox_layer_lowmid": "Medias-Bajas (800-600hPa)",
        "sandbox_shear_header": "Cizalladura del Viento", "sandbox_shear_low": "Capas Bajas", "sandbox_shear_mid": "Capas Medias",
        "sandbox_reset_wind": "🚫 Reiniciar Vientos", "sandbox_reset_all": "🔄 Reiniciar Todo al Perfil Original",

        "analysis_zones_title": "Análisis de Zonas",
        "follow_todays_zone_button": "Sigue la zona de cambios de hoy",
        "todays_changes_title": "Cambios de hoy",
        "select_region_desc": "Selecciona la comarca que quieres analizar. Cada zona representa un perfil atmosférico diferente basado en las previsiones más recientes.",
        "most_notable_zone_title": "Zona Más Destacable",
        "most_notable_zone_desc": "El perfil con el mayor potencial para fenómenos significativos.",
        "interesting_zone_title": "Zona Interesante",
        "interesting_zone_desc": "Un perfil que presenta algunas características de interés.",
        "loading_region": "Cargando {region}",


        # --- Modo Manual ---
        "manual_mode_page_title": "✍️ Analizador de Sondeo Manual", "manual_mode_page_desc": "Pega aquí el texto completo de tu sondeo. El analizador procesará los datos y mostrará los resultados abajo.",
        "manual_textarea_label": "Introduce los datos del sondeo:", "manual_textarea_placeholder": "Pega aquí el texto del sondeo...", "manual_analyze_button": "Analizar Sondeo",
        "manual_dialog_title": "Análisis Inicial Personalizado", "manual_dialog_header": "Datos del Lugar del Sondeo", "manual_dialog_desc": "Introduce la elevación base y la altura de la orografía para un análisis preciso.",
        "manual_dialog_elevation_label": "**1. Altura sobre el nivel del mar (en metros):**", "manual_dialog_orography_label": "**2. Altura de las montañas de alrededor (en metros):**", "manual_dialog_button": "Aceptar y Generar Análisis Completo",

        # --- Vista d'Anàlisi ---
        "analyst_assistant_tab": "💬 Asistente de Análisis", "parameters_tab": "📊 Parámetros", "hodograph_tab": "📈 Hodógrafo", "orography_tab": "⛰️ Orografía",
        "visualization_tab": "☁️ Visualización", "cloud_types_tab": "📋 Tipos de Nubes", "radar_tab": "📡 Radar",
        "skewt_diagram_title": "Diagrama Skew-T", "hodograph_title": "Hodógrafo del Perfil de Vientos", "orography_graph_title": "Potencial de Activación por Orografía",
        "cloud_viz_title": "Representaciones Gráficas de la Nube", "cloud_drawing_title": "Visualización de la Nube", "cloud_structure_title": "Estructura Vertical y Cizalladura",
        "cloud_list_title": "Lista de Géneros de Nubes Probables", "cloud_list_desc": "Esta lista se basa en el balance entre energía (CAPE), inhibición (CIN), humedad (HR) y viento en diferentes capas.",
        "representative_images_title": "Imágenes Representativas", "radar_sim_title": "Simulación de Reflectividad Radar",

        # --- Paràmetres ---
        "param_sfc_temp": "Temperatura Superficial", "param_usable_cape": "CAPE Útil", "param_usable_cape_help": "CAPE bruto menos la inhibición (CIN). La energía real disponible.",
        "param_srh_01": "SRH 0-1km", "param_cin": "CIN (Freno)", "param_lcl": "LCL (AGL)", "param_srh_03": "SRH 0-3km", "param_cape_raw": "CAPE (Bruto)",
        "param_lfc": "LFC (AGL)", "param_pwat": "PWAT Total", "param_shear_06": "Cizalladura 0-6km", "param_el": "EL (MSL)", "param_rh_04": "HR Media 0-4km",

        # --- Avisos Públics ---
        "warn_no_significant_title": "SIN AVISOS SIGNIFICATIVOS", "warn_no_significant_desc": "Las condiciones actuales no presentan riesgos meteorológicos destacables.",
        "warn_snow_title": "AVISO POR NEVADA", "warn_snow_desc": "Perfil favorable para nevadas. Temperatura en superficie de {temp:.1f}°C y ausencia de capas cálidas.",
        "warn_sleet_title": "AVISO POR AGUANIEVE", "warn_sleet_desc": "Una capa cálida en altura ({warm_layer_temp:.1f}°C) y frío intenso en superficie ({temp:.1f}°C) favorecen el aguanieve (gránulos de hielo).",
        "warn_ice_rain_title": "AVISO POR LLUVIA ENGELANTE / AGUANIEVE", "warn_ice_rain_desc": "Riesgo alto de lluvia engelante o aguanieve por capa cálida en altura y Tª superficial de {temp:.1f}°C. Peligro en las carreteras.",
        "warn_high_lfc_title": "SIN AVISOS SIGNIFICATIVOS (POR AHORA)", "warn_high_lfc_desc": "Aunque hay energía potencial, el Nivel de Convección Libre (LFC) es muy alto ({lfc_agl:.0f} m). Esto indica una fuerte 'tapadera' que podría impedir la formación de tormentas.",
        "warn_tornado_title": "AVISO POR TORNADO", "warn_tornado_desc": "Condiciones extremas (Energía Neta {cape:.0f}, SRH {srh:.0f}). Riesgo muy alto de supercélulas tornádicas.",
        "warn_extreme_severe_title": "AVISO POR TIEMPO SEVERO EXTREMO", "warn_extreme_severe_desc": "Potencial para supercélulas destructivas (Energía Neta {cape:.0f}). Riesgo muy alto de granizo grande (>5cm) y vientos severos.",
        "warn_severe_title": "AVISO POR TIEMPO SEVERO", "warn_severe_desc": "Atmósfera muy inestable y organizada (Energía Neta {cape:.0f}, Cizalladura {shear:.1f}). Riesgo de granizo grande y/o rachas de viento muy fuertes.",
        "warn_strong_storms_title": "AVISO POR TORMENTAS FUERTES", "warn_strong_storms_desc": "Inestabilidad elevada (Energía Neta {cape:.0f}). Riesgo de tormentas con granizo y fuertes vientos localizados.",
        "warn_moderate_storms_title": "RIESGO DE TORMENTAS MODERADAS", "warn_moderate_storms_desc": "Potencial para tormentas organizadas (Energía Neta {cape:.0f}). Pueden dejar chubascos fuertes y granizo pequeño.",
        "warn_isolated_storms_title": "RIESGO DE CHUBASCOS Y TORMENTAS AISLADAS", "warn_isolated_storms_desc": "Inestabilidad baja (Energía Neta {cape:.0f}). Se pueden formar algunos chubascos o tormentas de corta duración.",
        "warn_heavy_rain_title": "AVISO POR LLUVIAS INTENSAS", "warn_heavy_rain_desc": "Atmósfera muy húmeda ({pwat:.1f} mm) y saturada a niveles bajos ({rh:.0f}% HR). Riesgo de lluvias eficientes.",

        # --- Text del Xat ---
        "chat_analyst": "Analista", "chat_user": "Usuario",
        "chat_winter_intro": "Estamos en un escenario de tiempo invernal con una temperatura en superficie de {temp:.1f}°C.",
        "chat_winter_q_snow": "¿Hay potencial para nieve?", "chat_winter_a_no_snow": "No realmente. Las capas altas no son lo suficientemente frías para formar copos de nieve de manera eficiente. La precipitación, si la hay, sería en forma de lluvia.",
        "chat_winter_q_cold_aloft": "¿Hace suficiente frío arriba para nevar?", "chat_winter_a_cold_aloft": "Sí, las capas superiores a 700 hPa son una 'fábrica de nieve' perfecta. Los copos de nieve se formarán sin problemas.",
        "chat_winter_q_falling": "¿Y qué pasa cuando los copos caen?", "chat_winter_a_warm_layer": "Aquí viene la clave: al caer, se encuentran con una capa cálida de unos **{temp:.1f}°C**. Esto derretirá los copos y los convertirá en gotas de lluvia.",
        "chat_winter_q_surface": "Entonces, ¿qué llegará al suelo?", "chat_winter_a_sleet": "Como la superficie está a 0°C o menos, estas gotas de lluvia se volverán a congelar justo antes de tocar el suelo. El resultado será **aguanieve** (sleet) o la peligrosa **lluvia engelante**.",
        "chat_winter_a_rain": "A pesar de la nieve en altura, la capa cálida y la temperatura positiva en superficie harán que la precipitación final sea **lluvia**.",
        "chat_winter_a_cold_column": "La columna atmosférica se mantiene por debajo de 0°C durante todo su recorrido. Los copos de nieve no se derretirán.", "chat_winter_q_then": "Entonces...", "chat_winter_a_snow": "Exacto. ¡Tendremos una **nevada** en la superficie!",
        "chat_detailed_intro": "¡Hola! Vamos a analizar este perfil atmosférico, que comienza a una elevación de {height:.0f} metros.",
        "chat_detailed_energy_balance": "Primero, evaluemos el balance energético. Tenemos un CAPE (energía potencial) de **{cape:.0f} J/kg**.",
        "chat_detailed_q_cin": "¿Y qué pasa con la 'tapadera' (CIN)? ¿Puede frenarlo?", "chat_detailed_a_cin": "Muy buena pregunta. El CIN (inhibición) es de **{cin:.0f} J/kg**. Este valor actúa como un freno. Si restamos este freno a la energía potencial, nos queda un **CAPE útil de {usable_cape:.0f} J/kg**.",
        "chat_detailed_stable": "Como la energía neta es muy baja, la 'tapadera' es demasiado fuerte. Es **muy poco probable** que se formen tormentas significativas desde la superficie, a pesar del CAPE inicial. La atmósfera es estable en la práctica.",
        "chat_detailed_go_ahead": "Esta es la energía realmente disponible para formar tormentas. Ahora que sabemos que tenemos 'luz verde', podemos analizar el resto de ingredientes.",
        "chat_detailed_cape_desc_extreme": "un valor extremadamente alto. Esto significa que hay un potencial explosivo para corrientes ascendentes muy violentas.",
        "chat_detailed_cape_desc_strong": "un valor que indica una inestabilidad fuerte, suficiente para tormentas intensas.",
        "chat_detailed_cape_desc_moderate": "un valor moderado. Hay energía para chubascos o alguna tormenta.",
        "chat_detailed_q_orography": "¿Y una montaña de {oro_height} m podría ayudar a superar el CIN restante?", "chat_detailed_a_no_lfc": "En este caso no hay Nivel de Convección Libre (LFC), por lo que la orografía no podrá iniciar convección profunda.",
        "chat_detailed_a_orography_yes": "¡Sí! La orografía de {oro_height} m **ES suficientemente alta** para forzar el aire a superar el LFC (situado a {lfc_agl:.0f} m sobre el suelo). ¡Puede actuar como disparador definitivo!",
        "chat_detailed_a_orography_no": "En este caso, la orografía de {oro_height} m **NO es suficientemente alta** para alcanzar el LFC (situado a {lfc_agl:.0f} m sobre el suelo). Necesitaremos otro mecanismo de disparo (como un frente).",
        "chat_detailed_q_moisture": "¿Tenemos suficiente 'combustible' (humedad) para aprovechar esta energía?", "chat_detailed_a_moisture": "El agua precipitable es de {pwat:.1f} mm en los primeros 4 km. {pwat_analysis}",
        "chat_detailed_q_organization": "Perfecto. Y las tormentas, ¿se organizarán o serán caóticas?", "chat_detailed_a_organization": "Aquí entra en juego la cizalladura del viento (0-6 km), que es de {shear:.1f} m/s. {shear_analysis}",
        "chat_detailed_q_tornado": "¿Eso significa que hay riesgo de tornados?", "chat_detailed_a_tornado": "Para eso miramos la Helicosidad Relativa a la Tormenta (SRH 0-1km), que es de {srh:.1f} m²/s². {srh_analysis}",
        "chat_detailed_summary": "**En resumen:** {verdict}",
        "pwat_analysis_dry": "Es un ambiente relativamente seco. Esto podría limitar la intensidad de la precipitación.", "pwat_analysis_moist": "Hay humedad suficiente para alimentar tormentas y generar lluvia moderada o fuerte.", "pwat_analysis_loaded": "La atmósfera está muy cargada de humedad. Si se desarrollan tormentas, tienen potencial para ser muy eficientes y dejar grandes acumulaciones de lluvia.",
        "shear_analysis_weak": "Es débil. Las tormentas que se formen serán probablemente de ciclo de vida corto y desorganizadas (tormentas unicelulares).", "shear_analysis_moderate": "Es moderado. Esto es suficiente para organizar las tormentas en sistemas multicelulares más duraderos y con más potencial.", "shear_analysis_strong": "Es fuerte. Este es el ingrediente clave que ayuda a las tormentas a rotar y a evolucionar hacia supercélulas, mucho más organizadas y severas.",
        "srh_analysis_high_risk": "Sí, el riesgo es significativo. Valores de SRH por encima de 150 m²/s² con una base de la nube baja (LCL a {lcl_agl:.0f} m sobre el suelo) son un indicador clásico de potencial tornádico.",
        "srh_analysis_moderate_risk": "Indica una rotación considerable a niveles bajos. El riesgo de tornados no es extremo, pero se deben vigilar posibles embudos (funnels) o tubas.",
        "srh_analysis_low_risk": "La rotación a niveles bajos no es especialmente fuerte. El riesgo principal serían los vientos fuertes lineales y el granizo, más que los tornados.",
    },
    'en': {
        # --- General UI ---
        "lang_selector_label": "Language", "app_title": "Sounding Analyzer",
        "welcome_title": "TEMPESTES.CAT PRESENTS:", "welcome_subtitle": "A tool for visualizing and experimenting with atmospheric profiles.",
        "live_mode_title": "⚠️ Today's Alerts", "live_mode_text": "View the latest atmospheric soundings based on model data for today's most active areas.", "live_mode_button": "Access",
        "sandbox_mode_title": "🧪 Laboratory", "sandbox_mode_text": "Learn interactively how severe phenomena form by modifying a sounding step-by-step or experiment freely.", "sandbox_mode_button": "Access Laboratory",
        "manual_mode_title": "✍️ Manual Mode", "manual_mode_text": "Paste the text of a standard sounding and we will analyze it instantly, without needing external files.", "manual_mode_button": "Analyze Your Sounding",
        "coming_soon_title": "🗺️ Coming soon...", "coming_soon_subtitle": "WIND MAPS",
        "back_to_start_button": "⬅️ Back to start", "loading_message": "Loading",
        "analysis_zones_title": "Zone Analysis",
        "follow_todays_zone_button": "Follow today's active zone",
        "todays_changes_title": "Today's Changes",
        "select_region_desc": "Select the region you want to analyze. Each zone represents a different atmospheric profile based on the latest forecasts.",
        "most_notable_zone_title": "Most Notable Zone",
        "most_notable_zone_desc": "The profile with the greatest potential for significant phenomena.",
        "interesting_zone_title": "Interesting Zone",
        "interesting_zone_desc": "A profile that shows some interesting features.",
        "loading_region": "Loading {region}",

"tutorial_lab_title": "🧪 Sounding Laboratory - Tutorial Mode",
"tutorial_title": "Tutorial: {scenario}",
"tutorial_congrats": "🎉 Congratulations, you have completed the tutorial! 🎉",
"tutorial_finish_button": "Finish and See Result",
"tutorial_abandon_button": "Abandon Tutorial",
"tutorial_sleet_step1_title": "Step 1: The Snow Factory",
"tutorial_sleet_step1_instr": "We have loaded a sleet profile. Look at the upper layers (above 700 hPa). The temperatures are below freezing. This is where snowflakes form.",
"tutorial_sleet_step1_button": "Got it, step 1/3 →",
"tutorial_sleet_step1_expl": "This is where the initial snowflakes form. So far, so good.",
"tutorial_sleet_step2_title": "Step 2: The Warm Layer That Melts Everything",
"tutorial_sleet_step2_instr": "Now look at the mid-layer (~850 hPa). The temperature rises above 0°C. This is the problem: the flakes melt and turn into rain.",
"tutorial_sleet_step2_button": "I see, step 2/3 →",
"tutorial_sleet_step2_expl": "When snowflakes fall through this warm layer, they melt and become raindrops.",
"tutorial_sleet_step3_title": "Step 3: Refreezing at the Surface",
"tutorial_sleet_step3_instr": "Finally, near the ground, the temperature drops below freezing again. The raindrops refreeze just before hitting the ground.",
"tutorial_sleet_step3_button": "Understood, step 3/3 →",
"tutorial_sleet_step3_expl": "This is what produces sleet or hazardous freezing rain.",
"tutorial_sleet_step4_title": "Conclusion and Final Challenge",
"tutorial_sleet_step4_instr": "You have analyzed a classic sleet profile! Now you know that an intermediate warm layer is the culprit.",
"tutorial_sleet_step4_button": "Finish Tutorial",
"tutorial_sleet_step4_expl": "Challenge: Now that you're done, click 'Finish'. Use the '❄️ Cool Mid-Low Layer' tool in the sidebar and you'll see how you turn this profile into a perfect snowfall!",
"manual_dialog_error": "Invalid or empty sounding text. Please close and paste the data.",
"manual_dialog_success": "Sounding processed successfully with an elevation of {elevation} m ({pressure:.1f} hPa) and orography of {orography} m.",
"manual_dialog_process_error": "Could not process the text. Make sure the format is correct.",


        "caption_tornado": "A tornado formed under a supercell.",
"caption_funnel": "A funnel cloud descending from the cloud base.",
"caption_wallcloud": "A well-defined wall cloud.",
"caption_shelfcloud": "A spectacular shelf cloud.",
"caption_scud": "A rugged base with cloud fragments (scud).",
"caption_supercell": "An organized supercell.",
"caption_lenticularis": "Lenticular clouds, indicating strong high-altitude winds.",
"caption_castellanus": "Altocumulus Castellanus, indicating mid-level instability.",
"caption_fractus": "Cumulus Fractus, fragmented clouds.",
"caption_cumulonimbus": "A Cumulonimbus, the quintessential thunderstorm cloud.",
"caption_congestus": "Cumulus Congestus, with significant vertical development.",
"caption_humilis": "Cumulus Humilis, fair-weather clouds.",
"caption_cirrus": "High, thin clouds formed of ice crystals.",
"caption_altostratus": "Sky covered by Altostratus, may indicate approaching rain.",
"caption_nimbostratus": "Sky covered by Nimbostratus, associated with continuous rain.",
"caption_stratus": "A low, gray layer of Stratus, similar to fog.",
"caption_sleet": "Precipitation in the form of sleet.",
"caption_snow": "A snowfall covering the landscape.",

        # --- Laboratori ---
        "sandbox_welcome_title": "🧪 Welcome to the Laboratory!", "sandbox_welcome_desc": "Choose how you want to start. You can follow a guided tutorial to learn the key concepts or jump directly into free mode to experiment on your own.",
        "sandbox_tutorial_supercell_title": "🌪️ Tutorial: Supercell", "sandbox_tutorial_supercell_desc": "Learn to create an environment with explosive instability and the necessary shear for the most severe and organized storms.", "sandbox_tutorial_supercell_button": "Start Supercell Tutorial",
        "sandbox_tutorial_sleet_title": "💧 Tutorial: Sleet", "sandbox_tutorial_sleet_desc": "Analyze a sleet situation, identify the culprit warm layer, and learn how to transform the precipitation into snow.", "sandbox_tutorial_sleet_button": "Start Sleet Tutorial",
        "sandbox_freemode_title": "🛠️ Free Mode", "sandbox_freemode_desc": "Jump right into the action. You will have full control over the atmospheric profile from the start to create your own scenarios.", "sandbox_freemode_button": "Go to Free Mode",
        "sandbox_toolbox_header": "Toolbox", "sandbox_thermo_header": "Thermodynamic Modifications",
        "sandbox_layer_sfc": "Surface (>950hPa)", "sandbox_layer_low": "Low Levels (950-800hPa)", "sandbox_layer_lowmid": "Low-Mid Levels (800-600hPa)",
        "sandbox_shear_header": "Wind Shear", "sandbox_shear_low": "Low Levels", "sandbox_shear_mid": "Mid Levels",
        "sandbox_reset_wind": "🚫 Reset Winds", "sandbox_reset_all": "🔄 Reset All to Original Profile",

         "analysis_zones_title": "Zone Analysis",
        "follow_todays_zone_button": "Follow today's active zone",
        "todays_changes_title": "Today's Changes",
        "select_region_desc": "Select the region you want to analyze. Each zone represents a different atmospheric profile based on the latest forecasts.",
        "most_notable_zone_title": "Most Notable Zone",
        "most_notable_zone_desc": "The profile with the greatest potential for significant phenomena.",
        "interesting_zone_title": "Interesting Zone",
        "interesting_zone_desc": "A profile that shows some interesting features.",
        "loading_region": "Loading {region}",
        
        # --- Mode Manual ---
        "manual_mode_page_title": "✍️ Manual Sounding Analyzer", "manual_mode_page_desc": "Paste the full text of your sounding here. The analyzer will process the data and display the results below.",
        "manual_textarea_label": "Enter sounding data:", "manual_textarea_placeholder": "Paste sounding text here...", "manual_analyze_button": "Analyze Sounding",
        "manual_dialog_title": "Custom Initial Analysis", "manual_dialog_header": "Sounding Site Data", "manual_dialog_desc": "Enter the base elevation and orography height for an accurate analysis.",
        "manual_dialog_elevation_label": "**1. Height above sea level (in meters):**", "manual_dialog_orography_label": "**2. Height of surrounding mountains (in meters):**", "manual_dialog_button": "Accept and Generate Full Analysis",
        
        # --- Vista d'Anàlisi ---
        "analyst_assistant_tab": "💬 Analysis Assistant", "parameters_tab": "📊 Parameters", "hodograph_tab": "📈 Hodograph", "orography_tab": "⛰️ Orography",
        "visualization_tab": "☁️ Visualization", "cloud_types_tab": "📋 Cloud Types", "radar_tab": "📡 Radar",
        "skewt_diagram_title": "Skew-T Diagram", "hodograph_title": "Wind Profile Hodograph", "orography_graph_title": "Orographic Triggering Potential",
        "cloud_viz_title": "Cloud Graphic Representations", "cloud_drawing_title": "Cloud Visualization", "cloud_structure_title": "Vertical Structure & Shear",
        "cloud_list_title": "List of Probable Cloud Genera", "cloud_list_desc": "This list is based on the balance between energy (CAPE), inhibition (CIN), humidity (RH), and wind at different layers.",
        "representative_images_title": "Representative Images", "radar_sim_title": "Simulated Radar Reflectivity",

        # --- Paràmetres ---
        "param_sfc_temp": "Surface Temperature", "param_usable_cape": "Usable CAPE", "param_usable_cape_help": "Gross CAPE minus the inhibition (CIN). The actual available energy.",
        "param_srh_01": "SRH 0-1km", "param_cin": "CIN (Inhibition)", "param_lcl": "LCL (AGL)", "param_srh_03": "SRH 0-3km", "param_cape_raw": "CAPE (Gross)",
        "param_lfc": "LFC (AGL)", "param_pwat": "Total PWAT", "param_shear_06": "Shear 0-6km", "param_el": "EL (MSL)", "param_rh_04": "Mean RH 0-4km",

        # --- Avisos Públics ---
        "warn_no_significant_title": "NO SIGNIFICANT WARNINGS", "warn_no_significant_desc": "Current conditions do not present notable meteorological risks.",
        "warn_snow_title": "SNOW WARNING", "warn_snow_desc": "Favorable profile for snowfall. Surface temperature of {temp:.1f}°C and no warm layers present.",
        "warn_sleet_title": "SLEET WARNING", "warn_sleet_desc": "A warm layer aloft ({warm_layer_temp:.1f}°C) and intense cold at the surface ({temp:.1f}°C) favor sleet (ice pellets).",
        "warn_ice_rain_title": "FREEZING RAIN / SLEET WARNING", "warn_ice_rain_desc": "High risk of freezing rain or sleet due to a warm layer aloft and a surface temperature of {temp:.1f}°C. Hazardous road conditions.",
        "warn_high_lfc_title": "NO SIGNIFICANT WARNINGS (FOR NOW)", "warn_high_lfc_desc": "Although potential energy exists, the Level of Free Convection (LFC) is very high ({lfc_agl:.0f} m). This indicates a strong 'cap' that could prevent storm formation.",
        "warn_tornado_title": "TORNADO WARNING", "warn_tornado_desc": "Extreme conditions (Net Energy {cape:.0f}, SRH {srh:.0f}). Very high risk of tornadic supercells.",
        "warn_extreme_severe_title": "EXTREME SEVERE WEATHER WARNING", "warn_extreme_severe_desc": "Potential for destructive supercells (Net Energy {cape:.0f}). Very high risk of large hail (>5cm) and severe winds.",
        "warn_severe_title": "SEVERE WEATHER WARNING", "warn_severe_desc": "Highly unstable and organized atmosphere (Net Energy {cape:.0f}, Shear {shear:.1f}). Risk of large hail and/or very strong wind gusts.",
        "warn_strong_storms_title": "STRONG THUNDERSTORM WARNING", "warn_strong_storms_desc": "High instability (Net Energy {cape:.0f}). Risk of thunderstorms with hail and strong localized winds.",
        "warn_moderate_storms_title": "RISK OF MODERATE THUNDERSTORMS", "warn_moderate_storms_desc": "Potential for organized thunderstorms (Net Energy {cape:.0f}). They may produce heavy showers and small hail.",
        "warn_isolated_storms_title": "RISK OF ISOLATED SHOWERS AND THUNDERSTORMS", "warn_isolated_storms_desc": "Low instability (Net Energy {cape:.0f}). Some short-lived showers or thunderstorms may form.",
        "warn_heavy_rain_title": "HEAVY RAIN WARNING", "warn_heavy_rain_desc": "Very moist atmosphere ({pwat:.1f} mm) and saturated at low levels ({rh:.0f}% RH). Risk of efficient rainfall.",
        
        # --- Text del Xat ---
        "chat_analyst": "Analyst", "chat_user": "User",
        "chat_winter_intro": "We are in a winter weather scenario with a surface temperature of {temp:.1f}°C.",
        "chat_winter_q_snow": "Is there potential for snow?", "chat_winter_a_no_snow": "Not really. The upper layers are not cold enough to form snowflakes efficiently. Any precipitation would be in the form of rain.",
        "chat_winter_q_cold_aloft": "Is it cold enough aloft to snow?", "chat_winter_a_cold_aloft": "Yes, the layers above 700 hPa are a perfect 'snow factory'. Snowflakes will form without any issues.",
        "chat_winter_q_falling": "And what happens when the flakes fall?", "chat_winter_a_warm_layer": "Here's the key: as they fall, they encounter a warm layer of about **{temp:.1f}°C**. This will melt the flakes and turn them into raindrops.",
        "chat_winter_q_surface": "So, what will reach the ground?", "chat_winter_a_sleet": "Since the surface is at 0°C or below, these raindrops will refreeze just before hitting the ground. The result will be **sleet** or hazardous **freezing rain**.",
        "chat_winter_a_rain": "Despite the snow aloft, the positive surface temperature means the final precipitation will be **rain**.",
        "chat_winter_a_cold_column": "The atmospheric column remains below 0°C throughout its entire path. The snowflakes will not melt.", "chat_winter_q_then": "So then...", "chat_winter_a_snow": "Exactly. We will have a **snowfall** at the surface!",
        "chat_detailed_intro": "Hello! Let's analyze this atmospheric profile, which starts at an elevation of {height:.0f} meters.",
        "chat_detailed_energy_balance": "First, let's assess the energy balance. We have a CAPE (potential energy) of **{cape:.0f} J/kg**.",
        "chat_detailed_q_cin": "And what about the 'cap' (CIN)? Can it stop it?", "chat_detailed_a_cin": "Great question. The CIN (inhibition) is **{cin:.0f} J/kg**. This value acts as a brake. If we subtract this brake from the potential energy, we're left with a **usable CAPE of {usable_cape:.0f} J/kg**.",
        "chat_detailed_stable": "Since the net energy is very low, the 'cap' is too strong. It's **very unlikely** that significant storms will form from the surface. The atmosphere is effectively stable.",
        "chat_detailed_go_ahead": "This is the energy truly available to form storms. Now that we have the 'green light', we can analyze the other ingredients.",
        "chat_detailed_cape_desc_extreme": "an extremely high value. This means there's explosive potential for very violent updrafts.",
        "chat_detailed_cape_desc_strong": "a value indicating strong instability, sufficient for intense storms.",
        "chat_detailed_cape_desc_moderate": "a moderate value. There's energy for showers or some thunderstorms.",
        "chat_detailed_q_orography": "And could a mountain of {oro_height} m help overcome the remaining CIN?", "chat_detailed_a_no_lfc": "In this case, there is no accessible Level of Free Convection (LFC), so orography cannot initiate deep convection.",
        "chat_detailed_a_orography_yes": "Yes! The orography of {oro_height} m **IS high enough** to force the air above the LFC (located at {lfc_agl:.0f} m AGL). It can act as the definitive trigger!",
        "chat_detailed_a_orography_no": "In this case, the orography of {oro_height} m **IS NOT high enough** to reach the LFC (located at {lfc_agl:.0f} m AGL). We will need another triggering mechanism (like a front).",
        "chat_detailed_q_moisture": "Do we have enough 'fuel' (moisture) to take advantage of this energy?", "chat_detailed_a_moisture": "Precipitable water is {pwat:.1f} mm in the first 4 km. {pwat_analysis}",
        "chat_detailed_q_organization": "Perfect. And the storms, will they be organized or chaotic?", "chat_detailed_a_organization": "This is where wind shear (0-6 km) comes into play, which is {shear:.1f} m/s. {shear_analysis}",
        "chat_detailed_q_tornado": "Does that mean there's a risk of tornadoes?", "chat_detailed_a_tornado": "For that, we look at the Storm Relative Helicity (SRH 0-1km), which is {srh:.1f} m²/s². {srh_analysis}",
        "chat_detailed_summary": "**In summary:** {verdict}",
        "pwat_analysis_dry": "It's a relatively dry environment. This could limit precipitation intensity.", "pwat_analysis_moist": "There is enough moisture to fuel storms and generate moderate to heavy rain.", "pwat_analysis_loaded": "The atmosphere is heavily loaded with moisture. If storms develop, they have the potential to be very efficient and produce large rainfall accumulations.",
        "shear_analysis_weak": "It's weak. Any storms that form will likely be short-lived and disorganized (single-cell storms).", "shear_analysis_moderate": "It's moderate. This is sufficient to organize storms into more durable multicellular systems with more potential.", "shear_analysis_strong": "It's strong. This is the key ingredient that helps storms rotate and evolve into supercells, which are much more organized and severe.",
        "srh_analysis_high_risk": "Yes, the risk is significant. SRH values above 150 m²/s² with a low cloud base (LCL at {lcl_agl:.0f} m AGL) are a classic indicator of tornadic potential.",
        "srh_analysis_moderate_risk": "Indicates considerable low-level rotation. The tornado risk is not extreme, but funnels or tuba clouds should be watched for.",
        "srh_analysis_low_risk": "Low-level rotation is not particularly strong. The main risk would be straight-line winds and hail, rather than tornadoes.",


    }
}
def get_text(key, **kwargs):
    
    lang = st.session_state.get('language', 'ca')
    # Agafa la traducció de 'ca' si la clau no existeix en l'idioma seleccionat
    template = LANGUAGES.get(lang, {}).get(key) or LANGUAGES.get('ca', {}).get(key, f"NO_TEXT_FOR_{key}")
    return template.format(**kwargs) if kwargs else template

def set_language(lang_code):
    st.session_state.language = lang_code



def set_main_background():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(0deg, rgba(6,14,42,1) 0%, rgba(25,37,81,1) 100%);
        background-size: cover; background-position: center center;
        background-repeat: no-repeat; background-attachment: local;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    [data-testid="stToolbar"] {{ right: 2rem; }}
    .welcome-title {{
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FFD700, #FF8C00, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
    }}
    @keyframes glow {{
        from {{
            text-shadow: 0 0 10px #FFD700, 0 0 20px #FF8C00;
        }}
        to {{
            text-shadow: 0 0 20px #FFD700, 0 0 30px #FF4500, 0 0 40px #FF4500;
        }}
    }}
    .welcome-subtitle {{
        font-size: 1.5rem; color: #E0E0E0; text-align: center; margin-bottom: 40px;
    }}
    .mode-card {{
        background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 25px; border-radius: 15px; backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); color: white; height: 100%;
    }}
    .mode-card h3 {{ color: #FFFFFF; font-weight: bold; }}
    .mode-card p {{ color: #D0D0D0; }}
    .coming-soon {{
        text-align: center;
        margin-top: 60px;
        color: #a0a0a0;
    }}
    .coming-soon h2 {{
        font-size: 2.5rem;
        color: #ffffff;
        font-weight: bold;
        letter-spacing: 2px;
    }}
    .coming-soon p {{
        font-size: 1.5rem;
    }}
    .footer {{
        text-align: center;
        margin-top: 40px;
        padding-bottom: 20px;
        color: #6c757d;
        font-size: 0.9rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def create_city_mountain_scape():
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    star_x, star_y = np.random.uniform(0, 100, 200), np.random.uniform(15, 60, 200)
    star_s, star_alpha = np.random.uniform(0.5, 2.5, 200), np.random.uniform(0.5, 1, 200)
    ax.scatter(star_x, star_y, s=star_s, c='white', alpha=star_alpha, edgecolors='none')
    mountain_poly = Polygon([(55, 0), (68, 38), (75, 32), (85, 45), (95, 28), (100, 32), (100, 0)], facecolor='#12182c', edgecolor=None, zorder=5)
    ax.add_patch(mountain_poly)
    city_patches, light_patches = [], []
    for x_base in np.arange(0, 70, 0.5):
        height_factor = 1 - abs(x_base - 35) / 35
        building_height = (random.uniform(2, 12) * (1 + height_factor * 2))
        building_width = random.uniform(0.8, 3)
        color_val = random.uniform(0.05, 0.1)
        building = Rectangle((x_base, 0), building_width, building_height, facecolor=(color_val, color_val, color_val), edgecolor=None, zorder=10)
        city_patches.append(building)
        if random.random() < 0.08:
            light_x, light_y = x_base + random.uniform(0, building_width), random.uniform(1, building_height * 0.5)
            light = Circle((light_x, light_y), radius=0.15, color='#fde9a0', alpha=0.9)
            light_patches.append(light)
    ax.add_collection(PatchCollection(city_patches, match_original=True))
    ax.add_collection(PatchCollection(light_patches, match_original=True, zorder=11))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def styled_metric(label, value, unit, help_text=""):
    """
    Mostra una mètrica amb estils de color i emojis segons llindars predefinits.
    """
    thresholds = {
        "CAPE Utilitzable": (1000, 2500),
        "CIN (Fre)": (-25, -100),
        "Shear 0-6km": (15, 25),
        "SRH 0-1km": (100, 250),
        "SRH 0-3km": (150, 300)
    }
    
    color, emoji = "inherit", ""
    
    numeric_value = np.nan
    if value is not None and not isinstance(value, str):
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            pass

    if not np.isnan(numeric_value):
        if label in thresholds:
            warn_thresh, danger_thresh = thresholds[label]
            if label == "CIN (Fre)":
                if numeric_value < danger_thresh: color, emoji = "#dc3545", "⚠️"
                elif numeric_value < warn_thresh: color = "#ffc107"
            else:
                if numeric_value >= danger_thresh: color, emoji = "#dc3545", "⚠️"
                elif numeric_value >= warn_thresh: color = "#28a745"
        elif label == "Temperatura Superficial":
            if numeric_value > 35: color, emoji = "#dc3545", "🔥"
            elif numeric_value > 25: color = "#ffc107"
            elif numeric_value < -5: color, emoji = "#9932CC", "🥶"
            elif numeric_value < 5: color = "#1E90FF"

    if isinstance(value, float):
        formatted_value = f"{value:.1f}" if not np.isnan(value) else "N/A"
    else:
        formatted_value = f"{value}" if value is not None else "N/A"

    html = f"""
    <div title="{help_text}" style="font-family: sans-serif; margin-bottom: 10px;">
        <p style="font-size: 0.9rem; color: #808495; margin-bottom: -5px;">{label}</p>
        <p style="font-size: 1.6rem; font-weight: bold; color: {color}; margin-top: 0px;">
            {formatted_value} <span style="font-size: 1.1rem; font-weight: normal;">{unit}</span> {emoji}
        </p>
    </div>
    """
    return html


# =============================================================================
# === 1. FUNCIONS DE CÀRREGA I PROCESSAMENT DE DADES =========================
# =============================================================================

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
    time_keywords = ['observació', 'hora', 'time', 'run', 'z', 'date']
    
    for line in block_lines:
        line_strip = line.strip()
        
        if 'locale' in line_strip.lower():
            continue
            
        is_metadata_line = any(keyword in line_strip.lower() for keyword in time_keywords) and not (line_strip and line_strip[0].isdigit())

        if is_metadata_line:
            time_lines.append(line_strip)
            continue
            
        if not line_strip or line_strip.startswith('#') or 'Pression' in line_strip:
            continue
            
        try:
            line_to_process = re.sub(r'\([^)]*\)', '', line_strip).strip()
            parts = re.split(r'\s{2,}|[\t]', line_to_process)
            
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
        except Exception as e:
            st.warning(f"Advertència: Error processant línia '{line_strip}'. Error: {e}")
            continue
            
    if not p_list or len(p_list) < 2: return None
    
    observation_time = "\n".join(time_lines) if time_lines else "Hora no disponible"
    sorted_indices = np.argsort(p_list)[::-1]
    
    return {'p_levels': np.array(p_list)[sorted_indices] * units.hPa, 
            't_initial': np.array(t_list)[sorted_indices] * units.degC, 
            'td_initial': np.array(td_list)[sorted_indices] * units.degC, 
            'wind_speed_kmh': np.array(wspd_list)[sorted_indices] * units.kph, 
            'wind_dir_deg': np.array(wdir_list)[sorted_indices] * units.degrees, 
            'observation_time': observation_time}

def parse_all_soundings(filepath):
    all_soundings_data = []
    current_sounding_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        st.error(f"Error: No s'ha trobat el fitxer '{filepath}'. Assegura't que existeix al mateix directori.")
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
# === 2. FUNCIONS DE CÀLCUL I ANÀLISI =====================================
# =========================================================================

def calculate_thermo_parameters(p_levels, t_profile, td_profile):
    with integrator_lock:
        try:
            p, t, td = p_levels, t_profile, td_profile
            valid_indices = ~np.isnan(p.magnitude) & ~np.isnan(t.magnitude) & ~np.isnan(td.magnitude)
            if np.sum(valid_indices) < 2: raise ValueError("No hi ha prou dades.")
            p, t, td = p[valid_indices], t[valid_indices], td[valid_indices]
            p_sfc, t_sfc, td_sfc = p[0], t[0], td[0]
            
            parcel_prof = mpcalc.parcel_profile(p, t_sfc, td_sfc).to('degC')
            
            cape, cin = mpcalc.cape_cin(p, t, td, parcel_prof)
            lcl_p, _ = mpcalc.lcl(p_sfc, t_sfc, td_sfc)
            lfc_p, _ = mpcalc.lfc(p, t, td, parcel_prof)
            el_p, _ = mpcalc.el(p, t, td, parcel_prof)
            
            lcl_h = mpcalc.pressure_to_height_std(lcl_p).to('m').m if lcl_p is not None else 0
            lfc_h = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p is not None else np.inf
            el_h = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p is not None else lfc_h
            
            try:
                t_interp = interp1d(p.m, t.m, bounds_error=False, fill_value=np.nan)
                p_range = np.arange(p.m.max(), p.m.min(), -0.1)
                t_range = t_interp(p_range)
                fz_indices = np.where(t_range < 0)[0]
                fz_lvl = p_range[fz_indices[0]] * units.hPa if len(fz_indices) > 0 else np.nan * units.hPa
                fz_h = mpcalc.pressure_to_height_std(fz_lvl).to('m').m if not np.isnan(fz_lvl.m) else 0
            except Exception:
                fz_lvl = np.nan * units.hPa
                fz_h = 0
            
            if el_p is None and cape.magnitude > 0: el_p = p[-1]

            return cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h, fz_lvl
            
        except Exception as e:
            return (units.Quantity(0, 'J/kg'), units.Quantity(0, 'J/kg'), None, 0, None, np.inf, None, 0, 0, None)

def calculate_storm_parameters(p_levels, wind_speed, wind_dir):
    try:
        p, ws, wd = p_levels, wind_speed.to('m/s'), wind_dir
        u, v = mpcalc.wind_components(ws, wd)
        
        heights_raw = mpcalc.pressure_to_height_std(p).to('meter')
        valid_mask = ~np.isnan(heights_raw.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
        if np.sum(valid_mask) < 2: return 0.0, 0.0, 0.0, 0.0
        
        p_c, u_c, v_c, h_c = p[valid_mask], u[valid_mask], v[valid_mask], heights_raw[valid_mask]
        _, unique_indices = np.unique(h_c.m, return_index=True)
        if len(unique_indices) < 2: return 0.0, 0.0, 0.0, 0.0
        
        p_u, u_u, v_u, h_u = p_c[unique_indices], u_c[unique_indices], v_c[unique_indices], h_c[unique_indices]
        
        h_min, h_max = h_u.m.min(), min(h_u.m.max(), 12000)
        if h_max <= h_min: return 0.0, 0.0, 0.0, 0.0
        
        h_interp = np.arange(h_min, h_max, 50) * units.meter
        u_i = np.interp(h_interp.m, h_u.m, u_u.m) * units('m/s')
        v_i = np.interp(h_interp.m, h_u.m, v_u.m) * units('m/s')
        
        p_interp = mpcalc.height_to_pressure_std(h_interp)

        u_6, v_6 = mpcalc.bulk_shear(p_interp, u_i, v_i, height=h_interp, depth=6000 * units.meter)
        s_0_6 = mpcalc.wind_speed(u_6, v_6).m
        u_1, v_1 = mpcalc.bulk_shear(p_interp, u_i, v_i, height=h_interp, depth=1000 * units.meter)
        s_0_1 = mpcalc.wind_speed(u_1, v_1).m
        
        with integrator_lock:
            srh_0_3 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=3000 * units.meter)[0].m
            srh_0_1 = mpcalc.storm_relative_helicity(h_interp, u_i, v_i, depth=1000 * units.meter)[0].m
        
        return s_0_6, s_0_1, srh_0_1, srh_0_3
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

# --- NOVES FUNCIONS D'ANÀLISI DE XAT ---
def get_pwat_analysis(pwat_val):
    if pwat_val < 15: return "És un ambient relativament sec. Això podria limitar la intensitat de la precipitació."
    if pwat_val < 30: return "Hi ha humitat suficient per alimentar tempestes i generar pluja moderada o forta."
    return "L'atmosfera està molt carregada d'humitat. Si es desenvolupen tempestes, tenen potencial per a ser molt eficients i deixar grans acumulacions de pluja."

def get_shear_analysis(shear_val):
    if shear_val < 10: return "És feble. Les tempestes que es formin seran probablement de cicle de vida curt i desorganitzades (tempestes unicel·lulars)."
    if shear_val < 18: return "És moderat. Això és suficient per organitzar les tempestes en sistemes multicel·lulars més duradors i amb més potencial."
    return "És fort. Aquest és l'ingredient clau que ajuda les tempestes a rotar i a evolucionar cap a supercèl·lules, molt més organitzades i severes."

def get_srh_analysis(srh_val, lcl_agl):
    if srh_val > 150 and lcl_agl < 1200: return f"Sí, el risc és significatiu. Valors d'SRH per sobre de 150 m²/s² amb una base del núvol baixa (LCL a {lcl_agl:.0f} m sobre el terra) són un indicador clàssic de potencial tornàdic."
    if srh_val > 100: return "Indica una rotació considerable a nivells baixos. El risc de tornados no és extrem, però s'han de vigilar possibles embuts (funnels) o tubes."
    return "La rotació a nivells baixos no és especialment forta. El risc principal serien els vents forts lineals i la calamarsa, més que no pas els tornados."

def get_verdict(cloud_type):
    verdicts = {
        "Supercèl·lula (Tornàdica)": "Tenim tots els ingredients per a supercèl·lules amb un alt potencial de generar tornados.",
        "Supercèl·lula (Tuba/Funnel)": "Les condicions són molt favorables per a supercèl·lules amb rotació que podria generar tubes o embuts.",
        "Supercèl·lula (Mur de núvols)": "Perfil clàssic de supercèl·lula. Hi ha un alt risc de calamarsa gran i vents severs, amb la possible formació de murs de núvols.",
        "Supercèl·lula": "Tenim una combinació perillosa d'alta inestabilitat i fort cisallament. El risc de temps sever organitzat (calamarsa, ventades) és molt alt.",
        "Cumulonimbus (Shelf Cloud)": "L'ingredient dominant és l'energia extrema amb un cisallament més lineal. El perill principal són els 'reventones' o 'downbursts' (vents lineals destructius).",
        "Cumulonimbus (Multicèl·lula)": "Hi ha prou energia i organització per a sistemes de tempestes multicel·lulars que poden deixar pluja intensa i calamarsa.",
        "Cumulus congestus": "Tenim energia per a un bon desenvolupament vertical, donant lloc a núvols de gran mida que poden deixar ruixats forts i alguna tempesta local."
    }
    return verdicts.get(cloud_type, "L'anàlisi suggereix que el tipus de núvol predominant serà " + cloud_type.lower() + ".")

def generate_winter_analysis(p, t, td):
    """Genera una anàlisi conversacional específica per a temps hivernal."""
    chat_log = []
    precipitation_type = 'rain'
    t_c = t.to('degC').m
    p_hpa = p.to('hPa').m
    
    upper_mask = p_hpa <= 700
    is_cold_aloft = np.all(t_c[upper_mask] < -2) if np.any(upper_mask) else False
    
    # CORREGIT: Utilitza les claus de traducció
    chat_log.append(("chat_analyst", get_text("chat_winter_intro", temp=t_c[0])))
    
    if not is_cold_aloft:
        chat_log.append(("chat_user", get_text("chat_winter_q_snow")))
        chat_log.append(("chat_analyst", get_text("chat_winter_a_no_snow")))
        return chat_log, 'rain'
        
    chat_log.append(("chat_user", get_text("chat_winter_q_cold_aloft")))
    chat_log.append(("chat_analyst", get_text("chat_winter_a_cold_aloft")))

    mid_layer_mask = (p_hpa < 900) & (p_hpa > 650)
    warm_layer_temp = np.max(t_c[mid_layer_mask]) if np.any(mid_layer_mask) else -99
    
    chat_log.append(("chat_user", get_text("chat_winter_q_falling")))
    if warm_layer_temp > 0.5:
        chat_log.append(("chat_analyst", get_text("chat_winter_a_warm_layer", temp=warm_layer_temp)))
        chat_log.append(("chat_user", get_text("chat_winter_q_surface")))
        if t_c[0] <= 0.0:
            chat_log.append(("chat_analyst", get_text("chat_winter_a_sleet")))
            precipitation_type = 'sleet'
        else:
            chat_log.append(("chat_analyst", get_text("chat_winter_a_rain")))
            precipitation_type = 'rain'
    else:
        chat_log.append(("chat_analyst", get_text("chat_winter_a_cold_column")))
        chat_log.append(("chat_user", get_text("chat_winter_q_then")))
        chat_log.append(("chat_analyst", get_text("chat_winter_a_snow")))
        precipitation_type = 'snow'

    return chat_log, precipitation_type

def generate_detailed_analysis(p_levels, t_profile, td_profile, wind_speed, wind_dir, cloud_type, base_km, top_km, pwat_0_4, surface_height, orography_height, usable_cape):
    cape, cin, _, lcl_h, _, lfc_h, _, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    shear_0_6, _, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    precipitation_type = None
    
    # CORREGIT: Utilitza les claus de traducció
    chat_log = [( "chat_analyst", get_text("chat_detailed_intro", height=surface_height))]
    chat_log.append(("chat_analyst", get_text("chat_detailed_energy_balance", cape=cape.m)))
    chat_log.append(("chat_user", get_text("chat_detailed_q_cin")))
    chat_log.append(("chat_analyst", get_text("chat_detailed_a_cin", cin=cin.m, usable_cape=usable_cape.m)))

    if usable_cape.m < 100:
        chat_log.append(("chat_analyst", get_text("chat_detailed_stable")))
        return chat_log, None

    chat_log.append(("chat_analyst", get_text("chat_detailed_go_ahead")))
    
    if usable_cape.m > 2500: cape_desc = get_text("chat_detailed_cape_desc_extreme")
    elif usable_cape.m > 1000: cape_desc = get_text("chat_detailed_cape_desc_strong")
    else: cape_desc = get_text("chat_detailed_cape_desc_moderate")
    chat_log.append(("chat_analyst", cape_desc))

    if cin.m < -25 and orography_height > 0:
        lfc_agl = lfc_h - surface_height
        chat_log.append(("chat_user", get_text("chat_detailed_q_orography", oro_height=orography_height)))
        if lfc_h == np.inf:
            chat_log.append(("chat_analyst", get_text("chat_detailed_a_no_lfc")))
        elif orography_height >= lfc_agl:
            chat_log.append(("chat_analyst", get_text("chat_detailed_a_orography_yes", oro_height=orography_height, lfc_agl=lfc_agl)))
        else:
            chat_log.append(("chat_analyst", get_text("chat_detailed_a_orography_no", oro_height=orography_height, lfc_agl=lfc_agl)))

    chat_log.extend([("chat_user", get_text("chat_detailed_q_moisture")), 
                     ("chat_analyst", get_text("chat_detailed_a_moisture", pwat=pwat_0_4.m, pwat_analysis=get_pwat_analysis(pwat_0_4.m)))])
    
    chat_log.extend([("chat_user", get_text("chat_detailed_q_organization")), 
                     ("chat_analyst", get_text("chat_detailed_a_organization", shear=shear_0_6, shear_analysis=get_shear_analysis(shear_0_6)))])
    
    if shear_0_6 > 18:
        lcl_agl = lcl_h - surface_height
        chat_log.extend([("chat_user", get_text("chat_detailed_q_tornado")), 
                         ("chat_analyst", get_text("chat_detailed_a_tornado", srh=srh_0_1, srh_analysis=get_srh_analysis(srh_0_1, lcl_agl)))])

    chat_log.append(("chat_analyst", get_text("chat_detailed_summary", verdict=get_verdict(cloud_type))))
    
    if "Tornàdica" in cloud_type or "Tuba" in cloud_type or "Mur" in cloud_type: precipitation_type = 'hail'
    elif usable_cape.m > 100: precipitation_type = 'rain'
    elif "Nimbostratus" in cloud_type: precipitation_type = 'rain'

    return chat_log, precipitation_type

def generate_dynamic_analysis(p, t, td, ws, wd, cloud_type, surface_height):
    """Genera anàlisi conversacional per al mode laboratori."""
    cape, cin, _, lcl_h, _, lfc_h, _, _, _, _ = calculate_thermo_parameters(p, t, td)
    usable_cape_val = max(0, cape.m - abs(cin.m))
    shear_0_6, _, _, _ = calculate_storm_parameters(p, ws, wd)
    chat_log = [("Analista", f"Molt bé, analitzem el perfil des d'una elevació de {surface_height:.0f} m.")]

    chat_log.append(("Usuari", "Quin és el balanç energètic actual?"))
    chat_log.append(("Analista", f"Tenim un CAPE brut de {cape.m:.0f} J/kg i un CIN de {cin.m:.0f} J/kg. Això ens dóna un **CAPE utilitzable de {usable_cape_val:.0f} J/kg**."))

    if usable_cape_val < 50:
        chat_log.append(("Analista", "Amb aquesta energia neta, l'atmosfera és molt estable. No hi ha potencial per a tempestes."))
    else:
        cloud_mention = f"Això és un escenari típic per a la formació de {cloud_type}." if cloud_type else ""
        if "Cel Serè" in cloud_type:
             cloud_mention = "Encara que hi ha energia, la tapadora és tan forta que probablement no veuríem cap núvol significatiu."
        chat_log.append(("Analista", f"L'energia neta de {usable_cape_val:.0f} J/kg és suficient per desenvolupar convecció. {cloud_mention}"))
        
        if usable_cape_val > 500: # Només parlem del cisallament si hi ha energia suficient
            chat_log.append(("Usuari", "He modificat el vent. Com afecta?"))
            shear_analysis = get_shear_analysis(shear_0_6)
            chat_log.append(("Analista", f"El cisallament (0-6 km) és de {shear_0_6:.1f} m/s. {shear_analysis}"))
            
    return chat_log, None

def generate_tutorial_analysis(scenario, step):
    """Genera l'anàlisi del xat per a un pas específic d'un tutorial."""
    chat_log = []
    if scenario == 'aiguaneu':
        if step == 0: chat_log.extend([("Analista", "Benvingut! Anem a analitzar un perfil clàssic d'aiguaneu."), ("Usuari", "Perfecte. Què és el primer que he de mirar?"), ("Analista", "Observa la 'fàbrica de neu' a les capes altes. Per sobre de 700 hPa fa prou fred per formar flocs de neu.")])
        elif step == 1: chat_log.extend([("Analista", "Molt bé. Ara ve la part clau. Fixa't en la capa al voltant de 850 hPa. La temperatura puja per sobre dels 0°C."), ("Usuari", "Això és la 'capa càlida', oi? Què provoca?"), ("Analista", "Exacte. Aquesta capa actua com un 'bufador' i fon els flocs, convertint-los en gotes de pluja.")])
        elif step == 2: chat_log.extend([("Analista", "Ja gairebé ho tenim. Ara tenim gotes de pluja caient cap a la superfície. Però mira la temperatura a prop del terra..."), ("Usuari", "Torna a estar per sota de 0°C!"), ("Analista", "Precisament! Aquestes gotes es tornen a congelar just abans d'arribar a terra. Això és l'aiguaneu (sleet).")])
        elif step == 3: chat_log.extend([("Analista", "Has analitzat el perfil a la perfecció."), ("Usuari", "Entès. Llavors, com ho podria convertir en una nevada?"), ("Analista", "Aquest és el repte! Ara, quan finalitzis el tutorial, ves al Mode Lliure i utilitza l'eina '❄️ Refredar Capa Mitjana-Baixa'. Veuràs com el perfil es converteix en una nevada perfecta!")])
    elif scenario == 'supercel':
        if step == 0: chat_log.append(("Analista", "Comencem el tutorial de supercèl·lula. El primer pas és sempre crear energia. Necessitem un dia càlid d'estiu. Escalfem la superfície!"))
        elif step == 1: chat_log.append(("Analista", "Perfecte! Ara afegim el combustible. Injectarem una gran quantitat d'humitat per disparar el CAPE a valors extrems."))
        elif step == 2: chat_log.append(("Analista", "Fantàstic! Has afegit cisallament a nivells mitjans. Aquest és l'ingredient secret que fa que les tempestes rotin."))
        elif step == 3: chat_log.append(("Analista", "Hem potenciat el Jet Stream a les capes altes. Aquest vent fort ajuda a 'ventilar' la tempesta, fent-la més duradora i organitzada."))
        elif step == 4: chat_log.append(("Analista", "Missió complerta! Has creat un perfil amb molta energia (CAPE), humitat i un fort cisallament organitzat. Fixa't com han augmentat els paràmetres de cisallament (Shear) i helicitat (SRH)."))
    return chat_log, None
    
def generate_public_warning(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    """
    Versió corregida que retorna CLAU de traducció, dades per formatar i color.
    """
    t_profile_c = t_profile.to('degC').m
    p_levels_hpa = p_levels.to('hPa').m

    # --- LÒGICA DE PRECIPITACIÓ HIVERNAL ---
    if t_profile_c[0] < 4.5:
        low_mid_mask = (p_levels_hpa <= 1000) & (p_levels_hpa >= 700)
        warm_layer_exists = False
        temps_in_layer_max = -999
        if np.any(low_mid_mask):
            temps_in_layer = t_profile_c[low_mid_mask]
            if np.any(temps_in_layer > 0.5):
                warm_layer_exists = True
                temps_in_layer_max = np.max(temps_in_layer)
        sfc_mask = (p_levels_hpa <= 1000) & (p_levels_hpa >= 900)
        max_sfc_layer_temp = np.max(t_profile_c[sfc_mask]) if np.any(sfc_mask) else t_profile_c[0]
        if max_sfc_layer_temp < 1.0 and not warm_layer_exists:
            return "warn_snow_title", {'temp': t_profile_c[0]}, "dodgerblue"
        if warm_layer_exists and t_profile_c[0] <= 0.5:
            if t_profile_c[0] < -0.5: 
                return "warn_sleet_title", {'warm_layer_temp': temps_in_layer_max, 'temp': t_profile_c[0]}, "mediumorchid"
            else:
                return "warn_ice_rain_title", {'temp': t_profile_c[0]}, "crimson"

    # --- LÒGICA DE TEMPS SEVER (CONVECTIU) ---
    cape, cin, _, lcl_h, _, lfc_h, _, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    usable_cape = max(0, cape.m - abs(cin.m))
    surface_height = mpcalc.pressure_to_height_std(p_levels[0]).to('m').m
    shear_0_6, _, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    lfc_agl_m = (lfc_h - surface_height) if lfc_h != np.inf else np.inf

    if lfc_agl_m > 2800 and usable_cape > 100:
        return "warn_high_lfc_title", {'lfc_agl': lfc_agl_m}, "green"
    if usable_cape > 2000 and srh_0_1 > 150 and (lcl_h - surface_height) < 1000 and shear_0_6 > 20:
        return "warn_tornado_title", {'cape': usable_cape, 'srh': srh_0_1}, "darkred"
    if usable_cape > 2500 and srh_0_3 > 300 and shear_0_6 > 20:
        return "warn_extreme_severe_title", {'cape': usable_cape}, "purple"
    if usable_cape > 1500 and shear_0_6 > 18:
        return "warn_severe_title", {'cape': usable_cape, 'shear': shear_0_6}, "saddlebrown"
    if usable_cape > 1000:
        return "warn_strong_storms_title", {'cape': usable_cape}, "darkorange"
    if usable_cape > 500:
        return "warn_moderate_storms_title", {'cape': usable_cape}, "gold"
    if usable_cape > 100:
        return "warn_isolated_storms_title", {'cape': usable_cape}, "cornflowerblue"

    # --- LÒGICA DE PLUGES INTENSES ---
    try:
        pwat_total = mpcalc.precipitable_water(p_levels, td_profile).to('mm')
        low_level_mask = p_levels.m >= 850
        if np.any(low_level_mask):
            rh_low = mpcalc.relative_humidity_from_dewpoint(t_profile[low_level_mask], td_profile[low_level_mask])
            low_level_rh_mean = np.mean(rh_low).m
            if pwat_total.m > 35 and low_level_rh_mean > 0.80:
                return "warn_heavy_rain_title", {'pwat': pwat_total.m, 'rh': low_level_rh_mean * 100}, "darkblue"
    except Exception:
        pass

    return "warn_no_significant_title", {}, "green"

def count_parameter_anomalies(usable_cape, cin, shear_0_6, srh_0_1, srh_0_3, t_sfc):
    """Compta el nombre de paràmetres que superen els llindars d'alerta."""
    count = 0
    if usable_cape >= 1000: count += 1
    if cin < -25: count += 1
    if shear_0_6 >= 15: count += 1
    if srh_0_1 >= 100: count += 1
    if srh_0_3 >= 150: count += 1
    if t_sfc > 25 or t_sfc < 5: count +=1
    return count

def determine_potential_cloud_types(p, t, td, cape, cin, wind_speed, wind_dir):
    """
    Determina els gèneres de núvols probables amb lògica millorada basada en HR, CAPE i vent.
    """
    potential_clouds = set()
    
    try:
        if len(p) < 2: return ["Dades insuficients"]
        usable_cape_val = max(0, cape.m - abs(cin.m))
        surface_height = mpcalc.pressure_to_height_std(p[0]).m
        heights_agl = mpcalc.pressure_to_height_std(p).to('m').m - surface_height
        rh = mpcalc.relative_humidity_from_dewpoint(t, td) * 100
        
        low_mask = (heights_agl >= 0) & (heights_agl < 2000)
        low_to_3km_mask = (heights_agl >= 0) & (heights_agl < 3000)
        mid_mask = (heights_agl >= 2000) & (heights_agl < 7000)
        high_mask = (heights_agl >= 7000)
        
        if usable_cape_val > 1000: potential_clouds.add("Cumulonimbus (Cb)")
        elif usable_cape_val > 500: potential_clouds.add("Cumulus congestus (Cu con)")
        elif usable_cape_val > 50: potential_clouds.add("Cumulus humilis (Cu)")

        if usable_cape_val < 250:
            if np.any(low_to_3km_mask) and np.mean(rh[low_to_3km_mask]) > 90:
                potential_clouds.add("Nimbostratus (Ns)")
            else:
                if np.any(low_mask) and np.mean(rh[low_mask]) > 60: potential_clouds.add("Stratus (St) / Stratocumulus (Sc)")
                if np.any(mid_mask) and np.mean(rh[mid_mask]) > 60: potential_clouds.add("Altostratus (As) / Altocumulus (Ac)")

        if np.any(high_mask) and np.mean(rh[high_mask]) > 60:
            potential_clouds.add("Cirrus (Ci) / Cirrostratus (Cs)")
            
        try:
            p_hpa = p.to('hPa').m
            ws_kts = wind_speed.to('knots').m
            lenticular_mask = (p_hpa <= 700) & (p_hpa >= 600)
            if np.any(lenticular_mask):
                if np.mean(ws_kts[lenticular_mask]) >= 25:
                     potential_clouds.add("Lenticularis (Ac len)")
        except Exception: pass 

    except Exception as e: return [f"Error detectant núvols: {e}"]

    if "Cumulonimbus (Cb)" in potential_clouds:
        potential_clouds.discard("Cumulus congestus (Cu con)"); potential_clouds.discard("Cumulus humilis (Cu)"); potential_clouds.discard("Altocumulus (Ac)")
    if "Cumulus congestus (Cu con)" in potential_clouds:
        potential_clouds.discard("Cumulus humilis (Cu)")
    if "Nimbostratus (Ns)" in potential_clouds:
        potential_clouds.discard("Stratus (St) / Stratocumulus (Sc)"); potential_clouds.discard("Altostratus (As) / Altocumulus (Ac)")

    return sorted(list(potential_clouds)) if potential_clouds else ["Cel Serè"]

def get_cloud_type_for_chat(p, t, td, ws, wd, cape, cin, lcl_h, lfc_h, el_p):
    """
    Funció específica per determinar el tipus de núvol més rellevant per al xat.
    """
    base_clouds = determine_potential_cloud_types(p, t, td, cape, cin, ws, wd)
    surface_height = mpcalc.pressure_to_height_std(p[0]).to('m').m
    lcl_agl = lcl_h - surface_height
    usable_cape_val = max(0, cape.m - abs(cin.m))

    if any("Cumulonimbus" in s for s in base_clouds):
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
        
        if usable_cape_val > 1500 and srh_0_1 > 150 and lcl_agl < 1000 and shear_0_6 > 18: return "Supercèl·lula (Tornàdica)"
        if usable_cape_val > 1500 and srh_0_1 > 120 and lcl_agl < 1200 and shear_0_6 > 18: return "Supercèl·lula (Tuba/Funnel)"
        if usable_cape_val > 1800 and srh_0_3 > 250 and shear_0_6 > 18: return "Supercèl·lula (Mur de núvols)"
        if usable_cape_val > 2000 and shear_0_6 > 18 and srh_0_3 > 150: return "Supercèl·lula"
        if usable_cape_val > 1500 and shear_0_6 > 12 and not (srh_0_3 > 150): return "Cumulonimbus (Shelf Cloud)"
        if usable_cape_val > 1200 and s_0_1 > 8: return "Cumulonimbus (Base Rugosa)"
        return "Cumulonimbus (Multicèl·lula)"

    if base_clouds:
        return re.sub(r'\s*\([^)]*\)', '', base_clouds[0])
    
    return "Cel Serè"


# =========================================================================
# === 3. FUNCIONS DE DIBUIX ===============================================
# =========================================================================
def _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active):
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    usable_cape_val = max(0, cape.m - abs(cin.m))
    
    if usable_cape_val <= 10 or not lcl_p:
        return None, None
        
    cloud_base_km = lcl_h / 1000.0
    
    if convergence_active:
        cloud_top_km = el_h / 1000.0 if el_h > lcl_h else cloud_base_km
    else:
        if not lfc_p:
            cloud_top_km = cloud_base_km + 0.1
        else:
            try:
                rh = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile)
                indices_above_lcl = np.where(p_levels <= lcl_p)[0]
                p_top = p_levels[-1]
                if len(indices_above_lcl) > 0:
                    for idx in indices_above_lcl:
                        if rh[idx] < 0.7: 
                            p_top = p_levels[idx]
                            break
                cloud_top_km = mpcalc.pressure_to_height_std(p_top).to('km').m
            except:
                cloud_top_km = cloud_base_km
    return (cloud_base_km, cloud_top_km) if cloud_base_km is not None and cloud_top_km is not None and cloud_top_km > cloud_base_km else (None, None)

def _get_cloud_color(y, base, top, b_min=0.6, b_max=0.95):
    if top <= base: return (b_min,) * 3
    return (np.clip(b_min + (b_max-b_min)*((y-base)/(top-base))**0.7,0,1),)*3

def _draw_cumulonimbus(ax, base_km, top_km):
    updraft_center_x, num_points = 0, 20
    altitudes = np.linspace(base_km, top_km, num_points)
    anvil_base_alt = top_km * 0.8
    tower_indices = np.where(altitudes < anvil_base_alt)[0]
    if len(tower_indices) == 0: tower_indices = np.arange(len(altitudes))
    tower_alts = altitudes[tower_indices]
    widths = 0.5 * (1 + 0.8 * np.sin(np.pi * (tower_alts - base_km) / (top_km - base_km)))
    widths += np.random.uniform(-0.05, 0.05, len(tower_indices))
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
    anvil_altitudes = np.linspace(anvil_base_alt, top_km, 10)
    anvil_spread = 1.5 + random.uniform(-0.2, 0.2)
    for _ in range(80):
        y = random.uniform(anvil_base_alt, top_km)
        height_factor = 1 + (y - anvil_base_alt) / (top_km - anvil_base_alt)
        x = updraft_center_x + random.uniform(-anvil_spread * height_factor, anvil_spread * height_factor)
        width = random.uniform(0.5, 1.2) * height_factor
        height = random.uniform(0.05, 0.15)
        color = tuple([random.uniform(0.95, 1.0)]*3)
        ax.add_patch(Ellipse((x, y), width, height, facecolor=color, alpha=random.uniform(0.1, 0.3), lw=0, zorder=12))

def _draw_cumulus_mediocris(ax, base_km, top_km):
    center_x = 0
    num_particles = 250
    cloud_height = top_km - base_km
    altitudes = np.linspace(base_km, top_km, 20)
    base_width = 0.4 * (1 + 0.8 * np.sin(np.pi * (altitudes - base_km) / (cloud_height + 0.01)))
    noise = np.random.uniform(-0.1, 0.1, len(altitudes))
    widths = base_width + noise
    widths[0] = max(widths[0], 0.3)
    r_pts = [(center_x + widths[i], altitudes[i]) for i in range(len(altitudes))]
    l_pts = [(center_x - widths[i], altitudes[i]) for i in range(len(altitudes))]
    main_poly_pts = [l_pts[0]] + r_pts + l_pts[::-1]
    ax.add_patch(Polygon(main_poly_pts, facecolor='#d0d0d0', lw=0, zorder=10))
    patches = []
    for _ in range(num_particles):
        y_progress = random.betavariate(2, 2)
        y = base_km + y_progress * cloud_height
        max_x_at_y = np.interp(y, altitudes, widths)
        x = center_x + random.uniform(-max_x_at_y, max_x_at_y) * 0.95
        size = random.uniform(0.15, 0.5) * (1 + y_progress * 0.5)
        min_bright, max_bright = 0.8, 1.0
        brightness = min_bright + (max_bright - min_bright) * (y_progress ** 0.7)
        color = (brightness, brightness, brightness)
        alpha = random.uniform(0.15, 0.45)
        patch = Circle((x, y), size, facecolor=color, alpha=alpha, lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=11))

def _draw_cumulus_castellanus(ax, base_km, top_km):
    base_thickness = min(0.8, (top_km - base_km) * 0.25)
    patches_base = []
    for _ in range(120):
        x = random.uniform(-1.7, 1.7)
        y = base_km + (random.random() ** 2) * base_thickness
        b = random.uniform(0.8, 0.9)
        patch = Ellipse((x, y), width=random.uniform(0.7, 1.6), height=random.uniform(0.1, 0.25), facecolor=(b, b, b), alpha=random.uniform(0.1, 0.3), lw=0)
        patches_base.append(patch)
    ax.add_collection(PatchCollection(patches_base, match_original=True, zorder=8))
    num_turrets = random.randint(3, 5)
    turret_base_y = base_km + base_thickness * 0.5
    for i in range(num_turrets):
        turret_center_x = random.uniform(-1.3, 1.3)
        turret_top_y = turret_base_y + random.uniform(0.5, 0.95) * (top_km - turret_base_y)
        turret_height = turret_top_y - turret_base_y
        max_width = random.uniform(0.25, 0.4)
        patches_turret = []
        for _ in range(random.randint(60, 90)):
            y = turret_base_y + (random.random() ** 0.8) * turret_height
            normalized_y_in_turret = (y - turret_base_y) / turret_height
            current_width = max_width * np.sin(np.pi * normalized_y_in_turret)
            x = turret_center_x + random.uniform(-current_width * 0.9, current_width * 0.9)
            size = random.uniform(0.1, 0.3) * (1 + normalized_y_in_turret * 0.5)
            brightness = 0.75 + (0.98 - 0.75) * (normalized_y_in_turret ** 0.8)
            patch = Circle((x, y), size, facecolor=(brightness, brightness, brightness), alpha=random.uniform(0.2, 0.5), lw=0)
            patches_turret.append(patch)
        ax.add_collection(PatchCollection(patches_turret, match_original=True, zorder=9 + i))

def _draw_nimbostratus(ax, base_km, top_km, cloud_type):
    color, alpha = '#a9a9a9', 0.9
    ax.add_patch(Rectangle((-1.7, base_km), 3.4, top_km - base_km, facecolor=color, lw=0, zorder=8, alpha=alpha))
    patches = []
    for _ in range(150):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.6, 0.75)
        patch = Ellipse((x, y), width=random.uniform(0.8, 1.5), height=random.uniform(0.1, 0.3), facecolor=(b, b, b), alpha=random.uniform(0.2, 0.4), lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))

def _draw_cumulus_fractus(ax, base_km, thickness):
    patches=[Ellipse((random.gauss(0,0.5),random.uniform(base_km,base_km+thickness)), random.uniform(0.2,0.4), random.uniform(0.3,0.7)*random.uniform(0.2,0.4), angle=random.uniform(-25,25), facecolor=_get_cloud_color(random.uniform(base_km,base_km+thickness),base_km,base_km+thickness,b_min=0.6,b_max=0.8), alpha=0.5,lw=0) for _ in range(150)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=10))

def _draw_stratiform_cotton_clouds(ax, base_km, top_km):
    patches = []
    for _ in range(200):
        x = random.uniform(-1.7, 1.7)
        y = random.uniform(base_km, top_km)
        b = random.uniform(0.88, 0.98)
        patch = Ellipse((x, y), random.uniform(0.4, 0.9), random.uniform(0.15, 0.3), facecolor=(b, b, b), alpha=random.uniform(0.3, 0.6), lw=0)
        patches.append(patch)
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=9))

def _draw_clear_sky(ax):
    patches = [Ellipse((random.uniform(-1.5,1.5), random.uniform(10,14)), random.uniform(0.5,1.0), random.uniform(0.1,0.2), facecolor='white', alpha=random.uniform(0.05,0.1), lw=0) for _ in range(15)]
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=5))

def _draw_precipitation(ax, precip_base_km, ground_km, p_type, center_x=0.0, sub_cloud_rh=0.4):
    if p_type == 'virga':
        alpha = np.clip(sub_cloud_rh * 0.6, 0.15, 0.55)
        fall_percentage = sub_cloud_rh / 0.5
        fall_distance = (precip_base_km - ground_km) * fall_percentage
        end_y = precip_base_km - fall_distance
        if sub_cloud_rh < 0.5: end_y = max(end_y, ground_km + 0.3)
        else: end_y = ground_km
        top_width = random.uniform(0.6, 0.9)
        bottom_width = top_width * 0.5
        points = [(center_x - top_width / 2, precip_base_km), (center_x + top_width / 2, precip_base_km), (center_x + bottom_width / 2, end_y), (center_x - bottom_width / 2, end_y)]
        ax.add_patch(Polygon(points, facecolor='cornflowerblue', alpha=alpha, lw=0, zorder=7))
    elif p_type in ['rain', 'sleet']:
        width = 1.6
        ax.add_patch(Rectangle((center_x - width / 2, ground_km), width, precip_base_km - ground_km, facecolor='cornflowerblue', alpha=0.35, lw=0, zorder=5))
    elif p_type == 'hail':
        ax.scatter(center_x+np.random.normal(0,0.3,150),np.random.uniform(ground_km,precip_base_km,150), s=np.random.uniform(5,40,150),c='white',alpha=0.8,marker='o',edgecolor='gray',linewidth=0.5,zorder=8)
    elif p_type == 'snow':
        ax.scatter(center_x+np.random.normal(0,0.5,300),np.random.uniform(ground_km,precip_base_km,300), s=np.random.uniform(20,70,300),c='white',alpha=np.random.uniform(0.4,0.9,300),marker='*',zorder=8)

def _draw_saturation_layers(ax, p_levels, t_profile, td_profile):
    try:
        saturated_indices = np.where(t_profile.m-td_profile.m <= 1.5)[0]
        if not len(saturated_indices): return
        i=0
        while i < len(saturated_indices):
            start_idx, j = saturated_indices[i], i
            while j+1 < len(saturated_indices) and saturated_indices[j+1]==saturated_indices[j]+1: j+=1
            end_idx = saturated_indices[j]
            h_bottom = mpcalc.pressure_to_height_std(p_levels[start_idx]).to('km').m
            h_top = mpcalc.pressure_to_height_std(p_levels[end_idx]).to('km').m
            if h_top - h_bottom < 0.05: i=j+1; continue
            patches=[]
            for _ in range(int(100+300*(h_top-h_bottom))):
                y, x = random.uniform(h_bottom,h_top), random.uniform(-1.5,1.5)
                brightness = random.uniform(0.65,0.85)
                patches.append(Ellipse((x,y),random.uniform(0.3,0.8),random.uniform(0.05,0.1)*(1+h_top-h_bottom), facecolor=(brightness,)*3,alpha=random.uniform(0.1,0.5),lw=0))
            ax.add_collection(PatchCollection(patches, match_original=True, zorder=7))
            i=j+1
    except Exception: pass

def _draw_base_feature(ax, f_type, base_x_left, base_x_right, base_y, ground_y):
    z, center_x, width = 12, (base_x_left + base_x_right) / 2, base_x_right - base_x_left
    if f_type == 'lowering':
        ax.add_patch(Polygon([(base_x_left, base_y), (base_x_right, base_y), (base_x_right * 0.9 + center_x * 0.1, base_y - 0.2), (base_x_left * 0.9 + center_x * 0.1, base_y - 0.2)], facecolor='dimgray', edgecolor='gray', zorder=z))
    elif f_type == 'wall_cloud':
        top_l, top_r = center_x - (width * 0.75 / 2), center_x + (width * 0.75 / 2)
        bot_l, bot_r = center_x - (width * 0.55 / 2), center_x + (width * 0.55 / 2)
        ax.add_patch(Polygon([(top_l, base_y), (top_r, base_y), (bot_r, base_y - 0.35), (bot_l, base_y - 0.35)], facecolor='#383838', edgecolor='#202020', lw=0.5, zorder=z))
    elif f_type == 'funnel':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, max(base_y - 0.8, ground_y + 0.5))], facecolor='darkgray', alpha=0.8, zorder=z))
    elif f_type == 'tornado':
        ax.add_patch(Polygon([(center_x - 0.2, base_y), (center_x + 0.2, base_y), (center_x, ground_y)], facecolor='#505050', zorder=z))
        ax.add_patch(Ellipse((center_x, ground_y + 0.05), width=0.7, height=0.25, facecolor='#654321', alpha=0.7, zorder=z + 1))

def create_skewt_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    ax = skew.ax
    ax.set_ylim(1050, 100)
    ax.set_xlim(-50, 45)
    td_profile = np.minimum(t_profile, td_profile)
    with integrator_lock:
        skew.plot_dry_adiabats(alpha=0.3, color='orange')
        skew.plot_moist_adiabats(alpha=0.3, color='green')
        skew.plot_mixing_lines(alpha=0.4, color='blue', linestyle='--')
        skew.plot(p_levels, t_profile, 'r', linewidth=2, label='Temperatura (T)')
        skew.plot(p_levels, td_profile, 'b', linewidth=2, label='Punt de Rosada (Td)')
        parcel_prof = mpcalc.parcel_profile(p_levels, t_profile[0], td_profile[0]).to('degC')
        skew.plot(p_levels, parcel_prof, 'k--', linewidth=2, label='Bombolla Adiabàtica')
        wb_profile = mpcalc.wet_bulb_temperature(p_levels, t_profile, td_profile)
        skew.plot(p_levels, wb_profile, color='purple', linewidth=1.5, label='Tª Bombolla Humida')
        skew.shade_cape(p_levels, t_profile, parcel_prof, facecolor='yellow', alpha=0.3)
        skew.shade_cin(p_levels, t_profile, parcel_prof, facecolor='black', alpha=0.3)
    
    _, _, lcl_p, _, lfc_p, _, el_p, _, _, fz_lvl = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    xlims = ax.get_xlim()
    if lcl_p: ax.plot(xlims, [lcl_p.m, lcl_p.m], 'gray', linestyle='--', label='LCL')
    if lfc_p: ax.plot(xlims, [lfc_p.m, lfc_p.m], 'purple', linestyle='--', label='LFC')
    if el_p: ax.plot(xlims, [el_p.m, el_p.m], 'red', linestyle='--', label='EL')
    
    if fz_lvl is not None and not np.isnan(fz_lvl.m):
        ax.plot(xlims, [fz_lvl.m, fz_lvl.m], 'c', linestyle='-.', linewidth=1.5, label='Isoterma 0°C')
        
    ax.legend()
    plt.tight_layout()
    return fig

def create_cloud_drawing_figure(p_levels, t_profile, td_profile, convergence_active, precipitation_type, lfc_h, cape, base_km, top_km, cloud_type):
    fig, ax = plt.subplots(figsize=(5, 8))
    surface_height_m = mpcalc.pressure_to_height_std(p_levels[0]).to('m').m
    ax.set(ylim=(-0.5, 16), xlim=(-1.5,1.5), xticks=[], yticks=np.arange(0, 17, 2))
    ax.set_ylabel("Altitud sobre el terra (km)"); ax.set_title("Visualització del Núvol")
    ax.grid(True, linestyle='dashdot', alpha=0.5); ax.set_facecolor('#6495ED')
    ax.add_patch(Circle((1.2, 14.5), 0.2, color='#FFFACD', alpha=0.9, zorder=1))
    ground_color = 'white' if precipitation_type == 'snow' else '#8B4513'
    ax.add_patch(Rectangle((-1.5, -0.5), 3, 0.5, color=ground_color, zorder=3))

    base_agl_km = (base_km * 1000 - surface_height_m) / 1000 if base_km is not None else None
    top_agl_km = (top_km * 1000 - surface_height_m) / 1000 if top_km is not None else None
    
    if not convergence_active:
        _draw_saturation_layers(ax, p_levels, t_profile, td_profile) 
        
    if base_agl_km is not None and top_agl_km is not None and (top_agl_km - base_agl_km > 0.1):
        if "Nimbostratus" in cloud_type or "Hivernal" in cloud_type: _draw_nimbostratus(ax, base_agl_km, top_agl_km, cloud_type)
        elif "Altostratus" in cloud_type: _draw_stratiform_cotton_clouds(ax, base_agl_km, top_agl_km)
        elif "Cirrus" in cloud_type: _draw_clear_sky(ax)
        elif "Supercèl·lula" in cloud_type or "Cumulonimbus" in cloud_type: _draw_cumulonimbus(ax, base_agl_km, top_agl_km)
        elif "Castellanus" in cloud_type or "Altocumulus" in cloud_type: _draw_cumulus_castellanus(ax, base_agl_km, top_agl_km)
        elif "Cumulus" in cloud_type: _draw_cumulus_mediocris(ax, base_agl_km, top_agl_km)
        elif "Fractus" in cloud_type: _draw_cumulus_fractus(ax, base_agl_km, top_agl_km - base_agl_km)
    elif not np.any((t_profile.m - td_profile.m) <= 1.5):
        _draw_clear_sky(ax)

    if precipitation_type and base_agl_km is not None:
        precip_base_km = base_agl_km
        sub_cloud_rh_mean = 0.4
        try:
            p_base_precip = mpcalc.height_to_pressure_std((base_agl_km + surface_height_m / 1000) * units.kilometer)
            p_ground = p_levels[0]
            sub_cloud_mask = (p_levels >= p_base_precip) & (p_levels <= p_ground)
            if np.any(sub_cloud_mask):
                rh_profile = mpcalc.relative_humidity_from_dewpoint(t_profile, td_profile)
                sub_cloud_rh_mean = np.mean(rh_profile[sub_cloud_mask]).magnitude
        except Exception: pass
        _draw_precipitation(ax, precip_base_km, 0, precipitation_type, sub_cloud_rh=sub_cloud_rh_mean)
    plt.tight_layout()
    return fig

def create_cloud_structure_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir, convergence_active):
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1), wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_shear = fig.add_subplot(gs[0, 1], sharey=ax)
    surface_height_m = mpcalc.pressure_to_height_std(p_levels[0]).to('m').m
    
    ax.set_title("Estructura Vertical i Cisallament", fontsize=10); ax.set_facecolor('skyblue')
    ax.add_patch(Rectangle((-1.5, -0.5), 3, 0.5, color='darkgreen', zorder=1))
    ax.set(ylim=(-0.5, 20), xlim=(-1.5, 1.5), ylabel="Altitud sobre el terra (km)", xticks=[]); ax.grid(True, linestyle='--', alpha=0.3)
    ax_shear.set(xlim=(-1, 1), xticks=[]); ax_shear.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ax_shear.spines.values(): spine.set_visible(False)
    ax_shear.patch.set_alpha(0.0)
    
    cape, cin, _, lcl_h, _, _, _, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    usable_cape_val = max(0, cape.m - abs(cin.m))
    
    base_km, top_km = _calculate_dynamic_cloud_heights(p_levels, t_profile, td_profile, convergence_active)
    
    if not base_km or not top_km or usable_cape_val < 50 or not convergence_active:
        ax.text(0.5, 0.5, "Sense Estructura Convectiva\n(Energia neta insuficient o forçament inactiu)", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='white', bbox=dict(facecolor='darkblue', alpha=0.7))
        ax_shear.axis('off'); return fig
    
    base_agl_km = (base_km * 1000 - surface_height_m) / 1000
    top_agl_km = (top_km * 1000 - surface_height_m) / 1000
    visual_base_km = max(base_agl_km, 0.1)
    
    try:
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        h_msl_km = mpcalc.pressure_to_height_std(p_levels).to('km').m
        h_agl_km = h_msl_km - (surface_height_m / 1000)

        unique_h, idx = np.unique(h_agl_km, return_index=True)
        if len(unique_h) < 2: return fig
        
        f_u, f_v = interp1d(unique_h, u.m[idx], bounds_error=False, fill_value='extrapolate'), interp1d(unique_h, v.m[idx], bounds_error=False, fill_value='extrapolate')
        barb_heights = np.arange(0, min(20, h_agl_km.max()), 1)
        ax_shear.barbs(np.zeros_like(barb_heights), barb_heights, (f_u(barb_heights) * units('m/s')).to('knots').m, (f_v(barb_heights) * units('m/s')).to('knots').m, length=7, pivot='middle', color='k')
        
        altitudes = np.linspace(visual_base_km, top_agl_km, num=50)
        u_at_alts = f_u(altitudes)
        horizontal_offsets = u_at_alts * 0.02
        shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
        shear_factor = np.clip(shear_0_6 / 35, 0.4, 2.5)
        updraft_widths = 0.4 * (1 + 0.5 * np.sin(np.pi * (altitudes - visual_base_km) / (top_agl_km - visual_base_km + 0.01))) * shear_factor
        
        anvil_extension = np.zeros_like(altitudes)
        if (top_agl_km - visual_base_km) > 4.0:
            anvil_base_alt = top_agl_km * 0.80
            anvil_indices = np.where(altitudes >= anvil_base_alt)[0]
            if len(anvil_indices) > 0:
                u_anvil_top = f_u(top_agl_km)
                wind_direction = np.sign(u_anvil_top) if u_anvil_top != 0 else 1
                max_stretch = abs(u_anvil_top) * 0.06
                growth_factor = (altitudes[anvil_indices] - anvil_base_alt) / (top_agl_km - anvil_base_alt)
                anvil_extension[anvil_indices] = max_stretch * wind_direction * growth_factor**1.5
        r_pts = [(updraft_widths[i] + horizontal_offsets[i] + anvil_extension[i], altitudes[i]) for i in range(len(altitudes))]
        l_pts = [(-updraft_widths[i] + horizontal_offsets[i], altitudes[i]) for i in range(len(altitudes))]
        ax.add_patch(Polygon(r_pts + l_pts[::-1], facecolor='white', edgecolor='lightgray', alpha=0.95, zorder=10))
        
        lcl_agl = lcl_h - surface_height_m
        feature = None
        if top_agl_km - base_agl_km > 4.0 and usable_cape_val > 500:
            if (srh_0_1 >= 150 and lcl_agl <= 1000 and shear_0_6 > 15): feature = 'tornado'
            elif (srh_0_1 > 100 and lcl_agl < 1200 and shear_0_6 > 12): feature = 'funnel'
            elif srh_0_3 > 150 and shear_0_6 > 18 and usable_cape_val > 1000: feature = 'wall_cloud'
            elif s_0_1 > 8 and lcl_agl < 1500: feature = 'lowering'
        if feature:
            _draw_base_feature(ax, feature, l_pts[0][0], r_pts[0][0], visual_base_km, 0)
    except Exception as e: pass
    plt.tight_layout()
    return fig

def create_orography_figure(lfc_h, surface_height_m, fz_h, lcl_h):
    """
    Crea un gràfic visual de la muntanya necessària per assolir el LFC, 
    o el LCL si no hi ha LFC, de forma robusta.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- 1. Cel Atmosfèric i Sol ---
    n_steps = 256
    color_top = np.array([0.1, 0.3, 0.8])
    color_bottom = np.array([0.7, 0.8, 1.0])
    gradient_colors = np.array([np.linspace(c1, c2, n_steps) for c1, c2 in zip(color_top, color_bottom)]).T
    sky_cmap = ListedColormap(gradient_colors)
    ax.imshow(np.arange(n_steps).reshape(-1, 1), aspect='auto', cmap=sky_cmap, extent=[-2, 2, 0, 10], origin='lower', zorder=0)
    
    ax.add_patch(Circle((1.5, 8.5), 0.8, color='yellow', alpha=0.3, zorder=1))
    ax.add_patch(Circle((1.5, 8.5), 0.6, color='yellow', alpha=0.5, zorder=1))
    ax.add_patch(Circle((1.5, 8.5), 0.4, color='#FFFFE0', alpha=1.0, zorder=1))

    # --- 2. Configuració General ---
    ax.set_title("Potencial d'Activació per Orografia", fontsize=14, weight='bold')
    ax.set_ylabel("Altitud sobre el terra (km)")
    ax.set_xticks([])
    ax.set_xlim(-2, 2)
    
    # --- 3. Càlculs d'Alçada (AGL) robustos ---
    if lfc_h is None or np.isnan(lfc_h): lfc_h = np.inf
    if lcl_h is None or np.isnan(lcl_h): lcl_h = 0
    if fz_h is None or np.isnan(fz_h): fz_h = 0
    if surface_height_m is None or np.isnan(surface_height_m): surface_height_m = 0

    lfc_agl_m = (lfc_h - surface_height_m) if lfc_h != np.inf else np.inf
    lcl_agl_m = lcl_h - surface_height_m
    fz_h_agl_m = (fz_h - surface_height_m) if fz_h > 0 else np.inf
    
    has_lfc = lfc_agl_m != np.inf
    target_height_m = lfc_agl_m if has_lfc else lcl_agl_m
    drawing_height_m = max(target_height_m, 100)
    drawing_height_km = drawing_height_m / 1000.0
    rock_line_km = 1.6

    # --- 4. Dibuix de la Muntanya i l'Entorn ---
    mountain_points = [
        (-2, 0), (-1.5, 0.2 * drawing_height_km), (-1.1, 0.15 * drawing_height_km),
        (-0.7, 0.6 * drawing_height_km), (0, drawing_height_km), (0.6, 0.5 * drawing_height_km),
        (1.2, 0.2 * drawing_height_km), (2, 0)
    ]
    mountain_path = Polygon(mountain_points, color='none', zorder=5)
    ax.add_patch(mountain_path)

    def generate_texture(num, y_min, y_max, colors, size_range=(0.05, 0.15)):
        patches = []
        for _ in range(num):
            x, y = random.uniform(-2, 2), random.uniform(y_min, y_max)
            size, color = random.uniform(*size_range), colors[random.randint(0, len(colors)-1)]
            patches.append(Circle((x, y), size, color=color, lw=0, alpha=random.uniform(0.7, 1.0)))
        collection = PatchCollection(patches, match_original=True)
        collection.set_clip_path(mountain_path)
        ax.add_collection(collection)

    forest_colors, alpine_grass_colors, rock_colors, snow_colors = ['#003300', '#004d00', '#006400'], ['#556B2F', '#6B8E23', '#808000'], ['#696969', '#808080', '#A9A9A9'], ['#F0F8FF', '#E6E6FA', '#FFFFFF']
    generate_texture(800, 0, 0.3, forest_colors)
    generate_texture(1500, 0.3, rock_line_km, alpine_grass_colors)
    if drawing_height_km > rock_line_km:
        generate_texture(2000, rock_line_km, drawing_height_km, rock_colors)
    if drawing_height_km > fz_h_agl_m / 1000.0:
        generate_texture(1500, fz_h_agl_m / 1000.0, drawing_height_km, snow_colors)

    highlight_path = Polygon([(-2, 0), (-1.5, 0.2 * drawing_height_km), (-0.7, 0.6 * drawing_height_km), (0, drawing_height_km), (0,0)], color='white', alpha=0.1, zorder=6)
    highlight_path.set_clip_path(mountain_path); ax.add_patch(highlight_path)
    shadow_path = Polygon([(0, drawing_height_km), (0.6, 0.5 * drawing_height_km), (1.2, 0.2 * drawing_height_km), (2, 0), (0,0)], color='black', alpha=0.3, zorder=6)
    shadow_path.set_clip_path(mountain_path); ax.add_patch(shadow_path)

    def draw_volumetric_cloud_layer(y_center, thickness, num_puffs):
        for _ in range(num_puffs):
            x, y = random.uniform(-2, 2), y_center + random.gauss(0, thickness)
            base_size = random.uniform(0.1, 0.3)
            for i in range(5):
                offset_x, offset_y = random.gauss(0, base_size * 0.3), random.gauss(0, base_size * 0.3)
                size, brightness = base_size * random.uniform(0.5, 1.0), random.uniform(0.8, 1.0)
                ax.add_patch(Circle((x + offset_x, y + offset_y), size, color=(brightness, brightness, brightness), alpha=0.15, lw=0, zorder=4))
    
    if lcl_agl_m > 0: draw_volumetric_cloud_layer(lcl_agl_m / 1000.0, 0.08, 30)

    ground_colors = ['#556B2F', '#8B4513', '#228B22']
    for _ in range(500): ax.add_patch(Circle((random.uniform(-2, 2), random.uniform(-0.1, 0.05)), random.uniform(0.05,0.1), color=ground_colors[random.randint(0,2)], lw=0, zorder=9))
    for i in range(15):
        x_base, height = random.uniform(-2, 2), random.uniform(0.1, 0.4)
        ax.add_patch(Polygon([(x_base - 0.05, 0), (x_base, height), (x_base + 0.05, 0)], color='#001a00', zorder=10))

    # --- 5. Anotacions i Text ---
    ax.axhline(y=lcl_agl_m / 1000.0, color='gray', linestyle='--', linewidth=2, zorder=8)
    ax.text(ax.get_xlim()[0], lcl_agl_m / 1000.0, f' LCL ({lcl_agl_m:.0f} m)  ', color='white', va='center', ha='right', weight='bold', bbox=dict(facecolor='black', boxstyle='round,pad=0.2'))
    
    if has_lfc:
        ax.axhline(y=lfc_agl_m / 1000.0, color='red', linestyle='--', linewidth=2, zorder=8)
        ax.text(ax.get_xlim()[1], lfc_agl_m / 1000.0, f'  LFC ({lfc_agl_m:.0f} m)', color='red', va='center', ha='left', weight='bold', bbox=dict(facecolor='white', boxstyle='round,pad=0.2'))
        ax.text(0.5, 0.97, f"Altura de muntanya necessària per activar tempestes: {target_height_m:.0f} m", ha='center', va='top', color='black', fontsize=12, weight='bold', transform=ax.transAxes, bbox=dict(facecolor='yellow', boxstyle='round,pad=0.5'))
    else:
        ax.text(0.5, 0.97, "No hi ha LFC accessible.\nL'orografia no pot iniciar convecció profunda.", ha='center', va='top', color='white', fontsize=12, weight='bold', transform=ax.transAxes, bbox=dict(facecolor='darkblue', alpha=0.8, boxstyle='round,pad=0.5'))

    if fz_h_agl_m != np.inf and drawing_height_m > fz_h_agl_m:
        ax.axhline(y=fz_h_agl_m / 1000.0, color='cyan', linestyle=':', linewidth=1.5, zorder=8)
        ax.text(ax.get_xlim()[1], fz_h_agl_m / 1000.0, f'  Isoterma 0°C ({fz_h_agl_m:.0f} m)', color='cyan', va='center', ha='left', weight='bold', bbox=dict(facecolor='black', boxstyle='round,pad=0.2'))
    
    ax.set_ylim(0, max(drawing_height_km * 1.5, 4))
    plt.tight_layout(pad=0.5)
    return fig

def create_radar_figure(p_levels, t_profile, td_profile, wind_speed, wind_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('darkslategray'); ax.set_title("Eco Radar Simulat", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7, labelbottom=False, labelleft=False)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.grid(True, linestyle=':', alpha=0.3, color='white')
    
    cape, cin, *rest = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    usable_cape_val = max(0, cape.m - abs(cin.m))
    
    try:
        heights_amsl = mpcalc.pressure_to_height_std(p_levels).to('m')
        heights_agl = (heights_amsl - heights_amsl[0]).to('km')
        layer_mask = (heights_agl.m >= 0) & (heights_agl.m <= 4)
        if np.sum(layer_mask) > 2:
            rh_layer = mpcalc.relative_humidity_from_dewpoint(t_profile[layer_mask], td_profile[layer_mask])
            pwat_layer = mpcalc.precipitable_water(p_levels[layer_mask], td_profile[layer_mask]).to('mm')
            if np.mean(rh_layer) > 0.85 and usable_cape_val < 250:
                x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
                max_dbz = np.clip(15 + pwat_layer.m, 15, 45)
                noise = gaussian_filter(np.random.randn(100, 100), sigma=8) * (max_dbz * 0.2)
                Z = np.clip(max_dbz + noise, 0, 50)
                radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900']
                radar_levels = [0, 15, 20, 25, 30, 35, 45]
                radar_cmap = ListedColormap(radar_colors)
                radar_norm = BoundaryNorm(radar_levels, radar_cmap.N)
                ax.contourf(x, y, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
                return fig
    except Exception: pass
    
    if usable_cape_val < 100:
        ax.text(0, 0, "Sense precipitació significativa", ha='center', va='center', color='white', fontsize=9)
        return fig
        
    shear_0_6, *_ = calculate_storm_parameters(p_levels, wind_speed, wind_dir)
    _, _, lcl_p, _, lfc_p, _, el_p, _, _, _ = calculate_thermo_parameters(p_levels, t_profile, td_profile)
    mean_u, mean_v = (0,0) * units('m/s')
    if lfc_p and el_p:
        p_mask = (p_levels >= el_p) & (p_levels <= lfc_p)
        if np.sum(p_mask) > 1:
            u, v = mpcalc.wind_components(wind_speed[p_mask], wind_dir[p_mask])
            mean_u, mean_v = np.mean(u), np.mean(v)
            
    max_dbz = np.clip(20 + (usable_cape_val / 3000) * 55, 20, 75)
    elongation = np.clip(1 + (shear_0_6 / 20), 1, 2.5)
    angle_rad = np.arctan2(mean_u.m, mean_v.m)
    x, y = np.linspace(-50, 50, 150), np.linspace(-50, 50, 150)
    xx, yy = np.meshgrid(x, y)
    x_rot, y_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad), -xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    sigma_x, sigma_y = 15, 15 / elongation
    Z = max_dbz * np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
    Z += gaussian_filter(np.random.randn(150, 150), sigma=6) * (max_dbz * 0.1)
    Z = np.clip(Z, 0, 75)
    radar_colors = ['#00a0f0', '#0000ff', '#00ff00', '#008000', '#ffff00', '#ff9900', '#ff0000', '#c80000', '#ff00ff', '#960096']
    radar_levels = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75]
    radar_cmap, radar_norm = ListedColormap(radar_colors), BoundaryNorm(radar_levels, len(radar_colors))
    ax.contourf(xx, yy, Z, levels=radar_levels, cmap=radar_cmap, norm=radar_norm)
    return fig

def create_hodograph_figure(p, ws, wd, t, td):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    h = Hodograph(ax, component_range=40.)
    h.add_grid(increment=10, ls='--', color='gray')
    ax.set_xlabel('kt'); ax.set_ylabel('kt')
    
    try:
        p_hodo, ws_hodo, wd_hodo = p.to('hPa'), ws.to('kt'), wd.to('deg')
        u, v = mpcalc.wind_components(ws_hodo, wd_hodo)
        heights = mpcalc.pressure_to_height_std(p_hodo).to('km')
        h_interp = np.arange(0, min(12, heights.m.max()), 0.1) * units.km
        u_interp = np.interp(h_interp.m, heights.m, u.m) * units.kt
        v_interp = np.interp(h_interp.m, heights.m, v.m) * units.kt
        levels, colors = [0, 1, 3, 5, 8, 10], ['green', 'orange', 'red', 'purple', 'darkviolet']
        cmap, norm = ListedColormap(colors), BoundaryNorm(levels, len(colors))
        for i in range(len(h_interp) - 1):
            ax.plot(u_interp[i:i+2].m, v_interp[i:i+2].m, color=cmap(norm(h_interp[i].m)), linewidth=2)
        with integrator_lock:
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p_hodo, u, v, heights)
        ax.arrow(0, 0, rm[0].m, rm[1].m, color='black', width=0.5, head_width=2, length_includes_head=True, label="Moviment Tempesta (MD)")
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, pad=0.08)
        cbar.set_label('Altitud (km)')
    except Exception as e:
        ax.text(0.5, 0.5, "Dades de vent insuficients\nper generar hodògraf.", 
                ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    return fig

# =========================================================================
# === 4. ESTRUCTURA DE L'APLICACIÓ =======================================
# =========================================================================

def show_welcome_screen():
    set_main_background()
    # Aquesta funció ja NO ha de configurar la barra lateral.
    
    st.markdown(f'<p class="welcome-title">{get_text("welcome_title")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="welcome-subtitle">{get_text("welcome_subtitle")}</p>', unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="mode-card"><h3>{get_text("live_mode_title")}</h3><p>{get_text("live_mode_text")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("live_mode_button"), use_container_width=True, key='btn_live'):
            st.session_state.app_mode = 'live'
            st.rerun()
    with col2:
        st.markdown(f"""<div class="mode-card"><h3>{get_text("sandbox_mode_title")}</h3><p>{get_text("sandbox_mode_text")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("sandbox_mode_button"), use_container_width=True, key='btn_sandbox'):
            st.session_state.app_mode = 'sandbox'
            st.rerun()
    with col3:
        st.markdown(f"""<div class="mode-card"><h3>{get_text("manual_mode_title")}</h3><p>{get_text("manual_mode_text")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("manual_mode_button"), use_container_width=True, type="primary", key='btn_manual'):
            st.session_state.app_mode = 'manual'
            st.rerun()

    st.markdown(f"""
    <div class="coming-soon">
        <p>{get_text("coming_soon_title")}</p>
        <h2>{get_text("coming_soon_subtitle")}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <p>Versió 0.9.0 | © tempestes.cat</p>
    </div>
    """, unsafe_allow_html=True)

def show_full_analysis_view(p, t, td, ws, wd, obs_time, is_sandbox_mode=False, orography_preset=0):
    st.markdown(f"#### {obs_time}")
    
    title_key, data_for_message, color = generate_public_warning(p, t, td, ws, wd)
    title_text = get_text(title_key)
    desc_key = title_key.replace('_title', '_desc')
    message_text = get_text(desc_key, **data_for_message)

    st.markdown(f"""<div style="background-color:{color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;"><h3 style="color:white; text-align:center;">{title_text}</h3><p style="color:white; text-align:center; font-size:16px;">{message_text}</p></div>""", unsafe_allow_html=True)
    
    cape, cin, lcl_p, lcl_h, lfc_p, lfc_h, el_p, el_h, fz_h, fz_lvl = calculate_thermo_parameters(p, t, td)
    usable_cape = max(0, cape.m - abs(cin.m)) * units('J/kg')
    surface_height = mpcalc.pressure_to_height_std(p[0]).to('m').m
    t_sfc = t[0].to('degC').m

    convergence_active = st.session_state.get('convergence_active', False)

    shear_0_6, s_0_1, srh_0_1, srh_0_3 = calculate_storm_parameters(p, ws, wd)
    pwat_total = mpcalc.precipitable_water(p, td).to('mm')
    
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
    
    potential_clouds = determine_potential_cloud_types(p, t, td, cape, cin, ws, wd)
    cloud_type_for_chat = get_cloud_type_for_chat(p, t, td, ws, wd, cape, cin, lcl_h, lfc_h, el_p)

    st.subheader(get_text("skewt_diagram_title"), anchor=False)
    st.pyplot(create_skewt_figure(p, t, td, ws, wd), use_container_width=True)
    st.divider()

    orography_height_for_chat = orography_preset if not is_sandbox_mode else 0
    
    if t_sfc < 5:
        chat_log, precipitation_type = generate_winter_analysis(p, t, td)
    else:
        chat_log, precipitation_type = generate_detailed_analysis(p, t, td, ws, wd, cloud_type_for_chat, None, None, pwat_0_4, surface_height, orography_height_for_chat, usable_cape)

    anomaly_count = count_parameter_anomalies(usable_cape.m, cin.m, shear_0_6, srh_0_1, srh_0_3, t_sfc)
    params_label = get_text("parameters_tab")
    if anomaly_count > 0:
        params_label += f" ( {anomaly_count} )⚠️"

    tab_keys = ["analyst_assistant_tab", "parameters_tab", "hodograph_tab", "orography_tab", "visualization_tab", "cloud_types_tab", "radar_tab"]
    tab_labels = [get_text(key) for key in tab_keys]
    tab_labels[1] = params_label
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_labels)
    
    # ... (el codi per a tab1, tab2, tab3, tab4, tab5 es manté igual) ...
    
    with tab6:
        st.subheader(get_text("cloud_list_title"))
        st.markdown(get_text("cloud_list_desc"))
        if potential_clouds:
            for cloud in potential_clouds:
                st.markdown(f"- **{cloud}**")
        else:
            st.info("Segons l'anàlisi, no s'espera formació de núvols significatius.")
        
        st.markdown("---")
        st.subheader(get_text("representative_images_title"))
        
        image_triggers = {
            "tornado": ("tornado.jpg", get_text("caption_tornado")),
            "funnel": ("funnel.jpg", get_text("caption_funnel")),
            "wall cloud": ("wallcloud.jpg", get_text("caption_wallcloud")),
            "shelf cloud": ("shelfcloud.jpg", get_text("caption_shelfcloud")),
            "base rugosa": ("scud.jpg", get_text("caption_scud")),
            "supercèl·lula": ("supercell.jpg", get_text("caption_supercell")),
            "lenticular": ("lenticularis.jpg", get_text("caption_lenticularis")),
            "castellanus": ("castellanus.jpg", get_text("caption_castellanus")),
            "fractus": ("fractus.jpg", get_text("caption_fractus")),
            "cumulonimbus": ("cumulonimbus.jpg", get_text("caption_cumulonimbus")),
            "congestus": ("congestus.jpg", get_text("caption_congestus")),
            "humilis": ("humilis.jpg", get_text("caption_humilis")),
            "cirrus": ("cirrus.jpg", get_text("caption_cirrus")),
            "altostratus": ("altostratus.jpg", get_text("caption_altostratus")),
            "nimbostratus": ("nimbostratus.jpg", get_text("caption_nimbostratus")),
            "stratus": ("stratus.jpg", get_text("caption_stratus")),
            "aiguaneu": ("sleet.jpg", get_text("caption_sleet")),
            "neu": ("snow.jpg", get_text("caption_snow"))
        }

        images_to_show = set() 
        full_text_for_images = " ".join(potential_clouds).lower() + " " + cloud_type_for_chat.lower() + " ".join([msg for _, msg in chat_log]).lower()
        
        if "tornàdica" in full_text_for_images: full_text_for_images += " tornado"
        if "mur de núvols" in full_text_for_images: full_text_for_images += " wall cloud"

        for keyword, (filename, caption) in image_triggers.items():
            if keyword in full_text_for_images:
                images_to_show.add((filename, caption))

        if images_to_show:
            for filename, caption in sorted(list(images_to_show)):
                image_base64 = get_image_as_base64(filename)
                if image_base64: 
                    st.markdown(f"<div style='margin-top: 15px; text-align: center;'><img src='{image_base64}' style='max-width: 80%; border-radius: 10px;'><p style='font-style: italic; color: grey;'>{caption}</p></div>", unsafe_allow_html=True)
                else:
                    st.warning(f"Imatge '{filename}' no trobada. Assegura't que l'arxiu existeix al directori.", icon="🖼️")
        else:
            st.info("No s'han trobat imatges representatives per als núvols detectats.")

    with tab7:
        st.subheader(get_text("radar_sim_title"))
        st.pyplot(create_radar_figure(p, t, td, ws, wd), use_container_width=True)

def show_seguiment_selection_screen():
    st.title(get_text('todays_changes_title'))
    st.markdown(get_text('select_region_desc'))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="mode-card"><h4>🔥 {get_text('most_notable_zone_title')}</h4><p>{get_text('most_notable_zone_desc')}</p></div>""", unsafe_allow_html=True)
        if st.button("Solsonès", use_container_width=True, type="primary"):
            st.session_state.province_selected = 'seguiment_destacable'
            st.rerun()
    with c2:
        st.markdown(f"""<div class="mode-card"><h4>🤔 {get_text('interesting_zone_title')}</h4><p>{get_text('interesting_zone_desc')}</p></div>""", unsafe_allow_html=True)
        if st.button("Bages", use_container_width=True):
            st.session_state.province_selected = 'seguiment_interessant'
            st.rerun()

def show_seguiment_selection_screen():
    # TRADUÏT
    st.title(get_text('todays_changes_title'))
    st.markdown(get_text('select_region_desc'))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="mode-card"><h4>🔥 {get_text('most_notable_zone_title')}</h4><p>{get_text('most_notable_zone_desc')}</p></div>""", unsafe_allow_html=True)
        if st.button("Solsonès", use_container_width=True, type="primary"):
            st.session_state.province_selected = 'seguiment_destacable'
            st.rerun()
    with c2:
        st.markdown(f"""<div class="mode-card"><h4>🤔 {get_text('interesting_zone_title')}</h4><p>{get_text('interesting_zone_desc')}</p></div>""", unsafe_allow_html=True)
        if st.button("Bages", use_container_width=True):
            st.session_state.province_selected = 'seguiment_interessant'
            st.rerun()


def run_single_sounding_mode(mode):
    """Versió neta sense sidebar propi."""
    seguiment_map = {
        'seguiment_destacable': {'file': 'sondeig_destacable.txt', 'title': "", 'comarca': "Solsonès"},
        'seguiment_interessant': {'file': 'sondeig_interessant.txt', 'title': "", 'comarca': "Bages"}
    }
    
    config = seguiment_map[mode]
    comarca = config['comarca']
    st.title(f"{config['title']} - {comarca.upper()}")

    content_placeholder = st.empty()
    with content_placeholder.container():
        # TRADUÏT
        loading_msg = get_text("loading_region", region=config['comarca'])
        show_loading_animation(message=loading_msg)
        time.sleep(1) 

    try:
        soundings = parse_all_soundings(config['file'])
        content_placeholder.empty()
        if soundings:
            data = soundings[0]
            obs_time = data.get('observation_time', f"Sondeig de la {config['title'].lower()}")
            show_full_analysis_view(
                p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                obs_time=obs_time
            )
        else:
            content_placeholder.empty()
            st.error(f"No s'han pogut carregar dades del sondeig '{config['file']}'.")
    except FileNotFoundError:
        content_placeholder.empty()
        st.error(f"L'arxiu '{config['file']}' no existeix.")


def run_live_mode():
    """
    Versió corregida que va directament a la pantalla de selecció de comarques.
    """
    selection = st.session_state.get('province_selected')

    if selection and selection.startswith('seguiment_'):
        # Si ja s'ha seleccionat una comarca (ex: 'seguiment_destacable'), mostra l'anàlisi
        run_single_sounding_mode(selection)
    else:
        # Si no s'ha seleccionat res, mostra la pantalla per triar comarca
        show_seguiment_selection_screen()

# =================================================================================
# === NOU MODE MANUAL (CORREGIT) ==================================================
# =================================================================================

@st.experimental_dialog(get_text("manual_dialog_title"))
def get_elevation_dialog():
    """
    Dialog per al mode manual. Demana elevació i orografia, i mostra
    una anàlisi en viu de l'activació orogràfica. És un procés d'un sol pas.
    """
    st.markdown(f'##### {get_text("manual_dialog_header")}')
    st.write(get_text("manual_dialog_desc"))

    elevation_m = st.number_input(
        get_text("manual_dialog_elevation_label"),
        min_value=0, max_value=4000, value=st.session_state.get('dialog_elevation_val', 0), step=10,
        help="Aquesta serà la base del sondeig."
    )
    
    orography_height_m = st.number_input(
        get_text("manual_dialog_orography_label"),
        min_value=0, max_value=4000, value=st.session_state.get('dialog_orography_val', 0), step=50,
        help="Introdueix l'alçada mitjana de les muntanyes properes."
    )
    
    st.session_state.dialog_elevation_val = elevation_m
    st.session_state.dialog_orography_val = orography_height_m

    st.markdown("---")

    sounding_text = st.session_state.get("manual_sounding_text", "")
    lines = sounding_text.splitlines()
    data = process_sounding_block(lines)

    if not data:
        st.error(get_text("manual_dialog_error"))
    
    # ... (La resta de la lògica de la funció es manté igual)
    
    if st.button(get_text("manual_dialog_button"), type="primary", use_container_width=True):
        st.session_state.manual_elevation = st.session_state.dialog_elevation_val
        st.session_state.manual_orography = st.session_state.dialog_orography_val
        st.session_state.analysis_requested = True # Indica que s'ha de mostrar l'anàlisi
        for key in ['dialog_elevation_val', 'dialog_orography_val']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def run_manual_mode():
    """Versió neta sense sidebar propi i corregida."""
    st.title(get_text("manual_mode_page_title"))
    st.markdown(get_text("manual_mode_page_desc"))
    
    st.session_state.manual_sounding_text = st.text_area(
        get_text("manual_textarea_label"), 
        height=300, 
        placeholder=get_text("manual_textarea_placeholder"),
        key="manual_sounding_input"
    )
    
    if st.button(get_text("manual_analyze_button"), use_container_width=True, type="primary"):
        if st.session_state.manual_sounding_text:
            if 'manual_elevation' in st.session_state: del st.session_state.manual_elevation
            if 'manual_orography' in st.session_state: del st.session_state.manual_orography
            get_elevation_dialog()
        else:
            st.warning("Per favor, enganxa les dades del sondeig a la caixa de text abans d'analitzar.")

    if st.session_state.get('analysis_requested', False):
        placeholder = st.empty()
        with placeholder.container():
            show_loading_animation("Processant Sondeig")
            time.sleep(1)
        
        elevation_m = st.session_state.manual_elevation
        orography_m = st.session_state.manual_orography
        sounding_text = st.session_state.manual_sounding_text
        
        sfc_pressure = mpcalc.height_to_pressure_std(elevation_m * units.m).to('hPa').m
        lines = sounding_text.splitlines()
        
        temp_data = process_sounding_block(lines)
        if temp_data:
            p_orig, t_orig, td_orig = temp_data['p_levels'].m, temp_data['t_initial'].m, temp_data['td_initial'].m
            t_interp = interp1d(p_orig, t_orig, bounds_error=False, fill_value='extrapolate')(sfc_pressure)
            td_interp = interp1d(p_orig, td_orig, bounds_error=False, fill_value='extrapolate')(sfc_pressure)
            sfc_line = f"SFC    {sfc_pressure:.1f}    {t_interp:.1f}    N/A    {td_interp:.1f}    N/A    0/0"
            
            first_data_line_index = next((i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit()), -1)
            
            if first_data_line_index != -1: lines.insert(first_data_line_index, sfc_line)
            else: lines.append(sfc_line)

        data = process_sounding_block(lines)
        
        placeholder.empty()
        if data:
            st.success(f"Sondeig processat correctament amb una elevació de {elevation_m} m ({sfc_pressure:.1f} hPa) i orografia de {orography_m} m.")
            st.markdown("---")
            show_full_analysis_view(
                p=data['p_levels'], t=data['t_initial'], td=data['td_initial'], 
                ws=data['wind_speed_kmh'].to('m/s'), wd=data['wind_dir_deg'], 
                obs_time=data.get('observation_time', "Sondeig Manual"),
                orography_preset=orography_m
            )
        else:
            st.error("No s'ha pogut processar el text. Assegura't que el format és correcte.")
        
        st.session_state.analysis_requested = False

# =================================================================================
# === LABORATORI-TUTORIAL =========================================================
# =================================================================================

def get_tutorial_data():
    return {
        'supercel': [
            # ... (les traduccions per a supercèl·lula ja estaven bé) ...
        ],
        'aiguaneu': [
            {'action_id': 'conceptual', 'title': get_text("tutorial_sleet_step1_title"), 'instruction': get_text("tutorial_sleet_step1_instr"), 'button_label': get_text("tutorial_sleet_step1_button"), 'explanation': get_text("tutorial_sleet_step1_expl")},
            {'action_id': 'conceptual', 'title': get_text("tutorial_sleet_step2_title"), 'instruction': get_text("tutorial_sleet_step2_instr"), 'button_label': get_text("tutorial_sleet_step2_button"), 'explanation': get_text("tutorial_sleet_step2_expl")},
            {'action_id': 'conceptual', 'title': get_text("tutorial_sleet_step3_title"), 'instruction': get_text("tutorial_sleet_step3_instr"), 'button_label': get_text("tutorial_sleet_step3_button"), 'explanation': get_text("tutorial_sleet_step3_expl")},
            {'action_id': 'conceptual', 'title': get_text("tutorial_sleet_step4_title"), 'instruction': get_text("tutorial_sleet_step4_instr"), 'button_label': get_text("tutorial_sleet_step4_button"), 'explanation': get_text("tutorial_sleet_step4_expl")},
        ]
    }

def start_tutorial(scenario_name):
    st.session_state.sandbox_mode = 'tutorial'
    st.session_state.tutorial_active = True
    st.session_state.tutorial_scenario = scenario_name
    st.session_state.tutorial_step = 0
    
    if scenario_name == 'aiguaneu':
        profile_data = create_wintry_mix_profile()
    else: # 'supercel'
        profile_data = st.session_state.sandbox_original_data

    st.session_state.sandbox_p_levels = profile_data['p_levels'].copy()
    st.session_state.sandbox_t_profile = profile_data['t_initial'].copy()
    st.session_state.sandbox_td_profile = profile_data['td_initial'].copy()
    st.session_state.sandbox_ws = profile_data['wind_speed_kmh'].to('m/s').copy()
    st.session_state.sandbox_wd = profile_data['wind_dir_deg'].copy()

def exit_tutorial():
    st.session_state.sandbox_mode = 'free'
    st.session_state.tutorial_active = False
    if 'tutorial_scenario' in st.session_state: del st.session_state['tutorial_scenario']
    if 'tutorial_step' in st.session_state: del st.session_state['tutorial_step']

def apply_profile_modification(action):
    t = st.session_state.sandbox_t_profile.m
    td = st.session_state.sandbox_td_profile.m
    p = st.session_state.sandbox_p_levels.m
    ws = st.session_state.sandbox_ws.to('m/s').m
    wd = st.session_state.sandbox_wd.m

    sfc_mask = p >= 950
    low_mask = (p < 950) & (p >= 800)
    low_mid_mask = (p < 800) & (p >= 600)
    high_mid_mask = (p < 600) & (p >= 400)
    high_mask = p < 400

    mask = np.full_like(p, False, dtype=bool)
    if 'sfc' in action: mask = sfc_mask
    elif 'low' in action and 'mid' not in action: mask = low_mask
    elif 'low_mid' in action: mask = low_mid_mask
    elif 'high_mid' in action: mask = high_mid_mask
    elif 'high' in action and 'mid' not in action: mask = high_mask
    
    if 'warm' in action: t[mask] += 2.0
    elif 'cool' in action: t[mask] -= 2.0
    elif 'moisten' in action:
        if action == 'moisten_low_tutorial':
            td[sfc_mask] += 4.0
            td[low_mask] += 4.0
        else:
            td[mask] = np.minimum(t[mask] - 1.0, td[mask] + 2.0)
    elif 'dry' in action: td[mask] -= 2.0
    
    if action == 'add_inversion':
        t[low_mask] += 3.0
    
    if 'shear' in action:
        if 'low' in action: mask_shear = low_mask | sfc_mask
        elif 'mid' in action: mask_shear = low_mid_mask | high_mid_mask
        elif 'high' in action: mask_shear = high_mask
        else: mask_shear = np.full_like(p, False, dtype=bool)

        if '_N' in action: base_dir = 0
        elif '_E' in action: base_dir = 90
        elif '_S' in action: base_dir = 180
        elif '_W' in action: base_dir = 270
        else: base_dir = wd[mask_shear].mean() if np.any(mask_shear) else 180

        num_points = np.sum(mask_shear)
        if num_points > 0:
            ws[mask_shear] += np.linspace(5, 20, num_points)
            wd[mask_shear] = np.linspace(wd[mask_shear][0], base_dir, num_points) % 360

    ws = np.clip(ws, 0, 120)
    st.session_state.sandbox_ws = ws * units('m/s')
    st.session_state.sandbox_wd = wd * units.degrees
    
    td = np.minimum(t, td)
    st.session_state.sandbox_t_profile = t * units.degC
    st.session_state.sandbox_td_profile = td * units.degC

def show_tutorial_interface():
    tutorials = get_tutorial_data()
    scenario = st.session_state.tutorial_scenario
    step_index = st.session_state.tutorial_step
    steps = tutorials[scenario]
    
    st.title(get_text("tutorial_lab_title"))
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown(f"### {get_text('tutorial_title', scenario=scenario.replace('_', ' ').title())}")
            st.markdown("---")
            if step_index >= len(steps):
                st.success(get_text("tutorial_congrats"))
                if st.button(get_text("tutorial_finish_button"), use_container_width=True, type="primary"):
                    exit_tutorial(); st.rerun()
            else:
                current_step = steps[step_index]
                st.markdown(f"#### {current_step['title']}")
                with st.container(border=True):
                    st.markdown(current_step['instruction'])
                    if st.button(current_step['button_label'], key=f"tut_action_{step_index}", use_container_width=True, type="primary"):
                        if current_step['action_id'] != 'conceptual': apply_profile_modification(current_step['action_id'])
                        st.session_state.tutorial_step += 1
                        st.rerun()
                st.markdown(f"*{current_step['explanation']}*")
        with col2:
            chat_log, _ = generate_tutorial_analysis(scenario, step_index)
            # ... (la resta del codi del xat es manté igual) ...
        st.markdown("---")
        if st.button(get_text("tutorial_abandon_button"), use_container_width=True):
            exit_tutorial(); st.rerun()

def show_sandbox_selection_screen():
    st.title(get_text("sandbox_welcome_title"))
    st.markdown(get_text("sandbox_welcome_desc"))
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="mode-card"><h4>{get_text("sandbox_tutorial_supercell_title")}</h4><p>{get_text("sandbox_tutorial_supercell_desc")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("sandbox_tutorial_supercell_button"), use_container_width=True): 
            start_tutorial('supercel'); st.rerun()
    with c2:
        st.markdown(f"""<div class="mode-card"><h4>{get_text("sandbox_tutorial_sleet_title")}</h4><p>{get_text("sandbox_tutorial_sleet_desc")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("sandbox_tutorial_sleet_button"), use_container_width=True): 
            start_tutorial('aiguaneu'); st.rerun()
    with c3:
        st.markdown(f"""<div class="mode-card"><h4>{get_text("sandbox_freemode_title")}</h4><p>{get_text("sandbox_freemode_desc")}</p></div>""", unsafe_allow_html=True)
        if st.button(get_text("sandbox_freemode_button"), use_container_width=True, type="primary"):
            st.session_state.sandbox_mode = 'free'; st.rerun()
        
def run_sandbox_mode():
    """Versió neta sense sidebar propi."""
    if 'sandbox_mode' not in st.session_state:
        st.session_state.sandbox_mode = 'selection'

    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation(get_text("loading_message"))
        time.sleep(1)

    if 'sandbox_initialized' not in st.session_state:
        soundings = parse_all_soundings("sondeigproves.txt")
        if not soundings:
            placeholder.empty()
            st.error("No s'ha trobat 'sondeigproves.txt'. Assegura't que el fitxer existeix.")
            return
        st.session_state.sandbox_original_data = soundings[0]
        data = st.session_state.sandbox_original_data
        st.session_state.sandbox_p_levels = data['p_levels'].copy()
        st.session_state.sandbox_t_profile = data['t_initial'].copy()
        st.session_state.sandbox_td_profile = data['td_initial'].copy()
        st.session_state.sandbox_ws = data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = data['wind_dir_deg'].copy()
        st.session_state.sandbox_initialized = True
        st.session_state.convergence_active = False

    placeholder.empty()
    if st.session_state.sandbox_mode == 'selection':
        show_sandbox_selection_screen()
    elif st.session_state.sandbox_mode == 'tutorial':
        show_tutorial_interface()
    elif st.session_state.sandbox_mode == 'free':
        st.title("🧪 Laboratori de Sondejos - Mode Lliure")
        show_full_analysis_view(p=st.session_state.sandbox_p_levels, t=st.session_state.sandbox_t_profile,
                               td=st.session_state.sandbox_td_profile, ws=st.session_state.sandbox_ws,
                               wd=st.session_state.sandbox_wd, obs_time="Sondeig de Prova - Mode Laboratori",
                               is_sandbox_mode=True)

def setup_sandbox_sidebar():
    """Dibuixa els controls del laboratori a la barra lateral."""
    st.header(get_text("sandbox_toolbox_header"))
    st.subheader(get_text("sandbox_thermo_header"))
    
    st.markdown(f'**{get_text("sandbox_layer_sfc")}**')
    c1,c2 = st.columns(2);c1.button("☀️", on_click=apply_profile_modification, args=('warm_sfc',), use_container_width=True, key='warm_sfc'); c2.button("❄️", on_click=apply_profile_modification, args=('cool_sfc',), use_container_width=True, key='cool_sfc'); c1.button("💧", on_click=apply_profile_modification, args=('moisten_sfc',), use_container_width=True, key='moisten_sfc'); c2.button("💨", on_click=apply_profile_modification, args=('dry_sfc',), use_container_width=True, key='dry_sfc')
    
    st.markdown(f'**{get_text("sandbox_layer_low")}**')
    c1,c2 = st.columns(2);c1.button("☀️", on_click=apply_profile_modification, args=('warm_low',), use_container_width=True, key='warm_low'); c2.button("❄️", on_click=apply_profile_modification, args=('cool_low',), use_container_width=True, key='cool_low'); c1.button("💧", on_click=apply_profile_modification, args=('moisten_low',), use_container_width=True, key='moisten_low'); c2.button("💨", on_click=apply_profile_modification, args=('dry_low',), use_container_width=True, key='dry_low')

    st.markdown(f'**{get_text("sandbox_layer_lowmid")}**')
    c1,c2 = st.columns(2);c1.button("☀️", on_click=apply_profile_modification, args=('warm_low_mid',), use_container_width=True, key='warm_low_mid'); c2.button("❄️", on_click=apply_profile_modification, args=('cool_low_mid',), use_container_width=True, key='cool_low_mid'); c1.button("💧", on_click=apply_profile_modification, args=('moisten_low_mid',), use_container_width=True, key='moisten_low_mid'); c2.button("💨", on_click=apply_profile_modification, args=('dry_low_mid',), use_container_width=True, key='dry_low_mid')

    st.markdown("---"); st.subheader(get_text("sandbox_shear_header"))
    
    st.markdown(f'**{get_text("sandbox_shear_low")}**')
    c1,c2,c3,c4 = st.columns(4);c1.button("N", on_click=apply_profile_modification, args=('add_shear_low_N',), use_container_width=True, key='shear_low_N');c2.button("E", on_click=apply_profile_modification, args=('add_shear_low_E',), use_container_width=True, key='shear_low_E');c3.button("S", on_click=apply_profile_modification, args=('add_shear_low_S',), use_container_width=True, key='shear_low_S');c4.button("W", on_click=apply_profile_modification, args=('add_shear_low_W',), use_container_width=True, key='shear_low_W')

    st.markdown(f'**{get_text("sandbox_shear_mid")}**')
    c1,c2,c3,c4 = st.columns(4);c1.button("N", on_click=apply_profile_modification, args=('add_shear_mid_N',), use_container_width=True, key='shear_mid_N');c2.button("E", on_click=apply_profile_modification, args=('add_shear_mid_E',), use_container_width=True, key='shear_mid_E');c3.button("S", on_click=apply_profile_modification, args=('add_shear_mid_S',), use_container_width=True, key='shear_mid_S');c4.button("W", on_click=apply_profile_modification, args=('add_shear_mid_W',), use_container_width=True, key='shear_mid_W')

    st.markdown("---")
    
    def reset_wind_profile():
        st.session_state.sandbox_ws = st.session_state.sandbox_original_data['wind_speed_kmh'].to('m/s')
        st.session_state.sandbox_wd = st.session_state.sandbox_original_data['wind_dir_deg'].copy()
    
    if st.button(get_text("sandbox_reset_wind"), on_click=reset_wind_profile, use_container_width=True, key='reset_wind'):
        pass
    
    if st.button(get_text("sandbox_reset_all"), use_container_width=True, key='reset_all'):
        data = st.session_state.sandbox_original_data
        st.session_state.sandbox_p_levels, st.session_state.sandbox_t_profile, st.session_state.sandbox_td_profile = data['p_levels'].copy(), data['t_initial'].copy(), data['td_initial'].copy()
        reset_wind_profile()
        if st.session_state.get('tutorial_active', False): exit_tutorial()
        if 'convergence_active' in st.session_state: st.session_state.convergence_active = False
        st.rerun()

def setup_live_sidebar():
    """Dibuixa els controls del mode d'avisos."""
    if st.session_state.get('province_selected') not in [None, 'seguiment_menu']:
        if st.button("⬅️ Tornar a la selecció", use_container_width=True, key='live_back_to_selection'):
            st.session_state.province_selected = 'seguiment_menu'
            st.rerun()
# =========================================================================
# === PUNT D'ENTRADA DE L'APLICACIÓ (ESTRUCTURA FINAL) ====================
# =========================================================================

if __name__ == '__main__':
    # --- 1. Inicialització de l'estat (només si no existeix) ---
    if 'language' not in st.session_state:
        st.session_state.language = 'ca'
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'welcome'

    # --- 2. Configuració de la pàgina ---
    st.set_page_config(layout="wide", page_title=get_text("app_title"))

    # --- 3. Dibuix de la barra lateral (únic lloc) ---
    with st.sidebar:
        st.markdown(f"**{get_text('lang_selector_label')}**")
        c1, c2, c3 = st.columns(3)
        with c1:
              st.button("CAT", on_click=set_language, args=('ca',), use_container_width=True, key='lang_ca')
        with c2:
              st.button("ESP", on_click=set_language, args=('es',), use_container_width=True, key='lang_es')
        with c3:
             st.button("ENG", on_click=set_language, args=('en',), use_container_width=True, key='lang_en')
        st.divider()


        

        # Controls condicionals segons el mode
        if st.session_state.app_mode == 'sandbox':
            setup_sandbox_sidebar() # Mostra els controls del laboratori
        elif st.session_state.app_mode == 'live':
            setup_live_sidebar() # Mostra els controls del mode en directe
        
        # Botó global per tornar a l'inici (menys a la pantalla de benvinguda)
        if st.session_state.app_mode != 'welcome':
            st.divider()
            if st.button(get_text("back_to_start_button"), use_container_width=True, key='global_back_to_start'):
                # Esborrem estats específics de cada mode per evitar errors
                keys_to_clear = [
                    'province_selected', 'manual_sounding_text', 'manual_elevation', 
                    'manual_orography', 'analysis_requested', 'sandbox_mode', 
                    'tutorial_active', 'convergence_active'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.app_mode = 'welcome'
                st.rerun()

    # --- 4. Lògica principal per mostrar el contingut de la pàgina ---
    if st.session_state.app_mode == 'welcome':
        show_welcome_screen()
    elif st.session_state.app_mode == 'live':
        run_live_mode()
    elif st.session_state.app_mode == 'sandbox':
        run_sandbox_mode()
    elif st.session_state.app_mode == 'manual':
        run_manual_mode()
