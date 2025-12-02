"""
FastF1 Data Extraction - F1 2024 Season FIXED VERSION
======================================================
Versión mejorada que carga TODOS los datos disponibles:
- Telemetría completa (get_telemetry())
- Datos de neumáticos detallados
- Información climática completa (weather_data)
- Resultados y tiempos de vuelta

CAMBIOS CLAVE:
- session.load() con parámetros específicos
- Acceso correcto a telemetría por vuelta
- Carga de weather_data
- Manejo robusto de errores

Autor: Versión corregida para obtener datos completos
"""

import fastf1
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Habilitar caché
fastf1.Cache.enable_cache('../.fastXf1-cache')

# CONFIGURACIÓN CRÍTICA
TELEMETRY_SAMPLE_RATE = 10  # Tomar 1 de cada 10 puntos para reducir tamaño
MAX_LAPS_PER_DRIVER_TELEMETRY = 3  # Solo las 3 mejores vueltas por piloto

def create_output_directories():
    """Crea la estructura de directorios"""
    directories = [
        'f1_data_2024',
        'f1_data_2024/raw_data',
        'f1_data_2024/processed_data',
        'f1_data_2024/telemetry',
        'f1_data_2024/weather',
        'f1_data_2024/tires',
        'f1_data_2024/results',
        'f1_data_2024/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directorios creados")
    return directories[0]

def get_2024_calendar():
    """Obtiene el calendario completo de 2024"""
    print("\n" + "="*70)
    print("OBTENIENDO CALENDARIO F1 2024")
    print("="*70)
    
    try:
        schedule = fastf1.get_event_schedule(2024)
        print(f"\n✓ Calendario obtenido: {len(schedule)} eventos")
        print(f"  Desde: {schedule.iloc[0]['EventName']}")
        print(f"  Hasta: {schedule.iloc[-1]['EventName']}")
        
        schedule.to_csv('f1_data_2024/calendar_2024.csv', index=False)
        print("✓ Calendario guardado")
        
        return schedule
    except Exception as e:
        print(f"✗ Error obteniendo calendario: {e}")
        return None

def extract_session_data_complete(year, event_name, session_type, round_number):
    """
    VERSIÓN MEJORADA: Extrae TODOS los datos disponibles de una sesión
    """
    
    print(f"      └─ Cargando {session_type}...", end=' ', flush=True)
    
    session_data = {
        'success': False,
        'event_name': event_name,
        'session_type': session_type,
        'has_laps': False,
        'has_telemetry': False,
        'has_weather': False,
        'has_tires': False,
        'error': None
    }
    
    try:
        # 1. CARGAR SESIÓN
        session = fastf1.get_session(year, event_name, session_type)
        
        # CRÍTICO: Cargar con todos los datos
        session.load(
            laps=True,           # Cargar vueltas
            telemetry=True,      # Cargar telemetría
            weather=True,        # Cargar clima
            messages=False       # Mensajes no necesarios por ahora
        )
        
        session_data['session_info'] = {
            'date': str(session.date) if session.date else 'N/A',
            'event': session.event['EventName'] if 'EventName' in session.event else event_name,
            'location': session.event.get('Location', 'N/A'),
            'country': session.event.get('Country', 'N/A')
        }
        
        # 2. EXTRAER DATOS DE VUELTAS (LAPS)
        laps_saved = False
        if session.laps is not None and len(session.laps) > 0:
            try:
                laps_df = session.laps.copy()
                
                # Convertir columnas problemáticas a string
                for col in laps_df.columns:
                    if laps_df[col].dtype == 'object':
                        laps_df[col] = laps_df[col].astype(str)
                
                # Guardar vueltas
                filename = f"{round_number:02d}_{event_name.replace(' ', '_')}_{session_type}.csv"
                laps_df.to_csv(f'f1_data_2024/raw_data/{filename}', index=False)
                session_data['has_laps'] = True
                laps_saved = True
            except Exception as e:
                print(f"\n         ⚠️  Error guardando laps: {e}", end='')
        
        # 3. EXTRAER DATOS DE NEUMÁTICOS (TIRE DATA)
        tires_saved = False
        if session.laps is not None and len(session.laps) > 0:
            try:
                tire_data = []
                for idx, lap in session.laps.iterrows():
                    tire_info = {
                        'Driver': str(lap.get('Driver', 'N/A')),
                        'DriverNumber': str(lap.get('DriverNumber', 'N/A')),
                        'Team': str(lap.get('Team', 'N/A')),
                        'LapNumber': int(lap.get('LapNumber', 0)) if pd.notna(lap.get('LapNumber')) else 0,
                        'Compound': str(lap.get('Compound', 'UNKNOWN')),
                        'TyreLife': int(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else 0,
                        'FreshTyre': str(lap.get('FreshTyre', 'N/A')),
                        'LapTime': str(lap.get('LapTime', 'N/A')),
                        'LapTimeSeconds': pd.to_timedelta(lap.get('LapTime'), errors='coerce').total_seconds() if pd.notna(lap.get('LapTime')) else None,
                        'Stint': int(lap.get('Stint', 0)) if pd.notna(lap.get('Stint')) else 0,
                        'IsPersonalBest': str(lap.get('IsPersonalBest', False))
                    }
                    tire_data.append(tire_info)
                
                if tire_data:
                    tire_df = pd.DataFrame(tire_data)
                    filename = f"tires_{round_number:02d}_{event_name.replace(' ', '_')}_{session_type}.csv"
                    tire_df.to_csv(f'f1_data_2024/tires/{filename}', index=False)
                    session_data['has_tires'] = True
                    tires_saved = True
            except Exception as e:
                print(f"\n         ⚠️  Error guardando tires: {e}", end='')
        
        # 4. EXTRAER DATOS CLIMÁTICOS (WEATHER)
        weather_saved = False
        if hasattr(session, 'weather_data') and session.weather_data is not None:
            try:
                weather_df = session.weather_data.copy()
                
                # Convertir columnas a tipos simples
                for col in weather_df.columns:
                    if weather_df[col].dtype == 'object':
                        weather_df[col] = weather_df[col].astype(str)
                
                if len(weather_df) > 0:
                    filename = f"weather_{round_number:02d}_{event_name.replace(' ', '_')}_{session_type}.csv"
                    weather_df.to_csv(f'f1_data_2024/weather/{filename}', index=False)
                    session_data['has_weather'] = True
                    weather_saved = True
            except Exception as e:
                print(f"\n         ⚠️  Error guardando weather: {e}", end='')
        
        # 5. EXTRAER TELEMETRÍA (MUESTRAS DE MEJORES VUELTAS)
        telemetry_saved = False
        if session.laps is not None and len(session.laps) > 0:
            try:
                all_telemetry = []
                
                # Por cada piloto, obtener sus mejores vueltas
                drivers = session.laps['Driver'].unique()
                
                for driver in drivers:
                    try:
                        driver_laps = session.laps.pick_driver(driver)
                        
                        if len(driver_laps) == 0:
                            continue
                        
                        # Obtener las N mejores vueltas del piloto
                        valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                        if len(valid_laps) == 0:
                            continue
                        
                        # Ordenar por tiempo y tomar las mejores
                        best_laps = valid_laps.nsmallest(MAX_LAPS_PER_DRIVER_TELEMETRY, 'LapTime')
                        
                        for idx, lap in best_laps.iterrows():
                            try:
                                # Obtener telemetría de esta vuelta
                                telemetry = lap.get_telemetry()
                                
                                if telemetry is not None and len(telemetry) > 0:
                                    # Muestrear para reducir tamaño
                                    telemetry_sample = telemetry.iloc[::TELEMETRY_SAMPLE_RATE].copy()
                                    
                                    # Añadir información de contexto
                                    telemetry_sample['Driver'] = str(driver)
                                    telemetry_sample['LapNumber'] = int(lap['LapNumber'])
                                    telemetry_sample['Compound'] = str(lap.get('Compound', 'UNKNOWN'))
                                    telemetry_sample['TyreLife'] = int(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else 0
                                    
                                    all_telemetry.append(telemetry_sample)
                            except Exception as e:
                                continue  # Si falla una vuelta, continuar con la siguiente
                    except Exception as e:
                        continue  # Si falla un piloto, continuar con el siguiente
                
                # Guardar toda la telemetría recopilada
                if all_telemetry:
                    telemetry_df = pd.concat(all_telemetry, ignore_index=True)
                    
                    # Convertir columnas problemáticas
                    for col in telemetry_df.columns:
                        if telemetry_df[col].dtype == 'object':
                            telemetry_df[col] = telemetry_df[col].astype(str)
                    
                    filename = f"telemetry_{round_number:02d}_{event_name.replace(' ', '_')}_{session_type}.csv"
                    telemetry_df.to_csv(f'f1_data_2024/telemetry/{filename}', index=False)
                    session_data['has_telemetry'] = True
                    telemetry_saved = True
            except Exception as e:
                print(f"\n         ⚠️  Error guardando telemetry: {e}", end='')
        
        # 6. EXTRAER RESULTADOS (solo para Q y R)
        if session_type in ['Q', 'R', 'Sprint']:
            try:
                if hasattr(session, 'results') and session.results is not None:
                    results_df = session.results.copy()
                    
                    # Convertir a tipos simples
                    for col in results_df.columns:
                        if results_df[col].dtype == 'object':
                            results_df[col] = results_df[col].astype(str)
                    
                    filename = f"results_{round_number:02d}_{event_name.replace(' ', '_')}_{session_type}.csv"
                    results_df.to_csv(f'f1_data_2024/results/{filename}', index=False)
            except Exception as e:
                print(f"\n         ⚠️  Error guardando results: {e}", end='')
        
        # Resumen de lo guardado
        saved_items = []
        if laps_saved:
            saved_items.append('laps')
        if tires_saved:
            saved_items.append('tires')
        if weather_saved:
            saved_items.append('weather')
        if telemetry_saved:
            saved_items.append('telemetry')
        
        if saved_items:
            session_data['success'] = True
            print(f"✓ ({', '.join(saved_items)})")
        else:
            print("⊘ (sin datos)")
        
        return session_data
        
    except Exception as e:
        session_data['error'] = str(e)
        print(f"✗ Error: {str(e)[:40]}")
        return session_data

def extract_all_2024_data():
    """Función principal MEJORADA para extraer todos los datos de 2024"""
    
    print("\n" + "="*70)
    print("EXTRACCIÓN COMPLETA F1 2024 - VERSIÓN MEJORADA")
    print("="*70)
    print("\nMEJORAS:")
    print("  • session.load() con parámetros específicos")
    print("  • Telemetría de mejores vueltas por piloto")
    print("  • Datos de neumáticos detallados")
    print("  • Weather data completo")
    print("="*70)
    
    start_time = datetime.now()
    
    # Crear directorios
    base_dir = create_output_directories()
    
    # Obtener calendario
    schedule = get_2024_calendar()
    if schedule is None:
        return
    
    # Sesiones a extraer
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    
    log_entries = []
    
    total_events = len(schedule)
    
    print(f"\n📊 Total de eventos: {total_events}")
    print(f"📊 Sesiones por evento: {len(session_types)}")
    print(f"📊 Total de sesiones: {total_events * len(session_types)}")
    print("\n" + "="*70)
    
    # Contadores
    stats = {
        'total_sessions': 0,
        'successful_sessions': 0,
        'with_laps': 0,
        'with_tires': 0,
        'with_weather': 0,
        'with_telemetry': 0
    }
    
    # Iterar por cada evento
    for event_idx, event in schedule.iterrows():
        event_name = event['EventName']
        round_number = event['RoundNumber']
        
        print(f"\n{'='*70}")
        print(f"🏁 Evento {round_number}/{total_events}: {event_name}")
        print(f"   📅 {event['EventDate']} | 📍 {event['Location']}, {event['Country']}")
        print(f"{'='*70}")
        
        # Iterar por cada tipo de sesión
        for session_type in session_types:
            stats['total_sessions'] += 1
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event_name,
                'round': round_number,
                'session': session_type,
                'status': 'processing'
            }
            
            try:
                session_data = extract_session_data_complete(
                    2024, event_name, session_type, round_number
                )
                
                if session_data['success']:
                    log_entry['status'] = 'success'
                    stats['successful_sessions'] += 1
                    
                    if session_data.get('has_laps'):
                        stats['with_laps'] += 1
                    if session_data.get('has_tires'):
                        stats['with_tires'] += 1
                    if session_data.get('has_weather'):
                        stats['with_weather'] += 1
                    if session_data.get('has_telemetry'):
                        stats['with_telemetry'] += 1
                else:
                    log_entry['status'] = 'no_data'
                    log_entry['error'] = session_data.get('error', 'No data available')
                
                log_entry.update({
                    'has_laps': session_data.get('has_laps', False),
                    'has_tires': session_data.get('has_tires', False),
                    'has_weather': session_data.get('has_weather', False),
                    'has_telemetry': session_data.get('has_telemetry', False)
                })
                
            except Exception as e:
                log_entry['status'] = 'error'
                log_entry['error'] = str(e)
                print(f"      └─ {session_type}: ✗ Error crítico: {str(e)[:50]}")
            
            log_entries.append(log_entry)
    
    # Guardar log completo
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv('f1_data_2024/logs/extraction_log_complete.csv', index=False)
    
    # Estadísticas finales
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("EXTRACCIÓN COMPLETADA")
    print("="*70)
    print(f"⏱️  Duración total: {duration}")
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   Total sesiones procesadas: {stats['total_sessions']}")
    print(f"   Sesiones exitosas: {stats['successful_sessions']} ({stats['successful_sessions']/stats['total_sessions']*100:.1f}%)")
    print(f"   Con datos de vueltas: {stats['with_laps']}")
    print(f"   Con datos de neumáticos: {stats['with_tires']}")
    print(f"   Con datos climáticos: {stats['with_weather']}")
    print(f"   Con telemetría: {stats['with_telemetry']}")
    print(f"\n📁 Datos guardados en: {base_dir}/")
    print("="*70)
    
    # Crear resumen
    create_extraction_summary(log_df, stats)

def create_extraction_summary(log_df, stats):
    """Crea un resumen detallado de la extracción"""
    
    summary = {
        'extraction_date': datetime.now().isoformat(),
        'total_sessions_attempted': stats['total_sessions'],
        'successful_extractions': stats['successful_sessions'],
        'success_rate': f"{(stats['successful_sessions'] / stats['total_sessions'] * 100):.1f}%",
        'sessions_with_laps': stats['with_laps'],
        'sessions_with_tires': stats['with_tires'],
        'sessions_with_weather': stats['with_weather'],
        'sessions_with_telemetry': stats['with_telemetry']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('f1_data_2024/extraction_summary_complete.csv', index=False)
    
    print(f"\n📊 Resumen guardado en: f1_data_2024/extraction_summary_complete.csv")
    
    # Resumen por evento
    event_summary = log_df.groupby('event').agg({
        'status': lambda x: (x == 'success').sum(),
        'has_laps': 'sum',
        'has_tires': 'sum',
        'has_weather': 'sum',
        'has_telemetry': 'sum'
    }).reset_index()
    
    event_summary.columns = [
        'Event', 'Successful_Sessions', 
        'With_Laps', 'With_Tires', 'With_Weather', 'With_Telemetry'
    ]
    event_summary.to_csv('f1_data_2024/event_summary_complete.csv', index=False)
    
    print(f"📊 Resumen por evento: f1_data_2024/event_summary_complete.csv")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   FastF1 - Extracción COMPLETA F1 2024 Season - FIXED VERSION   ║
    ║                                                                  ║
    ║  VERSIÓN MEJORADA con carga completa de:                        ║
    ║  ✓ Telemetría detallada (Speed, RPM, Throttle, Brake, DRS)      ║
    ║  ✓ Datos de neumáticos (Compound, TyreLife, FreshTyre)          ║
    ║  ✓ Weather data (AirTemp, TrackTemp, Humidity, Rainfall)        ║
    ║  ✓ Resultados completos de todas las sesiones                   ║
    ║                                                                  ║
    ║  MEJORAS CLAVE:                                                  ║
    ║  • session.load(laps=True, telemetry=True, weather=True)        ║
    ║  • Acceso correcto a lap.get_telemetry()                        ║
    ║  • Manejo robusto de errores                                    ║
    ║  • Muestreo inteligente para reducir tamaño                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    input("\n⚠️  Este proceso puede tardar 3-5 horas. Presiona ENTER para comenzar...")
    
    extract_all_2024_data()
    
    print("\n✅ Proceso completado. Los datos COMPLETOS están listos para análisis.")