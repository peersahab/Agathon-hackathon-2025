



import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

def load_meteo_data(file_path, chunk_size=50000):
    """Load meteorological data in chunks."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks)

def preprocess_snotel_data(base_path: str):
    """Preprocess SNOTEL dataset with location-first organization."""
    base_path = Path(base_path)
    meteo_path = base_path / 'meteorological_data'
    swe_path = base_path / 'swe_data'

    # 1. Load station information
    stations = pd.read_csv(swe_path / 'Station_Info.csv')

    # 2. Load SWE dataset and sort by location first
    swe_chunks = []
    for chunk in pd.read_csv(swe_path / 'SWE_values_all.csv', chunksize=50000):
        chunk['Date'] = pd.to_datetime(chunk['Date'])
        swe_chunks.append(chunk)
    swe = pd.concat(swe_chunks)
    swe = swe.sort_values(['Latitude', 'Longitude', 'Date'])

    # 3. Load and prepare meteorological data
    meteo = load_meteo_data(meteo_path / 'Modified_Output_precip.csv')
    meteo['date'] = pd.to_datetime(meteo['date'])
    meteo = meteo.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude',
        'variable_value': 'precip'
    })

    # Get grid points
    grid_points = meteo[['latitude', 'longitude']].drop_duplicates()

    # Create station to grid mapping
    station_mappings = []
    for _, station in stations.iterrows():
        station_coord = np.array([[station['Latitude'], station['Longitude']]])
        grid_coords = grid_points[['latitude', 'longitude']].values
        distances = cdist(station_coord, grid_coords)
        nearest_idx = np.argmin(distances)
        nearest_grid = grid_points.iloc[nearest_idx]

        station_mappings.append({
            'Station': station['Station'],
            'station_lat': station['Latitude'],
            'station_lon': station['Longitude'],
            'grid_lat': nearest_grid['latitude'],
            'grid_lon': nearest_grid['longitude'],
            'elevation': station['Elevation'],
            'southness': station['Southness']
        })

    station_map_df = pd.DataFrame(station_mappings)

    # Load remaining meteorological variables
    meteo_vars = ['tmin', 'tmax', 'SPH', 'SRAD', 'Rmax', 'Rmin', 'windspeed']
    for var in meteo_vars:
        temp_df = load_meteo_data(meteo_path / f'Modified_Output_{var}.csv')
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df = temp_df.rename(columns={
            'lat': 'latitude',
            'lon': 'longitude',
            'variable_value': var
        })
        meteo = pd.merge(
            meteo,
            temp_df[['date', 'latitude', 'longitude', var]],
            on=['date', 'latitude', 'longitude'],
            how='outer'
        )
        del temp_df

    # Process each station separately
    final_chunks = []
    unique_stations = swe[['Latitude', 'Longitude']].drop_duplicates()

    for _, station in unique_stations.iterrows():
        # Get station's SWE data
        station_swe = swe[
            (swe['Latitude'] == station['Latitude']) &
            (swe['Longitude'] == station['Longitude'])
        ].copy()

        # Get station mapping info
        station_info = station_map_df[
            (station_map_df['station_lat'] == station['Latitude']) &
            (station_map_df['station_lon'] == station['Longitude'])
        ].iloc[0]

        # Get meteorological data for this station's grid point
        station_meteo = meteo[
            (meteo['latitude'] == station_info['grid_lat']) &
            (meteo['longitude'] == station_info['grid_lon'])
        ].copy()

        # Merge data for this station
        station_result = pd.merge(
            station_swe,
            station_meteo,
            left_on='Date',
            right_on='date',
            how='left'
        )

        # Add static features
        station_result['elevation'] = station_info['elevation']
        station_result['southness'] = station_info['southness']

        # Clean up columns
        station_result = station_result.rename(columns={
            'Date': 'date',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        })

        final_columns = [
            'date', 'latitude', 'longitude', 'SWE', 'precip',
            'tmin', 'tmax', 'SPH', 'SRAD', 'Rmax', 'Rmin',
            'windspeed', 'elevation', 'southness'
        ]
        station_result = station_result[final_columns]

        # Ensure date is datetime
        station_result['date'] = pd.to_datetime(station_result['date'])

        # Filter winter season
        mask = (station_result['date'].dt.month == 12) | (station_result['date'].dt.month <= 5)
        station_result = station_result[mask]

        # Sort by date
        station_result = station_result.sort_values('date')

        final_chunks.append(station_result)
        del station_result

    # Combine all stations' data
    result = pd.concat(final_chunks, ignore_index=True)

    # Ensure final sort order (by location, then date)
    result = result.sort_values(['latitude', 'longitude', 'date'])
    
    return result

# Example usage:
processed_data = preprocess_snotel_data('/Users/simarjeetss529/Desktop/hackathon/input_data')
processed_data.to_csv('processed_snotel_data.csv', index=False)