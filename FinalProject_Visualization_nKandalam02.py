'''
Project: Final Project Data Visualization
Author: Sai Nayana Sahishna Kandalam
Description: This Program is to plot the graphs based on user input.
Revisions:
	00 - Performed selection based on the user input
	01 - Plotting the graph based on the selected records
'''
import csv
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, asin
from datetime import datetime

def coord2rad(location):
    """
    Convert coordinates from degrees to radians.
    Inputs:
        location : coordinates in degrees(tuple: Lat, Lng)
    Output:
        location : coordinates in radians(dict: Lat, Lng)
    """
    lat, lng = location
    return {'lat': radians(lat), 'lng': radians(lng)}

def havDist(loc1, loc2, unit='km'):
    """
    Calculate the Haversine distance between two sets of latitude
    and longitude coordinates.
    Inputs:
        loc1: coordinates in degrees(tuple: Lat, Lng)
        loc1: coordinates in degrees(tuple: Lat, Lng)
        unit: 'km' for kilometers, otherwise miles
    Returns:
        distance between two locations
    """
    # Set the radius of earth based on the units specified
    R = 6371 if unit == 'km' else 3959

    # Convert coordinates from tuples to dictionaries
    loc1 = coord2rad(loc1)
    loc2 = coord2rad(loc2)

    # Haversine formula
    dlat = loc2['lat'] - loc1['lat']
    dlon = loc2['lng'] - loc1['lng']
    a = sin(dlat / 2) ** 2 + cos(loc1['lat']) * \
        cos(loc2['lat']) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Calculate distance based on the specified unit
    distance = R * c

    return distance

def getCityData(file_path='worldcitiesF23.csv'):
    """
    Reads a CSV file containing world cities data and returns a dictionary containing the city information.

    Inputs:
        file_path (str): The path to the CSV file containing world cities data. Default is 'worldcitiesF23.csv'.

    Returns:
        dict: A dictionary where the keys are tuples representing latitude and longitude, and the values are
            dictionaries containing information about each city. The inner dictionaries include:
        'city_name' (str): The name of the city.
        'country' (str): The name of the country where the city is located.
        'population' (int): The population of the city.
    """
    city_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Extract relevant data from the row
            city_name = row['city']
            latitude = float(row['lat'])
            longitude = float(row['lng'])
            country = row['country']
            iso3 = row['iso3']
            population = int(row['pop']) if row['pop'] and row['pop'].isdigit() else 0 
            # Add additional data fields as needed

            # Create a tuple from latitude and longitude
            coordinates = (latitude, longitude)

            # Check if coordinates already exist in the dictionary
            if coordinates not in city_data:
                # If not, add a new entry with an inner dictionary
                city_data[coordinates] = {
                    'city_name': city_name,
                    'country': country,
                    'iso3':iso3,
                    'pop': population                    
                }
            # If coordinates exist, update the existing entry with new data
            

    return city_data

def findCities(target_location, city_data, radius, unit='km'):
    """
    Finds cities within a specified distance from a target location.

    Inputs:
        target_location (tuple): Target location coordinates (latitude, longitude) in degrees.
        city_data (dict): Dictionary containing city information with coordinates as keys.
        radius (float): Maximum distance from the target location to include cities.
        unit (str): Unit for distance measurement ('km' for kilometers, 'mi' for miles). Default is 'km'.

    Returns:
        list: List of dictionaries, where each dictionary represents a city within the specified distance.
            Each dictionary contains:
                'city' (str): The name of the city.
                'country' (str): The name of the country where the city is located.
                'pop' (int): The population of the city.
                'distance' (float): The distance from the target location.
    """

    # Extract latitude and longitude from the target location tuple
    target_lat, target_lon = target_location

    # List to store information about nearby cities
    nearby_cities = []

    # Loop through each city in the city_data dictionary
    for coordinates, city_info in city_data.items():
        # Extract latitude and longitude from the city's coordinates tuple
        city_lat, city_lon = coordinates

        # Calculate the distance between the target location and the current city
        distance = havDist((target_lat, target_lon), (city_lat, city_lon), unit)

        # Check if the distance is within the specified radius
        if distance <= radius:
            # Add city information to the result list
            result_entry = {
                'city': city_info['city_name'],
                'country': city_info['country'],
                'pop': city_info['population'],
                'distance': distance
            }
            nearby_cities.append(result_entry)

    # Return the list of nearby cities
    return nearby_cities


def getQuakeData(file_path='earthquakesF23.csv'):
    """
    Reads a CSV file containing earthquake data and returns a dictionary containing the earthquake information.

    Inputs:
        file_path (str): The path to the CSV file containing earthquake data. Default is 'earthquakesF23.csv'.

    Outputs:
        dict: A dictionary where the keys are tuples representing latitude and longitude, and the values are
            dictionaries containing information about each earthquake. The inner dictionaries include:
                'time' (datetime): The date and time of the earthquake.
                'latitude' (float): The latitude of the earthquake.
                'longitude' (float): The longitude of the earthquake.
                'depth' (float): The depth of the earthquake.
                'magnitude' (float): The magnitude of the earthquake.

    """
    quake_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            # Extract relevant data from the row
            time_str = f"{row['Date']} {row['Time']}"
            try:
                time_str = f"{row['Date']} {row['Time']}"
                # Convert date and time strings to datetime objects
                time = datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S')
            except ValueError:
                # Skip this row and move to the next one
                continue  

            # Extract other earthquake data from the row
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            quake_type = row['Type']
            depth = float(row['Depth'])
            magnitude = float(row['Magnitude'])
            magnitude_type = row['Magnitude Type']
            
            # Create a tuple from latitude and longitude
            coordinates = (latitude, longitude)

            # Check if coordinates already exist in the dictionary
            if coordinates not in quake_data:
                # If not, add a new entry with an inner dictionary
                quake_data[coordinates] = {
                    
                    'Type': quake_type,
                    'Depth': depth,
                    'Magnitude': magnitude,
                    'Magnitude_type': magnitude_type,
                    'datetime': time
                    
                }

    return quake_data # Return the earthquake data dictionary


def get_date_range_input(prompt, min_val, max_val):
    while True:
        try:
            user_input = input(prompt)
            if not user_input:
                return min_val, max_val

            min_date, max_date = sorted(map(lambda x: datetime.strptime(x.strip(), '%m/%d/%Y'), user_input.split(',')))

            if (min_val is None or min_date >= min_val) and (max_val is None or max_date <= max_val):
                return min_date.strftime('%m/%d/%Y'), max_date.strftime('%m/%d/%Y')
            else:
                print(f"One or more values out-of-range: <({min_val}, {max_val})>")

        except ValueError:
            print("Invalid input. Please enter valid date values separated by a comma (format: '%m/%d/%Y').")

def get_range_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            user_input = input(prompt)
            if not user_input:
                return min_val, max_val

            min_value, max_value = sorted(map(float, user_input.split(',')))

            if (min_val is None or min_value >= min_val) and (max_val is None or max_value <= max_val):
                return min_value, max_value
            else:
                print(f"One or more values out-of-range: <({min_val}, {max_val})>")

        except ValueError:
            print("Invalid input. Please enter valid numeric values separated by a comma.")

def plot_earthquake_data(quakeType, minDt, maxDt, selected_quakes):
    selected_quakes_by_year = {}
    for quake in selected_quakes:
        year = quake[1]['datetime'].year
        if year not in selected_quakes_by_year:
            selected_quakes_by_year[year] = []
        selected_quakes_by_year[year].append(quake)

    selected_quakes_sorted = sorted(selected_quakes, key=lambda x:x[1]['Magnitude'])
    
    title_str = f"{quakeType}\n{minDt} to {maxDt}"
    
    lats = [quake[0][0] for quake in selected_quakes_sorted]
    lons = [quake[0][1] for quake in selected_quakes_sorted]
    mags = [v['Magnitude'] for _, v in selected_quakes_sorted]
    
    plt.scatter(lons, lats, s=mags, c=mags, cmap='viridis')
    plt.colorbar(label='Magnitude', orientation='horizontal')
    plt.xlabel('Longitude in degrees')
    plt.ylabel('Latitude in degrees')
    
    plt.title(title_str)
    plt.show()

    years = list(selected_quakes_by_year.keys())
    event_counts = [len(events) for events in selected_quakes_by_year.values()]
    plt.bar(years, event_counts, color='blue')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.title(title_str)
    plt.show()

    avg_magnitudes = [sum(quake[1]['Magnitude'] for quake in events) / len(events) for events in selected_quakes_by_year.values()]
    plt.scatter(years, avg_magnitudes, color='blue')
    plt.xlabel('Year')
    plt.ylabel('Average Magnitude')
    plt.title(title_str)
    plt.show()

def main():
    quakes = getQuakeData()
    cities = getCityData()
    earthQuakes = [quake for quake in quakes.items() if (quake[1]['Type']).strip()=='Earthquake' ]
    print(f"Acquired data {len(cities)} cities.")
    print(f"Acquired data {len(earthQuakes)} earthquakes.")

    proceed_list = ['yes','y','ok','sure','continue','proceed']

    skip_sel = input("Skip selection? ")
    if skip_sel in proceed_list:
        print(f"ANALYZED {len(quakes)} records.")
        print("(see plots for results)")
        dt_min = (min(dt[1]['datetime'] for dt in quakes.items()).strftime('%m/%d/%Y'))
        dt_max = (max(dt[1]['datetime'] for dt in quakes.items()).strftime('%m/%d/%Y'))
        qTypes = '/ '.join(sorted(set(quake['Type'] for quake in quakes.values())))
        plot_earthquake_data(qTypes, dt_min, dt_max, quakes.items())
        return

    print("SELECT tremor type:")
    print("Enter choices separated by commas", "\nChoices are...")
    print(', '.join(sorted(set(quake['Type'] for quake in quakes.values()))))

    selected_types = [t.strip() for t in input("Enter values: ").split()]

    selected_quakes = []
    for key, quake in quakes.items():
        for selected_type in selected_types:
            if quake['Type'].lower().startswith(selected_type):
                selected_quakes.append((key, quake))

    if len(selected_quakes) == 0:
        print(f"Please select any {', '.join(sorted(set(quake['Type'] for quake in quakes.values())))}")
        return

    types = sorted(list(set(quake[1]['Type'] for quake in selected_quakes)))
    print(f"Selected {len(selected_quakes)} records.")
    
    move_on_date = input("\nWant to move on to next item? ")

    dt_min = (min(dt[1]['datetime'] for dt in selected_quakes)).strftime('%m/%d/%Y')
    dt_max = (max(dt[1]['datetime'] for dt in selected_quakes)).strftime('%m/%d/%Y')
    
    if move_on_date.lower() not in proceed_list:
        if move_on_date == "":
            move_on_date = input("Please respond. Want to move on to dates? ")
        return

    min_date, max_date = get_date_range_input("Enter minimum/maximum date values: ",
                                          datetime.strptime(dt_min, '%m/%d/%Y').date(),
                                          datetime.strptime(dt_max, '%m/%d/%Y').date())


    print("\nAccepted...")
    print("('min',", f"'{min_date}')", "('max',", f"'{max_date}')")

    selected_dts = [quake for quake in selected_quakes if min_date <= quake[1]['datetime'].strftime('%m/%d/%Y') <= max_date]
    print(f"Selected {len(selected_dts)} records.\n")
    
    move_on_mag = input("\nWant to move on to next item? ")

    mag_min = (min(v[1]['Magnitude'] for v in selected_dts))
    mag_max = (max(v[1]['Magnitude'] for v in selected_dts))
    
    if move_on_mag.lower() not in proceed_list:
        if move_on_mag == "":
            move_on_mag = input("Please respond. Want to move on to magnitude? ")
        return

    min_mag, max_mag = get_range_input("Enter minimum/maximum magnitude values: ", mag_min, mag_max)
    print("\nAccepted...")
    print(f"{{'min': {min_mag}, 'max': {max_mag}}}")

    selected_mag = [quake for quake in selected_dts if min_mag <= quake[1]['Magnitude'] <= max_mag]
    print(f"Selected {len(selected_mag)} records.\n")
    
    move_on_lat = input("\nWant to move on to next item? ")

    lat_min = (min(k[0][0] for k in selected_mag))
    lat_max = (max(k[0][0] for k in selected_mag))
    
    if move_on_lat.lower() not in proceed_list:
        if move_on_lat == "":
            move_on_lat = input("Please respond. Want to move on to Latitude? ")
        return

    min_lat, max_lat = get_range_input("Enter minimum/maximum latitude values: ", lat_min, lat_max)
    print(f"\nAccepted...")
    print(f"{{'min': {min_lat}, 'max': {max_lat}}}")

    selected_lat = [quake for quake in selected_mag if min_lat <= quake[0][0] <= max_lat]
    print(f"Selected {len(selected_lat)} records.\n")
    
    move_on_lon = input("\nWant to move on to next item? ")

    lon_min = (min(k[0][1] for k in selected_mag))
    lon_max = (max(k[0][1] for k in selected_mag))
    
    if move_on_lon.lower() not in proceed_list:
        if move_on_lon == "":
            move_on_lon = input("Please respond. Want to move on to longitude? ")
        return

    min_lon, max_lon = get_range_input("Enter minimum/maximum longitude values: ", lon_min, lon_max)
    print(f"\nAccepted...")
    print(f"{{'min': {min_lon}, 'max': {max_lon}}}")

    selected_lng = [quake for quake in selected_lat if min_lon <= quake[0][1] <= max_lon]
    print(f"Selected {len(selected_lng)} records.\n")
    
    move_on_anl = input("\nWant to move on to next item? ")

    if move_on_anl.lower() not in proceed_list:
        if move_on_anl == "":
            move_on_anl = input("Please respond. Want to move on to analysis? ")
        return

    print(f"ANALYZED {len(selected_lng)} {', '.join(types).lower()} records.\n")
    print("(see plots for results)")

    plot_earthquake_data(' / '.join(types), min_date, max_date, selected_lng)

if __name__ == "__main__":
    print("*** Earthquake Data Analysis ***\n")
    main()
