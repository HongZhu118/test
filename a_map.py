import json
import pandas as pd
import matplotlib.pyplot as plt

# # Load the new Excel data for station coordinates
# new_stations_data = pd.read_excel('../files/全国火车站含经纬度（2023年12月27日更新）.xlsx')
#
# # Create a dictionary from the new data for station coordinates lookup
# station_coords = {row['站名']: (row['经度'], row['纬度']) for index, row in new_stations_data.iterrows()}
#
# station_coords = {key: (max(value), min(value)) for key, value in station_coords.items()}
with open('data.json', 'r',encoding='utf-8') as file:
    station_coords_data = json.load(file)

# Create a dictionary from the new data for station coordinates lookup
# station_coords = {item['name']: (item['lng'], item['lat']) for item in station_coords_data}

# Load the route data from the CSV file
route_file_path = '../files/route.csv'
route_data = pd.read_csv(route_file_path)
route_list = [route.split('、') for route in route_data['Mnst']]

# Function to plot routes based on station names
def plot_route(station_list):
    route_lons = []
    route_lats = []
    for station in station_list:
        new_station = station + '站'
        if new_station in station_coords_data and station_coords_data[new_station] != None:
            coord = station_coords_data[new_station]
            route_lons.append(coord[1])
            route_lats.append(coord[0])
    return route_lons, route_lats

# Plot each route
plt.figure(figsize=(12, 10))
for stations in route_list:
    route_lons, route_lats = plot_route(stations)
    if route_lons and route_lats:
        plt.plot(route_lons, route_lats, marker='o', color='black')


# Adding map details
plt.title('Station Routes')
plt.xlabel('Longitude')
plt.xlim((30, 140))
plt.ylabel('Latitude')
plt.ylim((10, 50))
plt.grid(True)
plt.show()
