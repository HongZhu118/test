import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def read_station_coords(file_path):
    station_coords = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                station_name, coords_str = line.strip().split(':')
                # 移除括号，提取纬度和经度
                coords_str = coords_str.strip()[1:-1]  # 去掉外围的括号
                lat, lon = coords_str.split(',')
                lat = lat.strip().strip("'")  # 去掉多余的空格和引号
                lon = lon.strip().strip("'")
                station_coords[station_name.strip()] = (float(lat), float(lon))
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
    return station_coords
def plot_stations_on_map(station_coords):
    # 创建一个地图实例，设置范围为中国区域
    fig, ax = plt.subplots(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=18, urcrnrlat=54, llcrnrlon=73, urcrnrlon=135, lat_ts=20, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral',lake_color='aqua')

    # 将经纬度转换为地图上的x和y坐标，并绘制
    for station, (lat, lon) in station_coords.items():
        x, y = m(lon, lat)
        m.plot(x, y, 'bo', markersize=5)
        plt.text(x, y, station, fontsize=9)

    plt.title('Station Coordinates Plot')
    plt.show()

# 假设txt文件路径
file_path = 'C:\\Users\\yee\\pythonProject1\\train\\坐标.txt'
station_coords = read_station_coords(file_path)
plot_stations_on_map(station_coords)
