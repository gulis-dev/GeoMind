import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


metadata = pd.read_csv("../../data/metadata_final.csv", keep_default_na=False) # because NA values in 'countryCode' column is treated as NaN

def plot_region_distribution(metadata):
    plt.figure(figsize=(20, 10), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()

    # South Asia
    south_asia = metadata[metadata['superRegion'] == 'South Asia']
    lat_south_asia = south_asia['latitude']
    lon_south_asia = south_asia['longitude']
    plt.scatter(lon_south_asia, lat_south_asia, s=2, color='red', alpha=0.3, transform=ccrs.PlateCarree())

    # East Asia
    east_asia = metadata[metadata['superRegion'] == 'East Asia']
    lat_east_asia = east_asia['latitude']
    lon_east_asia = east_asia['longitude']
    plt.scatter(lon_east_asia, lat_east_asia, s=2, color='purple', alpha=0.3, transform=ccrs.PlateCarree())

    # Southeast Asia
    southeast_asia = metadata[metadata['superRegion'] == 'Southeast Asia']
    lat_southeast_asia = southeast_asia['latitude']
    lon_southeast_asia = southeast_asia['longitude']
    plt.scatter(lon_southeast_asia, lat_southeast_asia, s=2, color='orange', alpha=0.3, transform=ccrs.PlateCarree())

    # Oceania
    oceania = metadata[metadata['superRegion'] == 'Oceania']
    lat_oceania = oceania['latitude']
    lon_oceania = oceania['longitude']
    plt.scatter(lon_oceania, lat_oceania, s=2, color='teal', alpha=0.3, transform=ccrs.PlateCarree())

    # Western & Northern Europe
    western_europe = metadata[metadata['superRegion'] == 'Western & Northern Europe']
    lat_western_europe = western_europe['latitude']
    lon_western_europe = western_europe['longitude']
    plt.scatter(lon_western_europe, lat_western_europe, s=2, color='green', alpha=0.3, transform=ccrs.PlateCarree())

    # Latin America
    latin_america = metadata[metadata['superRegion'] == 'Latin America']
    lat_latin_america = latin_america['latitude']
    lon_latin_america = latin_america['longitude']
    plt.scatter(lon_latin_america, lat_latin_america, s=2, color='gold', alpha=0.3, transform=ccrs.PlateCarree())

    # Eastern Europe & Balkans
    eastern_europe = metadata[metadata['superRegion'] == 'Eastern Europe & Balkans']
    lat_eastern_europe = eastern_europe['latitude']
    lon_eastern_europe = eastern_europe['longitude']
    plt.scatter(lon_eastern_europe, lat_eastern_europe, s=2, color='magenta', alpha=0.3, transform=ccrs.PlateCarree())

    # North America
    north_america = metadata[metadata['superRegion'] == 'North America']
    lat_north_america = north_america['latitude']
    lon_north_america = north_america['longitude']
    plt.scatter(lon_north_america, lat_north_america, s=2, color='maroon', alpha=0.3, transform=ccrs.PlateCarree())

    # Southern Europe
    southern_europe = metadata[metadata['superRegion'] == 'Southern Europe']
    lat_southern_europe = southern_europe['latitude']
    lon_southern_europe = southern_europe['longitude']
    plt.scatter(lon_southern_europe, lat_southern_europe, s=2, color='cyan', alpha=0.3, transform=ccrs.PlateCarree())

    # Russia & Cyrillic
    russia_cyrillic = metadata[metadata['superRegion'] == 'Russia & Cyrillic']
    lat_russia_cyrillic = russia_cyrillic['latitude']
    lon_russia_cyrillic = russia_cyrillic['longitude']
    plt.scatter(lon_russia_cyrillic, lat_russia_cyrillic, s=2, color='deeppink', alpha=0.3,
                transform=ccrs.PlateCarree())

    # Africa
    africa = metadata[metadata['superRegion'] == 'Africa']
    lat_africa = africa['latitude']
    lon_africa = africa['longitude']
    plt.scatter(lon_africa, lat_africa, s=2, color='saddlebrown', alpha=0.3, transform=ccrs.PlateCarree())

    # Arabia
    arabia = metadata[metadata['superRegion'] == 'Arabia']
    lat_arabia = arabia['latitude']
    lon_arabia = arabia['longitude']
    plt.scatter(lon_arabia, lat_arabia, s=2, color='darkkhaki', alpha=0.3, transform=ccrs.PlateCarree())

    # Rare Regions
    rare_regions = metadata[metadata['superRegion'] == 'Rare Regions']
    lat_rare_regions = rare_regions['latitude']
    lon_rare_regions = rare_regions['longitude']
    plt.scatter(lon_rare_regions, lat_rare_regions, s=2, color='blue', alpha=0.3, transform=ccrs.PlateCarree())

    # legend
    plt.title('Distribution of Super Regions on World Map')
    plt.legend([
        'South Asia', 'East Asia', 'Southeast Asia', 'Oceania',
        'Western & Northern Europe', 'Latin America', 'Eastern Europe & Balkans',
        'North America', 'Southern Europe', 'Russia & Cyrillic',
        'Africa', 'Arabia', 'Rare Regions'
    ], loc='upper left')

    plt.show()


def plot_rare_regions(metadata):
    plt.figure(figsize=(20, 10), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()

    rare_regions = metadata[metadata['superRegion'] == 'Rare Regions']
    lat_rare_regions = rare_regions['latitude']
    lon_rare_regions = rare_regions['longitude']
    plt.scatter(lon_rare_regions, lat_rare_regions, s=20, color='blue', alpha=1, transform=ccrs.PlateCarree())

    plt.show()

if __name__ == "__main__":
    print(metadata['superRegion'].value_counts())
    plot_region_distribution(metadata)
    # plot_rare_regions(metadata)
    print(metadata[metadata['superRegion'] == 'Rare Regions']["countryCode"].value_counts())
    print(metadata[metadata["countryCode"].isnull()]["countryCode"])

