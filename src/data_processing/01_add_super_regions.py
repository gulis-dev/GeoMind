import reverse_geocoder as rg
import pandas as pd

metadata = pd.read_csv('../../data/metadata.csv')

region_map = {
    "North America": ["US", "CA"],
    "Latin America": ["MX", "GT", "SV", "HN", "NI", "CR", "PA", "CO", "VE", "EC", "PE", "BO", "PY", "AR", "CL", "UY", "BR", "GY", "SR", "GF", "CU", "HT", "DO", "PR"],
    "Western & Northern Europe": ["FR", "DE", "NL", "BE", "LU", "AT", "CH", "LI", "GB", "IE", "DK", "SE", "NO", "FI", "IS"],
    "Southern Europe": ["ES", "PT", "IT", "GR", "MT", "AD", "SM", "VA", "CY"],
    "Eastern Europe & Balkans": ["PL", "CZ", "SK", "HU", "EE", "LV", "LT", "SI", "HR", "BA", "RS", "ME", "MK", "AL", "RO", "BG", "MD"],
    "Russia & Cyrillic": ["RU", "UA", "BY", "BG", "RS", "MK", "MN", "KZ", "KG"],
    "Asia": ["JP", "KR", "TW", "TH", "MY", "SG", "ID", "PH", "VN", "KH", "LA", "IN", "BD", "LK", "BT", "HK"],
    "Oceania & Rare Regions": ["AU", "NZ", "ZA", "BW", "LS", "SZ", "NG", "GH", "SN", "KE", "UG", "TZ"] # and other rare ones
}


region_to_id_map = {
    "North America": 0,
    "Latin America": 1,
    "Western & Northern Europe": 2,
    "Southern Europe": 3,
    "Eastern Europe & Balkans": 4,
    "Russia & Cyrillic": 5,
    "Asia": 6,
    "Oceania & Rare Regions": 7
}


def add_contry_code(metadata):
    """
    Adds a country code to the metadata DataFrame based on latitude and longitude.

    Parameters:
    metadata (DataFrame): DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
    DataFrame: Updated DataFrame with a new 'countryCode' column.
    """

    coordinates = list(zip(metadata['latitude'], metadata['longitude']))

    results = rg.search(coordinates)

    country_list = [result['cc'] for result in results]

    metadata['countryCode'] = country_list
    return metadata

def add_region(metadata):
    """
    Adds a super region to the metadata DataFrame based on country codes.

    Parameters:
    metadata (DataFrame): DataFrame containing 'countryCode' column.

    Returns:
    DataFrame: Updated DataFrame with a new 'superRegion' column.
    """
    super_region_list = []
    for index, row in metadata.iterrows():
        country_code = row['countryCode']
        found = False
        for region, countries in region_map.items():
            if country_code in countries:
                super_region_list.append(region)
                found = True
                break
        if not found:
            super_region_list.append("Oceania & Rare Regions")

        print(f"Processed {index + 1}/{len(metadata)}: {country_code} -> {super_region_list[-1]}")



    metadata['superRegion'] = super_region_list
    return metadata

def add_region_id(metadata):
    """
    Adds a region ID to the metadata DataFrame based on the super region name.

    Parameters:
    metadata (DataFrame): DataFrame containing 'superRegion' column.

    Returns:
    DataFrame: Updated DataFrame with a new 'region_id' column.
    """
    metadata['region_id'] = metadata['superRegion'].map(region_to_id_map)
    print(f"Processed {len(metadata)}: {metadata['region_id']}")
    return metadata

if __name__ == '__main__':
    metadata = add_contry_code(metadata)
    metadata = add_region(metadata)
    metadata = add_region_id(metadata)

    print(metadata.head())
    metadata.to_csv('../../data/metadata_final.csv', index=False)