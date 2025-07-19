import reverse_geocoder as rg
import pandas as pd

metadata = pd.read_csv('../../data/metadata.csv')

region_map = {
    "North America": ["US", "CA"],
    "Latin America": ["MX", "GT", "SV", "HN", "NI", "CR", "PA", "CO", "VE", "EC", "PE", "BO", "PY", "AR", "CL", "UY", "BR", "GY", "SR", "GF", "CU", "HT", "DO", "PR", "VI", "BZ"],
    "Western & Northern Europe": ["FR", "DE", "NL", "BE", "LU", "AT", "CH", "LI", "GB", "IE", "DK", "SE", "NO", "FI", "IS", "AX", "FO"],
    "Southern Europe": ["ES", "PT", "IT", "GR", "MT", "AD", "SM", "VA", "CY"],
    "Eastern Europe & Balkans": ["TR", "PL", "CZ", "SK", "HU", "EE", "LV", "LT", "SI", "HR", "BA", "RS", "ME", "MK", "AL", "RO", "BG", "MD"],
    "Russia & Cyrillic": ["RU", "UA", "BY", "MN", "KZ", "KG", "UZ"],
    "East Asia": ["JP", "KR", "TW", "HK", "CN"],
    "Southeast Asia": ["TH", "MY", "SG", "ID", "PH", "VN", "KH", "LA", "BN", "TL"],
    "South Asia": ["IN", "BD", "LK", "BT", "NP"],
    "Africa": ["NA", "ZA", "NG", "KE", "SZ", "LS", "SN", "BW", "GH", "RW", "UG", "GM", "CI", "BF", "TG", "GN", "TZ", "ET", "ML", "ZW", "CD", "GW"],
    "Arabia": ["PS", "LB", "QA", "IL", "AE", "OM", "TN", "JO", "SY", "YE", "MR"],
    "Oceania": ["AU", "NZ"],
}


region_to_id_map = {
    "North America": 0,
    "Latin America": 1,
    "Western & Northern Europe": 2,
    "Southern Europe": 3,
    "Eastern Europe & Balkans": 4,
    "Russia & Cyrillic": 5,
    "East Asia": 6,
    "Southeast Asia": 7,
    "South Asia": 8,
    "Africa": 9,
    "Arabia": 10,
    "Oceania": 11,
    "Rare Regions": 12
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

    metadata['countryCode'].fillna("NA")
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
            super_region_list.append("Rare Regions")

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