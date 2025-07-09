import requests
from geopy.geocoders import Nominatim

def get_latitude_longitude(location):
    # Use Geopy's Nominatim geocoder
    geolocator = Nominatim(user_agent="hospital_locator")
    location = geolocator.geocode(location)
    
    if location:
        return (location.latitude, location.longitude)
    else:
        print("Location not found.")
        return None

def fetch_nearby_hospitals(latitude, longitude, radius=5000):
    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Overpass QL query to find hospitals within the specified radius
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
    );
    out body;
    """
    
    # Send request to Overpass API
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code == 200:
        data = response.json()
        hospitals = []
        
        # Extract relevant information
        for element in data['elements']:
            name = element.get('tags', {}).get('name', 'N/A')
            address = element.get('tags', {}).get('addr:full', 'N/A')
            hospitals.append({
                'name': name,
                'address': address,
                'latitude': element['lat'] if 'lat' in element else None,
                'longitude': element['lon'] if 'lon' in element else None
            })
        
        return hospitals
    else:
        print("Error fetching data from Overpass API")
        return []

