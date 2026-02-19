# %%
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import os

# Size extraction 

import re

def extract_size_from_text(text):
    """
    Extract realistic built-up/property size from messy real estate descriptions.
    Supports sqm, m², square meters, sq ft, and acre.
    """

    if not text:
        return "N/A"

    text = text.replace(",", "")
    candidates = []

    # ---------------------------------------------------
    # 1. Ranges in sqm (e.g. 350 – 400 sqm, 465 to 476 sqm)
    # ---------------------------------------------------
    range_matches = re.findall(
        r'(\d+(\.\d+)?)\s*(?:–|-|to)\s*(\d+(\.\d+)?)\s*(sqm|m²|square meters?)',
        text,
        re.IGNORECASE
    )

    for match in range_matches:
        low = float(match[0])
        high = float(match[2])
        if high >= 30:  # realistic lower bound for Kenya
            candidates.append((high, f"{match[0]}–{match[2]} sqm"))

    # ---------------------------------------------------
    # 2. Single sqm values
    # ---------------------------------------------------
    sqm_matches = re.findall(
        r'(\d+(\.\d+)?)\s*(sqm|m²|square meters?)',
        text,
        re.IGNORECASE
    )

    for match in sqm_matches:
        value = float(match[0])
        if value >= 30:  # ~ small studio size
            candidates.append((value, f"{match[0]} sqm"))

    # ---------------------------------------------------
    # 3. Square feet (convert to sqm for comparison)
    # ---------------------------------------------------
    sqft_matches = re.findall(
        r'(\d+(\.\d+)?)\s*(sq\.?\s*ft\.?|sqft)',
        text,
        re.IGNORECASE
    )

    for match in sqft_matches:
        sqft_value = float(match[0])
        if sqft_value >= 300:  # adjusted for Kenya
            sqm_equivalent = sqft_value * 0.092903
            candidates.append((sqm_equivalent, f"{match[0]} sq ft"))

    # ---------------------------------------------------
    # 4. Acres (convert to sqm for comparison)

    acre_matches = re.findall(
        r'(\d+/\d+|\d+(\.\d+)?)\s*-?\s*(acre)',
        text,
        re.IGNORECASE
    )

    for match in acre_matches:
        raw_value = match[0]

        if "/" in raw_value:
            num, denom = raw_value.split("/")
            acre_value = float(num) / float(denom)
        else:
            acre_value = float(raw_value)

        if acre_value >= 0.05:  # ignore unrealistic tiny plots
            sqm_equivalent = acre_value * 4046.86
            candidates.append((sqm_equivalent, f"{raw_value} acre"))

    # ---------------------------------------------------
    # 5. Return largest realistic size, and avoid small sizes that are likely not the property size
    if candidates:
        # Filter out candidates that are too small (e.g., less than 30 sqm)
        filtered_candidates = [c for c in candidates if c[0] >= 30]
        if filtered_candidates:
            return max(filtered_candidates, key=lambda x: x[0])[1]

    return "N/A"


# ======================================
# Fetch with retries

def fetch_page(url, headers, retries=3):
    """Fetch a page with retries and error handling."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {attempt+1}: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: Request failed - {e}")
        time.sleep(2)
    return None

# ======================================================
# Extract from listing card

def extract_property_type(title):

    # Extract property type from title (e.g., 'Apartment', 'Townhouse')
    title_lower = title.lower()
    types = ['apartment', 'townhouse', 'bungalow', 'villa', 'mansion', 'house', 'land', 'commercial']
    for t in types:
        if t in title_lower:
            return t.capitalize()
    return 'Unknown'


def extract_bedrooms_bathrooms_size(listing):
    # Extract bedrooms, bathrooms, size from swiper slides and description on listing card
    bedrooms = bathrooms = "N/A"
    size_from_swiper = "N/A"

    # 1. Swiper slides (bedrooms, bathrooms, quick size)
    swiper_div = listing.find('div', class_='scrollable-list')
    if swiper_div:
        slides = swiper_div.find_all('div', class_='swiper-slide')
        for slide in slides:
            text = slide.get_text(strip=True)
            if 'Bedroom' in text:
                bedrooms = text
            elif 'Bathroom' in text:
                bathrooms = text
            elif 'm²' in text or 'sq' in text.lower() or 'acre' in text.lower():
                size_from_swiper = text

    # 2. extract size from description on the listing card
    size_from_desc = "N/A"
    # Look for common description containers
    desc_div = (listing.find('div', id='truncatedDescription') or 
                listing.find('div', class_='description') or
                listing.find('p', class_='description'))
    if desc_div:
        desc_text = desc_div.get_text(" ", strip=True)
        size_from_desc = extract_size_from_text(desc_text)

    # Prefer description-based size if found
    if size_from_desc != "N/A":
        size = size_from_desc
    else:
        size = size_from_swiper

    return bedrooms, bathrooms, size


# ===============================================
# Extract from detail page

def parse_detail_page(detail_soup):

    # Extract creation date, amenities, nearby, and size from property detail page
    utilities = []
    nearby = []
    created_at = 'N/A'
    size_from_detail = "N/A"

    # Creation date
    created_tag = detail_soup.find(string=lambda x: x and "Created At:" in x)
    if created_tag:
        created_at = created_tag.strip().replace("Created At:", "").strip()

    # Size from detail page (full description)
    # Look in common description containers first
    desc_section = (detail_soup.find('div', class_='property-description') or 
                    detail_soup.find('div', class_='description') or
                    detail_soup.find('section', class_='description'))
    if desc_section:
        desc_text = desc_section.get_text(" ", strip=True)
        size_from_detail = extract_size_from_text(desc_text)
    else:
        # Fallback: search entire page text
        page_text = detail_soup.get_text(" ", strip=True)
        size_from_detail = extract_size_from_text(page_text)

    # Amenities and nearby facilities
    sections = detail_soup.find_all("div", class_="px-3 py-3 even:bg-gray-50")
    for section in sections:
        title_span = section.find("span", class_="font-semibold")
        if not title_span:
            continue

        section_name = title_span.get_text(strip=True).lower()
        items_div = section.find("div", class_="flex flex-wrap gap-3")
        if not items_div:
            continue

        items = [span.get_text(strip=True) for span in items_div.find_all("span") if span.get_text(strip=True)]
        if "internal features" in section_name or "external features" in section_name:
            utilities.extend(items)
        elif "nearby" in section_name:
            nearby.extend(items)

    return created_at, utilities, nearby, size_from_detail

# ======================================
# Scrape one listing

def scrape_listing(listing, base_url, headers):

    # Scrape data from a single listing card and its detail page

    # --- Basic info from card ---
    title_tag = listing.find('h2')
    title = title_tag.get_text(strip=True) if title_tag else 'No title'

    price_tag = listing.find('a', class_='pointer-events-none z-10 no-underline')
    price = price_tag.get_text(strip=True) if price_tag else 'No price'

    location_tag = listing.find('p', class_='w-full truncate font-normal capitalize')
    location = location_tag.get_text(strip=True) if location_tag else 'No location'

    property_type = extract_property_type(title)
    bedrooms, bathrooms, size = extract_bedrooms_bathrooms_size(listing)

    # --- Detail page URL ---
    property_tag = listing.find('a', href=True)
    if not property_tag:
        return None
    property_url = urljoin(base_url, property_tag['href'])

    # --- Fetch detail page ---
    detail_response = fetch_page(property_url, headers)
    if not detail_response:
        # Return basic info only
        return {
            'Title': title,
            'Property Type': property_type,
            'Price': price,
            'Location': location,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Size': size,
            'Amenities': [],
            'Surroundings': [],
            'Created At': 'N/A',
            'URL': property_url
        }

    detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
    created_at, utilities, nearby, size_from_detail = parse_detail_page(detail_soup)

    # Override size with detail page version if found
    if size_from_detail != "N/A":
        size = size_from_detail

    return {
        'Title': title,
        'Property Type': property_type,
        'Price': price,
        'Location': location,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Size': size,
        'Amenities': utilities,
        'Surroundings': nearby,
        'Created At': created_at,
    }

# ===============================================
# Main scraping loop

def scrape_pages(start_page, end_page, max_listings=800):
    """
    Iterates over pages and listings, collects data, stops when max_listings reached.
    """
    base_url = 'https://www.buyrentkenya.com/houses-for-sale'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    properties = []
    page_num = start_page

    while page_num <= end_page and len(properties) < max_listings:
        url = f'{base_url}?page={page_num}'
        print(f"Scraping page {page_num}: {url}")

        response = fetch_page(url, headers)
        if not response:
            print(f"Failed to retrieve page {page_num}, moving to next.")
            page_num += 1
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        listings = soup.find_all('div', class_='listing-card')

        if not listings:
            print(f"No listings found on page {page_num}. Stopping.")
            break

        for listing in listings:
            if len(properties) >= max_listings:
                break
            

        # SCRAPPED DATA ======================
            property_data = scrape_listing(listing, base_url, headers)
            if property_data:
                properties.append(property_data)
                # print(f"  Scraped: {property_data['Title']} ({len(properties)} total)")

            # Polite delay between detail page requests
            time.sleep(1)

        page_num += 1
        time.sleep(2)  # delay between pages

    print(f"Scraping finished. Total listings collected: {len(properties)}")
    return pd.DataFrame(properties)

# ========================
# Save data functions

def save_raw_data(df, filename='../data/raw_listings.csv'):
    """Save raw dataframe to CSV, creating directory if needed."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)


    df.to_csv(filename, index=False)
    print(f"Raw data saved to {filename}")

def create_data_dictionary():
    """Return the data dictionary as a list of dicts."""
    return [
        {"Column Name": "Title", "Description": "Name of property listing", "Data Type": "String", "Example": "5 Bed Townhouse with En Suite in Nyali Area"},
        {"Column Name": "Property Type", "Description": "Type of property (apartment, townhouse, etc.)", "Data Type": "String", "Example": "Townhouse"},
        {"Column Name": "Price", "Description": "Listed price", "Data Type": "String / Numeric", "Example": "KSh 29,500,000"},
        {"Column Name": "Location", "Description": "Property location", "Data Type": "String", "Example": "Nyali Area, Nyali"},
        {"Column Name": "Bedrooms", "Description": "Number of bedrooms", "Data Type": "String / Int", "Example": "5 Bedrooms"},
        {"Column Name": "Bathrooms", "Description": "Number of bathrooms", "Data Type": "String / Int", "Example": "5 Bathrooms"},
        {"Column Name": "Size", "Description": "Property size with unit", "Data Type": "String", "Example": "250 m²"},
        {"Column Name": "Amenities", "Description": "Internal & external features", "Data Type": "List (comma-separated in CSV)", "Example": "Aircon, Alarm, Backup Generator"},
        {"Column Name": "Surroundings", "Description": "Nearby facilities / landmarks", "Data Type": "List (comma-separated in CSV)", "Example": "Bus Stop, Golf Course, Hospital"},
        {"Column Name": "Created At", "Description": "Date listing was created", "Data Type": "Date / String", "Example": "09 February 2026"},
        {"Column Name": "URL", "Description": "Link to the property page", "Data Type": "String", "Example": "https://www.buyrentkenya.com/..."},
    ]




# %%
# Main execution
def main():
    # Configuration
    START_PAGE = 1
    END_PAGE = 40      
    MAX_LISTINGS = 800     # max

    # Scrape
    df = scrape_pages(START_PAGE, END_PAGE, MAX_LISTINGS)

    df = df.copy()
    df['Amenities'] = df['Amenities'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    df['Surroundings'] = df['Surroundings'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Display sample data with flattened amenities and surroundings
    print("Sample data with flattened amenities and surroundings:")
    print(df.head())

    # Save raw data
    save_raw_data(df, '../data/raw_listings.csv')

    # Save data dictionary as JSON
    data_dict = create_data_dictionary()
    dd_df = pd.DataFrame(data_dict)
    dd_df.to_json('../data/data_dictionary.json', orient='records', indent=2)
    print("Data dictionary saved as data_dictionary.json")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total listings: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

   
if __name__ == "__main__":
    main()


