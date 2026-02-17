# Property Price Scraper and Dataset

This project scrapes property listings within Nairobi and compiles a structured dataset suitable for analysis or building a machine learning model to predict property prices.

## Dataset Overview

The scraped dataset contains information about properties for sale, including price, location, size, and amenities. The data is stored in `properties.csv`.

### Data Dictionary

The following table explains each column in the dataset. The data dictionary is also available in `data_dictionary.json` in the repo.

| Column Name   | Description                          | Data Type                | Example |
|---------------|--------------------------------------|-------------------------|---------|
| Title         | Name or headline of the property     | String                  | "5 Bed Townhouse with En Suite in Nyali Area" |
| Price         | Listed price of the property         | String / Numeric        | "KSh 29,500,000" |
| Location      | Property location                     | String                  | "Nyali Area, Nyali" |
| Bedrooms      | Number of bedrooms                    | String / Integer        | "5 Bedrooms" |
| Bathrooms     | Number of bathrooms                   | String / Integer        | "5 Bathrooms" |
| Amenities     | Internal & external features          | String (comma-separated)| "Aircon, Alarm, Backup Generator" |
| Surroundings  | Nearby facilities / landmarks         | String (comma-separated)| "Bus Stop, Golf Course, Hospital" |
| Created At    | Date the listing was created          | Date / String           | "09 February 2026" |
| URL           | Link to the property detail page      | String                  | "https://www.buyrentkenya.com/property/12345" |

### Usage

```python
import pandas as pd

# Load scraped data
df = pd.read_csv("properties.csv")

# Inspect data
print(df.head())
