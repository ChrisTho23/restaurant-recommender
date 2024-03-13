import json

from config import DATA, business

def extract_gastronomy(input_file, output_file, keywords):
    print(f"Extracting gastronomy businesses ...")
    with open(DATA["business"], 'r') as file, open(DATA["gastronomy"], 'w') as outfile:
        for line in file:
            try:
                entry = json.loads(line)
                categories = entry.get('categories', '')
                categories = '' if categories is None else categories
                if any(keyword in categories for keyword in keywords):
                    json.dump(entry, outfile)
                    outfile.write('\n')
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
    print(f"Gastronomy businesses extracted to {DATA['gastronomy']}")

if __name__ == "__main__":
    extract_gastronomy(DATA["business"], DATA["gastronomy"], business['type'])

