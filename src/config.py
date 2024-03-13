from pathlib import Path    

DATA_DIR = Path('../data')

# dict with the names of the files and their respective paths
DATA = {
    'business': DATA_DIR / 'yelp_academic_dataset_business.json',
    'gastronomy': DATA_DIR / 'yelp_academic_dataset_gastronomy.json',
    'review': DATA_DIR / 'yelp_academic_dataset_review.json',
}

business = {
    'type': [
        "Bakeries", "Bar", "Barbeque", "Bars", "Belgian", "Bistros", "Brasseries", 
        "Brewpubs", "Bubble", "Buffets", "Cafes", "Candy", "Caribbean", "Cheese", 
        "Chinese", "Chocolate", "Cocktail", "Coffee", "Creperies", "Cuban", "Delis", 
        "Diners", "Donuts", "Egyptian", "Empanadas", "Ethiopian", "Fast", "Filipino", 
        "Fish", "French", "Fusion", "Gelato", "Greek", "Grill", "Halal", "Hawaiian", 
        "Hookah", "Indian", "Indonesian", "Italian", "Japanese", "Juice", "Karaoke", 
        "Kebab", "Korean", "Kosher", "Lebanese", "Lounges", "Macarons", "Malaysian", 
        "Mediterranean", "Mexican", "Mongolian", "Moroccan", "Nightlife", "Noodles", 
        "Pancakes", "Pasta", "Pastry", "Patisserie", "Pizza", "Polish", "Portuguese", 
        "Pub", "Ramen", "Restaurants", "Russian", "Salad", "Sandwiches", "Seafood", 
        "Senegalese", "Shanghainese", "Sicilian", "Sushi", "Syrian", "Tacos", 
        "Taiwanese", "Tapas", "Tea", "Tex-Mex", "Thai", "Turkish", "Vegetarian", 
        "Vietnamese", "Waffles", "Wine",
    ]
}