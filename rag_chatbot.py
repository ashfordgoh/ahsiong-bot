import json
import os
import sqlite3
from openai import OpenAI
from fuzzywuzzy import fuzz
from typing import Dict, List
import insert_data  # Import for inserting initial data
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Make sure to set the API key properly

def simplify_output(matches):
    # Flatten the list and extract only the first element of each tuple (product ingredient)
    unique_matches = list(set([match[0] for match in matches]))
    return unique_matches

class ChatGPTIntentClassifier:
    """Classify the user's intent and extract information like product name or allergies."""
    
    def classify_intent(self, user_query: str):
        """Classify the user intent using OpenAI GPT-3.5 model."""
        prompt = f"""
        Classify the user's query into one of the following categories:
        1. Check allergies for a specific product.
        2. Find ingredient list of product(s).
        3. Answer general allergy-related questions.
        Please provide a single response with the category and extracted information (product name) if applicable.

        User query: "{user_query}"
        """
        
        # Updated API Call to match with the latest OpenAI Python Client
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the latest model you have access to (gpt-3.5-turbo or gpt-4)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        classified_output = response.choices[0].message.content.strip()
        
        # Print the classification output for debugging
        print(f"[41]ChatGPT classification output: {classified_output}")
        # Parse the classification response to extract the category and information
        category = None
        extracted_info = None
        
        # Split the classified_output into lines and extract information
        lines = classified_output.split("\n")
        
        # Extract category (first part of the response)
        if "Check allergies for a specific product" in lines[0]:
            category = "Check allergies for a specific product"
        elif "Find ingredient list of product(s)" in lines[0]:
            category = "Find ingredient list of product(s)"
        elif "Answer general allergy-related questions" in lines[0]:
            category = "General question"
        
        # Extract product name or allergy list (second part of the response)
        if category == "Find ingredient list of product(s)":
           # Extract product names using split
            extracted_info = lines[1].replace("Product names - ", "").strip()

            # Split the string at the colon ":"
            split_info = extracted_info.split(":")  # This will give us a list where the first element is before the colon and the second part is after

            # Now, we only need the part after the colon (which will be the actual product names)
            extracted_info = split_info[1] if len(split_info) > 1 else split_info[0]  # In case the string doesn't have a colon, just use the whole string

            product_names = [name.strip() for name in extracted_info.replace("and", ",").split(",")]
            extracted_info = product_names
            print(f"[Debug] Extracted product names: {extracted_info}")
        elif category == "Check allergies for a specific product":
            # Ensure the extracted information contains only the product name and remove any additional text
            extracted_info = lines[1].split(":")  # Split at the colon to isolate product name from other info
            extracted_info = extracted_info[1] if len(extracted_info) > 1 else extracted_info[0]  # Take the part after the colon

            # Remove any extra spaces and unnecessary characters
            extracted_info = extracted_info.strip().lower()  # Convert it to lowercase and strip any excess spaces

            print(f"[Debug] Extracted product name: {extracted_info}")
        elif category == "General question":
            extracted_info = user_query  # For general questions, use the user query itself
        
        return category, extracted_info

class AllergyChecker:
    def __init__(self, db_path='ingredient_database.db'):
        self.db_manager = DatabaseManager(db_path)
        self.intent_classifier = ChatGPTIntentClassifier()

    def insert_data_if_needed(self):
        """Inserts data into the database if needed."""
        print("Inserting data into the database...")
        insert_data.insert_into_products()  # Ensure insert functions are called
        insert_data.insert_into_allergies()  # Ensure insert functions are called
        print("Data inserted successfully.")

    def get_product_allergies(self, product_name: str):
        """Retrieve potential allergies for a given product"""
        allergies_dict = self.db_manager.get_allergies()
        products = self.db_manager.get_products()

        # Normalize the product name
        product_name = product_name.lower().strip()
        print(f"[Debug] Normalized product name: {product_name}")

        best_match = self.find_closest_match(product_name, list(products.keys()), threshold=60)
        print(f"[Debug] Best match for product: {best_match}")  # Print the best match

        potential_allergies = {}
        
        if best_match:
            product_ingredients = products[best_match]['ingredients']  # Get the ingredients of the matched product
            print(f"[Debug] Ingredients for {best_match}: {product_ingredients}")

            for allergy, allergy_ingredients in allergies_dict.items():
                # Normalize both product and allergy ingredients
                cleaned_allergy_ingredients = [ing.strip().lower() for ing in allergy_ingredients]
                cleaned_product_ingredients = [ing.strip().lower() for ing in product_ingredients]
                print(f"[Debug] Checking allergy: {allergy}, ingredients: {cleaned_allergy_ingredients}")
                
                matching_ingredients = []

                for product_ingredient in cleaned_product_ingredients:
                    # Check for partial matches using fuzzy matching
                    for allergy_ingredient in cleaned_allergy_ingredients:
                        ratio = fuzz.partial_ratio(product_ingredient, allergy_ingredient)
                        if ratio > 80:  # If the match is above a certain threshold
                            matching_ingredients.append((product_ingredient, allergy_ingredient))
                            print(f"[Debug] Found a match: {product_ingredient} matches {allergy_ingredient} with ratio {ratio}")

                if matching_ingredients:
                    print(f"[Debug] Found matching ingredients for {allergy}: {matching_ingredients}")
                    potential_allergies[allergy] = simplify_output(matching_ingredients)
                else:
                    print(f"[Debug] No matching ingredients for {allergy} in {best_match}")

            # Debugging output if no match found
            if not potential_allergies:
                print(f"[Debug] No allergies matched for {best_match}.")
        
        return potential_allergies
    
    def find_closest_match(self, input_str, options, threshold=50):
        """Find the closest match from a list of options using fuzzy matching"""
        best_match = None
        best_ratio = 0
        for option in options:
            ratio = fuzz.ratio(input_str.lower(), option.lower())
            partial_ratio = fuzz.partial_ratio(input_str.lower(), option.lower())
            token_sort_ratio = fuzz.token_sort_ratio(input_str.lower(), option.lower())
            combined_ratio = (ratio + partial_ratio + token_sort_ratio) / 3
            if combined_ratio > best_ratio and combined_ratio >= threshold:
                best_match = option
                best_ratio = combined_ratio
        return best_match
    
    def get_ingredient_list(self, product_name: str):
        """Retrieve the ingredient list for a single product."""
        products = self.db_manager.get_products()
        # Normalize the product name by stripping spaces and converting to lowercase
        product_name = product_name.lower().strip()
        print(f"[Debug] Normalized product name: {product_name}")

        # Find the best match for the product name using fuzzy matching
        best_match = self.find_closest_match(product_name, list(products.keys()), threshold=60)
        print(f"[Debug] Best match for product: {best_match}")  # Print the best match

        # Initialize ingredients variable
        ingredients = None
        
        # If a match is found, get the ingredients
        if best_match:
            product_ingredients = products[best_match]['ingredients']  # Get the ingredients of the matched product
            print(f"[Debug] Ingredients for {best_match}: {product_ingredients}")
            ingredients = product_ingredients  # Store the ingredients
        
        # If no match is found, return None
        if not ingredients:
            print(f"[Debug] No matching products found for {product_name}.")
            return None

        return ingredients  # Return the list of ingredients for the matched product

class DatabaseManager:
    def __init__(self, db_path='ingredient_database.db'):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT UNIQUE, ingredients TEXT, category TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS allergies (id INTEGER PRIMARY KEY, type TEXT UNIQUE, ingredients TEXT)''')
            conn.commit()

    def add_product(self, name: str, ingredients: list, category: str):
        """Add product into the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO products (name, ingredients, category) VALUES (?, ?, ?)''', (name.lower(), json.dumps(ingredients), category))
            conn.commit()

    def add_allergy(self, allergy_type: str, ingredients: list):
        """Add allergy into the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO allergies (type, ingredients) VALUES (?, ?)''', (allergy_type.lower(), json.dumps(ingredients)))
            conn.commit()

    def get_products(self):
        """Get all products from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name, ingredients, category FROM products')
            return {row[0]: {"ingredients": json.loads(row[1]), "category": row[2]} for row in cursor.fetchall()}

    def get_allergies(self):
        """Get all allergies from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT type, ingredients FROM allergies')
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

def process_user_query(user_query: str):
    # Step 1: Use ChatGPT to classify the intent
    intent_classifier = ChatGPTIntentClassifier()
    category, extracted_info = intent_classifier.classify_intent(user_query)
    print(f"[193]Intent classified as: {category}")  # Print the intent for debugging
    print(f"[193]Extracted information: {extracted_info}")  # Print the extracted information

    # Step 2: Initialize AllergyChecker to access the database
    checker = AllergyChecker()

    # Step 3: Handle the query based on intent
    if "check allergies" in category.lower():
        print("Classified as checking allergies for a product.")
        
        # Force extracted_info to be a string and clean it properly
        if isinstance(extracted_info, list):
            # If it's a list, extract the first item and ensure it's a string
            product_name = extracted_info[0].strip()  # Taking the first item and cleaning it
        else:
            # If it's already a string, clean it directly
            product_name = extracted_info.strip()

        # Now ensure product_name is a clean string
        print(f"Extracted product name: {product_name}")
        allergies = checker.get_product_allergies(product_name)
        formatted_response = call_chatgpt_to_format(allergies, product_name)
        return formatted_response
    elif "find ingredient list" in category.lower():
        # For Category 2, handle multiple products
        print("Classified as finding ingredient list of product(s).")
        product_names = extracted_info  # This is now a list of products
        print(f"Extracted product names: {product_names}")
        
        response = []
        for product_name in product_names:
            product_name = product_name.lower().strip()  # Normalize product name
            ingredients = checker.get_ingredient_list(product_name)  # Get ingredient list
            if ingredients:
                response.append(f"The ingredients for {product_name} are: {', '.join(ingredients)}.")
            else:
                reply = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant working in Singapore's Sheng Siong Grocery Store."},
                        {"role": "user", "content": "Keep it short, simple and to the point. What ingredients are in this product:" + product_name}
                    ],
                    temperature=0.7
                )
                response.append(reply.choices[0].message.content)
        
        return "\n".join(response)  # Join all responses for multiple products
    else:
        # Handle general queries using ChatGPT
        print("Classified as a general question.")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant working in Singapore's Sheng Siong Grocery Store."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
    return response.choices[0].message.content

def call_chatgpt_to_format(response_data, query_info):
    """Helper function to call ChatGPT to format the response concisely."""
    # Create a prompt for ChatGPT to give a concise and direct response
    prompt = f"""
    Based on the following allergy information for the product(s), generate a short and direct message suitable for a customer service reply on an AI chatbot:

    Allergy Information: {response_data}

    Query Information: {query_info}

    The response should be clear, concise, and to the point, without unnecessary details, focusing only on the key information. Keep it professional but casual, like a quick reply in a messaging app.
    """

    # Make the ChatGPT call to format the response
    formatted_response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the latest model
        messages=[
            {"role": "system", "content": "You are a customer service representative at a grocery store."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return formatted_response.choices[0].message.content.strip()