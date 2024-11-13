# get_embeddings.py

from pymongo import MongoClient
import pandas as pd
import openai

# MongoDB connection setup
def get_mongo_collection(collection_name):
    # Replace with your MongoDB connection string
    mongo_client = MongoClient("mongodb+srv://username:password@cluster0.mongodb.net/propertyai")
    db = mongo_client["propertyai"]
    collection = db[collection_name]
    return collection

# Load data from the specified collection into a DataFrame
def load_data(collection):
    data = list(collection.find())
    return pd.DataFrame(data)

# Generate embeddings for a given text
def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding

# Generate embeddings for each document in the DataFrame and update the collection
def generate_and_save_embeddings(collection_name):
    collection = get_mongo_collection(collection_name)
    df = load_data(collection)

    if df.empty:
        return "No documents found in the collection."

    # Generate embeddings only if 'embedding' field is None, else keep existing
    df['embedding'] = df.apply(
        lambda row: generate_embedding(' '.join(map(str, row.values))) 
                           if row['embedding'] is None else row['embedding'], 
                           axis=1
                           )

    # Update each document in MongoDB with the new embeddings
    for _, row in df.iterrows():
        collection.update_one({'_id': row['_id']}, {"$set": {"embedding": row['embedding']}})

    return "Embeddings generated and saved successfully where needed."
