from pymongo import MongoClient
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class MongoDBEmbeddings:
    def __init__(self, db_name, collection_name, mongo_uri=None):
        # Set up MongoDB connection
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[os.getenv("DB_NAME")]
        self.collection = self.db[os.getenv("COLLECTION_NAME")]

        # Set up OpenAI client and API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client_openai = OpenAI(api_key=self.api_key)

    def get_collection_name(self):
        """Retrieve the name of the current MongoDB collection."""
        return self.collection.name

    def fetch_data(self):
        """Fetch data from MongoDB and load it into a DataFrame."""
        data = list(self.collection.find())
        df = pd.DataFrame(data)
        return df

    def generate_embedding(self, text):
        """Generate embedding for a given text using OpenAI's API."""
        response = self.client_openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return embedding

    def embedd_and_update_mongo(self, df):
        """Generate embeddings for DataFrame rows and update MongoDB."""
        try:
            # Apply embeddings to rows that don't already have one
            df['embedding'] = df.apply(
                lambda row: row['embedding'] if row.get('embedding') else self.generate_embedding(' '.join(map(str, row.values))),
                axis=1
            )   
            # Update MongoDB with the new embeddings
            for _, row in df.iterrows():
                self.collection.update_one(
                    {'_id': row['_id']},
                    {'$set': {'embedding': row['embedding']}}
                )
            print("Embeddings generated and updated in MongoDB successfully.")
        except Exception as e:
            print(f"Error generating embeddings: {e}")

# Usage example
# if __name__ == "__main__":
#     mongo_uri = "your_mongo_uri"  # Optionally, set your MongoDB URI here
#     openai_api_key = "your_openai_api_key"  # Optionally, set your OpenAI API key here

#     embedding_handler = MongoDBEmbeddings(db_name="propertyai", collection_name="properties", mongo_uri=mongo_uri)
#     df = embedding_handler.fetch_data()
#     embedding_handler.embedd_and_update_mongo(df)
