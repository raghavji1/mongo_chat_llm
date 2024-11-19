import os
import datetime
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from mongoembedding import MongoDBEmbeddings
from get_embeddings import *

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Initialize MongoDB client and specify the database and collection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[os.getenv("DB_NAME")]
chat_history_collection = db["chat_history"]

# Initialize OpenAI client for LLM
client = OpenAI()

# Initialize embedding handler for property data retrieval
embedding_handler = MongoDBEmbeddings(
    db_name=os.getenv("DB_NAME"), 
    collection_name=os.getenv("COLLECTION_NAME"), 
    mongo_uri=os.getenv("MONGO_URI")
)

# Route for health check
@app.route('/', methods=['GET'])
def check():
    """API route to check if the service is up and running."""
    return jsonify({"status": "Service is running"}), 200

def save_message_to_mongo(session_id, user_content, ai_content, email, collection):
    """
    Save a message in MongoDB in the following format:
    - Adds the message as a new entry in the chat history.
    - Upserts the document if it doesn’t exist for a given email.
    """
    message = {
        "req": user_content,
        "res": ai_content,
        "timestamp": datetime.datetime.utcnow()
    }
    collection.update_one(
        {"email": email},
        {"$push": {f"history.{session_id}.chat": message}},
        upsert=True
    )

def fetch_chat_history(email, session_id, limit=10):
    """
    Retrieve the last 'limit' messages from MongoDB for a given email and session_id.
    - This enables the LLM to maintain continuity in the conversation.
    """
    record = chat_history_collection.find_one({"email": email})
    if record:
        session_chat = record.get("history", {}).get(session_id, {}).get("chat", [])
        # Return only the last 'limit' number of messages for brevity in memory
        return session_chat[-limit:]
    return []

def get_query_results(user_input, limit=5):
    """
    Fetches relevant property recommendations from MongoDB based on the user input.
    - Generates an embedding for the user input to use in vector search.
    - Runs a MongoDB aggregate pipeline with vector search for relevant results.
    """
    query_embedding = embedding_handler.generate_embedding(user_input)
    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv('INDEX_NAME'),
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1
            }
        }
    ]
    collection = embedding_handler.db[embedding_handler.get_collection_name()]
    results = collection.aggregate(pipeline)
    return list(results)

def generate_answer(user_input, email, session_id):
    """
    Generate an AI answer for the user input.
    - Fetches recent chat history from MongoDB to provide context.
    - Retrieves property data based on the user query using embeddings.
    - Combines memory and property context to create a prompt for the LLM.
    """
    
    # Fetch previous chat history for context
    chat_history = fetch_chat_history(email, session_id)
    memory_context = "\n".join([f"User: {msg['req']}\nAssistant: {msg['res']}" for msg in chat_history])
    
    # Fetch property recommendations based on the current user query
    property_context = get_query_results(user_input)
    
    # Format the prompt to include chat history and property data for LLM response
    prompt = f"""Given the following memory context:\n{memory_context}\n
    And the following property context: {property_context}\n
    You are a Property Recommender chatbot, a professional yet friendly AI assistant:
    
    **Mission:** Guide the user with property-related inquiries. If interested in buying, ask about location, budget, and property type.
    **Tone:** Friendly and professional, like JARVIS from Iron Man.
    
    **Response Rules:**
    1. You have all the customer search history in the search history you will ananyse based on his search histoy what kind of property he is interested in. In history, you will get all the things like, place, price, bedrooms, location etc. for the further question and answering
    2. When you start chatting, you will ask multiple questions to user (maximum 6) like:
        i. location: in which location he want to search also the area of location
        ii. budget: what will be the budget of the user,
        iii. bedroom: how many bedroom he need to have 
        iv. amenities: What type of amenities he want nearby like school, college or anything?
        v. Reason: why he is buying this property for .... or for .....
    aftter succesfully gathering all the information about the clinet requirements, you will give the properties are listed in these criteria
    
    Example Interactions:
    User: "Hello"
    Assistant: Hello, how can I assist you?

    User: "Can you suggest properties in Indore?"
    Assistant: Sure, can you specify a preferred area within Indore?
    """
    
    # Generate response from OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
    )
    assistant_response = completion.choices[0].message.content
    
    # Save the generated answer to MongoDB chat history
    save_message_to_mongo(session_id, user_input, assistant_response, email, chat_history_collection)

    return assistant_response
@app.route('/', methods=['POST'])
def welcome():
    return "welcome to the chatbot api"


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat route for user interaction.
    - Accepts JSON input with 'session_id', 'email', and 'question'.
    - Generates a response considering the conversation history and property data.
    - Returns the generated response as JSON.
    """
    data = request.json
    session_id = data.get("session_id")
    email = data.get("email")
    question = data.get("question")
    
    # Validate required parameters
    if not session_id or not email or not question:
        return jsonify({"error": "session_id, email, and question are required fields."}), 400
    
    # Generate response from LLM with memory and property context
    response = generate_answer(question, email, session_id)
    return jsonify({"response": response})



@app.route('/embed', methods=['POST'])
def embed_collection():
    # Get the collection name from the request JSON
    data = request.get_json()
    collection_name = data.get("collection_name")

    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400

    # Generate and save embeddings for the specified collection
    result = generate_and_save_embeddings(collection_name)
    
    return jsonify({"message": result})





# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
