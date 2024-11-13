import os
from dotenv import load_dotenv
from openai import OpenAI
from mongoembedding import MongoDBEmbeddings

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize embedding handler for property data retrieval
embedding_handler = MongoDBEmbeddings(
    db_name=os.getenv("DB_NAME"), 
    collection_name=os.getenv("COLLECTION_NAME"), 
    mongo_uri=os.getenv("MONGO_URI")
)

# File path for conversation memory
MEMORY_FILE = "chat_memory.txt"

def get_query_results(user_input, limit=5):
    """Fetches property recommendations based on user input."""
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

    # Fetching results from MongoDB
    collection = embedding_handler.db[embedding_handler.get_collection_name()]
    results = collection.aggregate(pipeline)
    return results

def update_memory(user_input, assistant_response):
    """Append conversation history to memory file."""
    with open(MEMORY_FILE, "a") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Assistant: {assistant_response}\n")

def fetch_memory_context():
    """Retrieve the last few lines of memory for continuity."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory_lines = f.readlines()
            return "".join(memory_lines[-10:])  # Use the last 10 lines as context
    return ""

def generate_answer(user_input):
    """Generate an answer for user input, considering memory and database results."""
    
    # Fetch previous memory for context
    memory_context = fetch_memory_context()
    context = get_query_results(user_input)
    
    # Prepare the prompt with memory and context from property data
    prompt = f"""Given the following memory context:\n{memory_context}\n
    And the following property context: {context}\n
    You are Property Recommender chatbot, an advanced AI system with the following responsibilities:
    
    **Mission:** Assist the user by answering property-related inquiries. If the user expresses interest in buying, guide them by asking location, budget, and property type.

    **Tone:** Professional yet friendly, similar to JARVIS from Iron Man.

    **Response Rules:**
    1. Engage in natural conversation, responding to casual questions.
    2. For property inquiries, ask up to 5 questions on details like location, budget, area, and type.
    
    Example Interactions:
    User: "Hello"
    Assistant: Hello, how can I assist you?

    User: "Can you suggest properties in Indore?"
    Assistant: Sure, can you specify a preferred area within Indore?
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
    )
    assistant_response = completion.choices[0].message.content

    # Print the assistant's response
    print(assistant_response)
    
    # Update memory with current conversation
    update_memory(user_input, assistant_response)

# Start an interactive loop
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        generate_answer(user_input)
