from llama_index.chat_engine import SimpleChatEngine
from llama_index.llms import Gemini
from llama_index import ServiceContext
from llama_index.embeddings import GeminiEmbedding
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

GAK = os.getenv("GEMINI_API")

llm = Gemini(api_key=GAK)
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GAK)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

app = Flask(__name__)
CORS(app) 
chat_engine = SimpleChatEngine.from_defaults(service_context=service_context)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data["prompt"]
    if prompt:
        prompt = data['prompt']
        response = jsonify(chat_engine.chat(prompt))
        response_data = response.get_json()
        if response_data:
            response_data = remove_asterisks(response_data)
            response = jsonify(response_data)
        return response
    topic = data['topic']
    time = data['time']
    level = data['level']
    addComm = data['addComm']
    prompt = f"""You are a excellent in the {topic}. Now a person wants to learn the same topic in {time}.
        He/She has {level} level of knowledge in the field. Now generate a roadmap with references and tags (tags must be related to the topic) for the person.
        Show topics in details and all tags should start with '#' only and in an array format, Like [#john, #doe].
        If the time is too low (like in minutes or seconds, then reply appropriately (No if not possible))."""
    if addComm:
        prompt = prompt + "Additional Notes by user: " + addComm
    response = chat_engine.chat(prompt)
    response_data = remove_asterisks(response)
    print(response_data)
    return jsonify(response_data)

def remove_asterisks(response):
    if isinstance(response, dict):
        return {key: remove_asterisks(value) for key, value in response.items()}
    elif isinstance(response, list):
        return [remove_asterisks(item) for item in response]
    elif isinstance(response, str):
        return response.replace("*", "")
    else:
        return response

@app.route("/chat", methods=["DELETE"])
def delete():
    chat_engine.clear_chat()
    return jsonify({"message": "Chat history deleted."})

if(__name__ == '__main__'):
    app.run(debug=True)