from flask import Flask, request, jsonify
from chatbot import ChatBot
from latency_monitor import measure_latency

app = Flask(__name__)
chatbot = ChatBot()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data['query']
    
    response, latency = measure_latency(chatbot.get_response)(query)
    
    return jsonify({
        'response': response,
        'latency': latency
    })

if __name__ == '__main__':
    app.run(debug=True)