import json

class ResponseDatabase:
    def __init__(self, db_path='data/previous_responses.json'):
        self.db_path = db_path
        self.load_db()

    def load_db(self):
        try:
            with open(self.db_path, 'r') as file:
                self.responses = json.load(file)
        except FileNotFoundError:
            self.responses = {}

    def save_response(self, query, response):
        self.responses[query] = response
        with open(self.db_path, 'w') as file:
            json.dump(self.responses, file)

    def get_response(self, query):
        return self.responses.get(query, None)