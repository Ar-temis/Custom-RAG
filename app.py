from flask import Flask, render_template, request, jsonify
from ChatBot import ChatBot

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        userQuery = str(request.json.get('message'))
        response = ChatBot.askBot(userQuery)
        # print(response)
        return jsonify({'message': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888, debug=True)
