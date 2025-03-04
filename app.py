from flask import Flask, render_template, request
import ChatBot

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        userQuery = request.json["message"]
        

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888, debug=True)