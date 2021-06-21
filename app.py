from flask import Flask, url_for, render_template, request
import json
import main

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/home', methods=['POST'])
def home():
    text = format(request.form['paragraph_text'])
    data = main.execute(text)
    #with open('data/predictions.json', 'rb') as file:
    #    data = json.load(file)

    return render_template("home.html", data=data)


if __name__ == '__main__':
    app.run()
