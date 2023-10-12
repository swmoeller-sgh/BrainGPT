# app.py

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    name = "John"
    
    if request.method == 'POST':
        name = request.form['name']

    return render_template('custom_content.html', name=name)


app.run(host="127.0.0.1",  # type: ignore
        port=4450,
        debug=True)