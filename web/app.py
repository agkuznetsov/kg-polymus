from flask import Flask, render_template

from web import api

app = Flask(__name__)
app.register_blueprint(api.bp)


@app.route('/')
def index():
    return render_template('index.html')


app.run(host='0.0.0.0', port=8080, debug=True)
