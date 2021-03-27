from flask import Flask, render_template, url_for, redirect

from web import api

app = Flask(__name__)
app.register_blueprint(api.bp)


@app.route('/search')
def search():
    return render_template('search.html')


@app.route('/browse')
def browser():
    return render_template('browse.html')


@app.route('/')
def index():
    return redirect(url_for('search'))


app.run(host='0.0.0.0', port=8080, debug=True)
