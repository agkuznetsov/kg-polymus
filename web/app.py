from flask import Flask, render_template, url_for, redirect, request

from web import api

app = Flask(__name__)
app.register_blueprint(api.bp)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')

    search = dict(
        query=query,
        results = [
            dict(url='browse?node=python', title='My large python'),
            dict(url='browse?node=groovy', title='My groovy groovy'),
        ]
    )

    if query is not None:
        pass

    return render_template('search.html', search=search)


@app.route('/browse', methods=['GET'])
def browse():
    node = request.args.get('node')

    return render_template('browse.html', node=node)


@app.route('/')
def index():
    return redirect(url_for('browse'))


app.run(host='0.0.0.0', port=8080, debug=True)
