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


@app.route('/node', methods=['GET'])
def search():
    node = request.args.get('node')

    result = dict(
        node=dict(url=f'browse?node={node}', title=f'Статья про {node}', text='На каждый {node} найдётся анти-{node}'),
        parents=[
            dict(
                url='browse?node={node}_parent_{i}',
                title='Статья про родителя {node} №{i}',
                head='Одной из абстракций {node} является следующий узел.')
            for i in range(10)],
        children=[
            dict(
                url='browse?node={node}_child_{i}',
                title='Статья про потомка {node} №{i}',
                head='Одним из частных случаев {node} является следующий узел.')
            for i in range(10)]
    )

    if query is not None:
        pass

    return render_template('search.html', search=search)



@app.route('/')
def index():
    return redirect(url_for('browse'))


app.run(host='0.0.0.0', port=8080, debug=True)
