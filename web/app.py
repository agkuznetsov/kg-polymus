from flask import Flask, render_template, url_for, redirect, request

from web import api

import json

from web.graph import Graph

app = Flask(__name__)
app.register_blueprint(api.bp)

g = Graph('Polytech_total.graphml')


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')

    search = dict()

    if query is not None:
        g_results = g.similar(query)
        search = dict(
            query=query,
            results=[
                # dict(url='browse?node=python', title='My large python'),
                # dict(url='browse?node=groovy', title='My groovy groovy'),
                dict(url=f'browse?node={node}', title=title)
                for node, title, score in g_results
            ]
        )

    return render_template('search.html', search=search)


@app.route('/browse', methods=['GET'])
def browse():
    node = request.args.get('node')

    return render_template('browse.html', node=node)


@app.route('/node', methods=['GET'])
def node():
    _node = request.args.get('node')

    title = g.node_labels[_node]
    children = g.children(_node)
    parents = g.parents(_node)

    print(title)
    print(children)
    print(parents)

    result = dict(
        node=dict(url=f'browse?node={_node}', title=f'Статья про {title}', text=f"""
        На каждый {title} найдётся анти-{title}
        <br/>
        Python can be easy to pick up whether you're a first time programmer or you're experienced with other languages.
        The following pages are a useful first step to get on your way writing programs with Python!
        """),
        parents=[
            dict(
                url=f'browse?node={parent}',
                title=f'{g.node_labels[parent]}',
                head=f'{g.node_labels[parent]}')
            for parent in parents if parent in g.node_labels
        ],
        children=[
            dict(
                url=f'browse?node={child}',
                title=f'{g.node_labels[child]}',
                head=f'{g.node_labels[child]}')
            for child in children if child in g.node_labels]
    )

    response = app.response_class(
        response=json.dumps(result, ensure_ascii=False, indent=4),
        mimetype='application/json'
    )
    return response


@app.route('/')
def index():
    return redirect(url_for('browse'))


app.run(host='0.0.0.0', port=8080, debug=True)
