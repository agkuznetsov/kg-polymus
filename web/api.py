from flask import Blueprint, request

bp = Blueprint('api', __name__, url_prefix='/api')


@bp.route('find_node', methods=('GET', 'POST'))
def find_node():
    """ Find node by text """
    data = request.json
    return data


@bp.route('find_similar_nodes', methods=('GET', 'POST'))
def find_similar_nodes():
    """ Find similar nodes """

    raise NotImplemented()


@bp.route('get_node_content', methods=('GET', 'POST'))
def get_node_content():
    """ Return node content """
    raise NotImplemented()


@bp.route('get_node_downstream_links', methods=('GET', 'POST'))
def get_node_downstream_links():
    """ Return new topics to travel """
    raise NotImplemented()


@bp.route('get_node_upstream_links', methods=('GET', 'POST'))
def get_node_upstream_links():
    """ Return top level topics """
    raise NotImplemented()
