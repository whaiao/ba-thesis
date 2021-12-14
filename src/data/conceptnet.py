import requests

Response = requests.Response


def make_concept_request(concept: str, /, relation_to: str = None, query_on: str = None) -> Response:
    """Creates HTTP request to `conceptnet.io`

    Args:
        concept: Concept to look up in concept net
        relation_to: Must be a relation type from `['start', 'end', 'rel', 'node', 'other', 'sources']`
        query_on: query specific to relation

    Returns:
        dict: response body given concept

    Raises:
        RequestException: if anything goes wrong while requesting string


    """
    possible_relations = ['start', 'end', 'rel', 'node', 'other', 'sources']
    request_str = f'/c/en/{concept}'

    if relation_to is not None and query_on is not None:
        assert relation_to in possible_relations, f'Relation not in {possible_relations}.'
        request_str = f'/query?node={request_str}&{relation_to}={query_on}'

    try:
        response = requests.get(f'http://api.conceptnet.io{request_str}')
            
    except requests.RequestException as e:
        raise(e)

    return response.json()


