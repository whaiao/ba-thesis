import os

DATA_ROOT = os.getcwd() + '/data'
PNAME_PLACEHOLDER_RE = ['Person\s?X', 'Person\s?Y', 'Person\s?Z']
PNAME_SUB = ['Peter', 'Shannon', 'Clara', 'Jacob', 'Sandra', 'Nick']
T5_TURN_TEMPLATES = {
    1: 'inform: ',
    2: 'ask a question: ',
    3: 'make a suggestion: ',
    4: 'accept a suggestion: '
}
