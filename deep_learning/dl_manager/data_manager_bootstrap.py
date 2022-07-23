import json
import os

from .config import conf

##############################################################################
##############################################################################
# Raw text
##############################################################################


def get_raw_text_file_name() -> str:
    if conf.get('system.peregrine'):
        path = os.path.join(conf.get('system.peregrine.data'), 'raw_words')
    else:
        path = 'raw_words'
    os.makedirs(path, exist_ok=True)
    options = []
    params = conf.get('make-features.params')
    keys = ('max-len',
            'disable-lowercase',
            'disable-stopwords',
            'disable-stemmnig',
            'use-lemmatization',
            'use-pos', )
    for key in keys:
        if key in params['default']:
            options.append(f'{key}_{params["default"][key]}')
    if conf.get('make-features.ontology-classes'):
        options.append('with_ontologies')
    options.sort()
    filename = f'raw_words_{"__".join(options)}.json'
    return filename
