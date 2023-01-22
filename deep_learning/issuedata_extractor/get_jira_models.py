import nltk

import pandas as pd

from jira import JIRA
from gensim.models import Word2Vec as GensimWord2Vec
from gensim.models.doc2vec import Doc2Vec as GensimDoc2Vec
from gensim.models.doc2vec import TaggedDocument

import config

from text_cleaner import remove_formatting
from issuedata_extractor import get_issue_var
from deep_learning.dl_manager.feature_generators.util.ontology import load_ontology, apply_ontologies_to_sentence
from preprocess_to_json import clean_issue_text
from text_cleaner import FormattingHandling

APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'
POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


# Read csv items into a dict
def read_csv(path):
    dt = pd.read_csv(path, index_col=0, skiprows=0).T.to_dict()
    return dt


def main():
    # Authenticate with the jira server
    jira = JIRA(APACHE_JIRA_SERVER, basic_auth=(config.username, config.password))

    # Obtain issues from all projects
    issues = []
    previous_len = 0
    for project in ['HADOOP', 'CASSANDRA', 'TAJO', 'HDFS', 'MAPREDUCE', 'YARN']:
        print(f'searching issues {project}')
        next_issues = jira.search_issues(f'project={project} order by key desc', maxResults=1000,
                                         fields="key, summary, description")
        issues.extend(next_issues)

        while True:
            limit = issues[-1]
            next_issues = jira.search_issues(f'project={project} and key < {limit} order by key desc', maxResults=1000,
                                             fields="key, summary, description")
            if len(next_issues) == 0:
                break
            issues.extend(next_issues)

        print(f'# issues for project {project}:', len(issues) - previous_len)
        previous_len = len(issues)

    # # Read the issue keys
    # keys = []
    # with open('../data/labels/EBSE_labels.json') as file:
    #     labels = json.load(file)
    # for label in labels:
    #     keys.append(label['key'])
    #
    # issues = []
    # while len(keys) > 100:
    #     key_str = ','.join(keys[0:100])
    #     keys = keys[100:]
    #     issues.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields='key, summary, description'))
    # key_str = ','.join(keys)
    # issues.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields='key, summary, description'))

    json_list = []
    # ontology_table = load_ontology('../dl_manager/feature_generators/util/ontologies.json')
    # ontology_table = load_ontology('../dl_manager/feature_generators/util/ontologies_lexical.json')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # stemmer = nltk.stem.PorterStemmer()
    use_pos = False

    for issue in issues:
        # Create a dict and store it in the json list
        fields = issue.fields
        formatting = FormattingHandling.Markers
        summary = clean_issue_text(get_issue_var(fields, ['summary'], str), '', formatting)
        description = clean_issue_text(get_issue_var(fields, ['description'], str), '', formatting)
        text = summary + description

        for sentence in text:
            words = nltk.word_tokenize(sentence)

            # Transform lowercase
            words = [word.lower() for word in words]

            # words = apply_ontologies_to_sentence(words, ontology_table)

            words = nltk.pos_tag(words)

            # Remove stopwords
            stopwords = nltk.corpus.stopwords.words('english')
            words = [(word, tag) for word, tag in words if word not in stopwords]

            words = [(lemmatizer.lemmatize(word, pos=POS_CONVERSION.get(tag, 'n')), tag)
                     for word, tag in words]

            if use_pos:
                words = [f'{word}_{POS_CONVERSION.get(tag, tag)}' for word, tag in words]
            else:
                words = [word for word, _ in words]
            # if setting == 'stemming':
            #     words = [stemmer.stem(word) for word in words]

            json_list.append(words)

    for vector_size in [10, 25, 100, 300]:
        min_count = 5
        if vector_size != 100:
            # Train word2vec model
            model = GensimWord2Vec(json_list, min_count=min_count, vector_size=vector_size)
            filename = f'../embeddings/bhat/w2v_vector-size-{str(vector_size)}_markers_no-ontology_pos-{use_pos}.bin'
            model.wv.save_word2vec_format(filename, binary=True)

        if vector_size in [25, 100]:
            # Train doc2vec model
            documents = []
            for idx in range(len(json_list)):
                documents.append(TaggedDocument(json_list[idx], [idx]))

            doc2vec_model = GensimDoc2Vec(documents, min_count=min_count, vector_size=vector_size)
            filename = f'../embeddings/bhat/d2v_vector-size-{str(vector_size)}_markers_no-ontology_pos-{use_pos}.bin'
            doc2vec_model.save(filename)


if __name__ == '__main__':
    main()
