import json

import gensim
import nltk

from text_cleaner import remove_formatting, reporter, fix_punctuation, FormattingHandling


# Cleans the text of an issue
def clean_issue_text(text: str, key: str, formatting_handling) -> list[str]:
    text = fix_punctuation(remove_formatting(text, key, formatting_handling))
    sentences = nltk.tokenize.sent_tokenize(text)
    # text = '. '.join([' '.join(list(gensim.utils.tokenize(sentence))) for sentence in sentences])
    return [f"{' '.join(gensim.utils.tokenize(sent))}" for sent in sentences]


def get_one_hot_encoding(items: list):
    item_to_idx = {}
    idx = 0
    for item in items:
        if item not in item_to_idx.keys():
            item_to_idx[item] = idx
            idx += 1

    item_to_onehot = {}
    for key, value in item_to_idx.items():
        one_hot_encoding = [0] * idx
        one_hot_encoding[value] = 1
        item_to_onehot[key] = one_hot_encoding

    return item_to_onehot


def get_encodings(issues, keys):
    encodings = {}
    for key in keys:
        values = []
        for issue in issues:
            if type(issue[key]) is list:
                for value in issue[key]:
                    values.append(value)
            else:
                values.append(issue[key])
        value_to_idx = {}
        idx = 0
        for value in values:
            if value not in value_to_idx.keys():
                value_to_idx[value] = idx
                idx += 1
        encodings[key] = value_to_idx
    return encodings


def assign_labels(issues, label_filename):
    with open(label_filename) as file:
        label_list = json.load(file)

    # Convert label list to a dictionary
    labels = dict()
    for label in label_list:
        labels[label['key']] = label

    # Assign labels to each issue
    for issue in issues:
        issue |= labels[issue['key']]


def create_json(input_filename, output_filename, label_filename, study, encodings):
    with open(input_filename) as file:
        raw_issues = json.load(file)

    issues = []
    for raw_issue in raw_issues:
        issue = {
            'key': raw_issue['key'],
            'summary': clean_issue_text(raw_issue['summary'], raw_issue['key'], FormattingHandling.Markers),
            'description': clean_issue_text(raw_issue['description'], raw_issue['key'], FormattingHandling.Markers),
            'study': study
        }

        comments_len = 0
        for comment in raw_issue['comments']:
            comments_len += len(remove_formatting(str(comment), issue['key'], FormattingHandling.Markers).split())
        summary_len = len(remove_formatting(raw_issue['summary'], issue['key'], FormattingHandling.Markers).split())
        description_len = len(remove_formatting(raw_issue['description'], issue['key'], FormattingHandling.Markers).split())

        metadata = {
            'n_attachments': [raw_issue['n_attachments']],
            'n_comments': [raw_issue['n_comments']],
            'len_comments': [comments_len],
            'n_components': [raw_issue['n_components']],
            'len_description': [description_len],
            'n_issuelinks': [raw_issue['n_issuelinks']],
            'n_labels': [raw_issue['n_labels']],
            'parent': [raw_issue['parent']],
            'n_subtasks': [raw_issue['n_subtasks']],
            'len_summary': [summary_len],
            'n_votes': [raw_issue['n_votes']],
            'n_watches': [raw_issue['n_watches']]
        }

        for key in ['components', 'issuetype', 'labels', 'priority', 'resolution', 'status']:
            value_to_idx = encodings[key]
            vector = [0] * len(value_to_idx.keys())
            if type(raw_issue[key]) is list:
                for value in raw_issue[key]:
                    idx = value_to_idx[value]
                    vector[idx] = 1
            else:
                idx = value_to_idx[raw_issue[key]]
                vector[idx] = 1
            metadata[key] = vector

        issue['metadata'] = metadata

        issues.append(issue)

    assign_labels(issues, label_filename)

    with open(output_filename, 'w') as file:
        json.dump(issues, file, indent=2)


def main():
    with open('../data/issuedata/EBSE_issues_raw.json') as file:
        issues = json.load(file)
    with open('../data/issuedata/BHAT_issues_raw.json') as file:
        issues += json.load(file)
    encodings = get_encodings(issues, ['components', 'issuetype', 'labels', 'priority', 'resolution', 'status'])

    create_json('../data/issuedata/EBSE_issues_raw.json',
                '../data/issuedata/EBSE_issues_formatting-markers.json',
                '../data/labels/EBSE_labels.json',
                'EBSE',
                encodings)
    create_json('../data/issuedata/BHAT_issues_raw.json',
                '../data/issuedata/BHAT_issues_formatting-markers.json',
                '../data/labels/BHAT_labels.json',
                'BHAT',
                encodings)
    # reporter.print_report()


if __name__ == '__main__':
    main()
