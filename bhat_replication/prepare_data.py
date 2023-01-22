import csv
import json

# ID, summary, description, is-design, design-type

with open('../deep_learning/data/issuedata/EBSE_issues_raw.json') as file:
    issues = json.load(file)

EBSE_labels = {}
with open('../deep_learning/data/labels/EBSE_labels.json') as file:
    data = json.load(file)
for label in data:
    category = 'N/A'
    if label['is-cat1']['value'] == 'True':
        category = 'Existence'
    if label['is-cat3']['value'] == 'True':
        category = 'Property'
    if label['is-cat2']['value'] == 'True':
        category = 'Executive'
    EBSE_labels[label['key']] = (label['is-design'], category)

with open('./ebse_issues_raw.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(['ID', 'Summary', 'Description', 'Is-Design', 'Design-Type'])
    for issue in issues:
        label = EBSE_labels[issue['key']]
        writer.writerow([
            issue['key'],
            issue['summary'],
            issue['description'],
            label[0],
            label[1]
        ])
