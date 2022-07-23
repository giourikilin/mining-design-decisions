import json
import config
from jira import JIRA

APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'


# Get issue metadata (field)
def get_issue_var(fields, path, field_type):
    value = fields
    for step in path:
        if not hasattr(value, step):
            value = None
            break
        value = getattr(value, step)

    if value is None:
        if field_type is bool or field_type is int:
            return 0
        if field_type is list:
            return []
        if field_type is str:
            return ''
    if field_type is bool:
        return 1
    if field_type is list:
        if path == ['issuelinks'] or path == ['labels'] or path == ['subtasks']:
            return value
        return [item.name for item in value]
    return value


def main():
    # Authenticate with the jira server
    jira = JIRA(APACHE_JIRA_SERVER, basic_auth=(config.username, config.password))

    # Read the issue keys
    keys = []
    with open('../data/labels/EBSE_labels.json') as file:
        labels = json.load(file)
        for label in labels:
            keys.append(label['key'])

    # Request issues from the JIRA API
    fields = 'key, parent, summary, description, ' \
             'attachment, comment, issuelinks, ' \
             'issuetype, labels, priority, ' \
             'resolution, status, subtasks, ' \
             'votes, watches, components'

    issue_list = []
    while len(keys) > 100:
        key_str = ','.join(keys[0:100])
        keys = keys[100:]
        issue_list.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields=fields))
    key_str = ','.join(keys)
    issue_list.extend(jira.search_issues(f'key in ({key_str})', maxResults=1000, fields=fields))

    # # Obtain issues from all projects
    # issue_list = []
    # for project in ['HADOOP', 'CASSANDRA', 'TAJO', 'HDFS', 'MAPREDUCE', 'YARN', 'SPARK']:
    #     print(f'searching issues {project}')
    #     next_issues = jira.search_issues(f'project={project} order by key desc', maxResults=1000,
    #                                      fields=fields)
    #     issue_list.extend(next_issues)
    #
    #     while True:
    #         limit = issue_list[-1]
    #         next_issues = jira.search_issues(f'project={project} and key < {limit} order by key desc', maxResults=1000,
    #                                          fields=fields)
    #         if len(next_issues) == 0:
    #             break
    #         issue_list.extend(next_issues)

    # # Extract data from the issues
    json_list = []
    for issue in issue_list:
        fields = issue.fields

        comments = []
        if hasattr(fields, 'comment') and fields.comment is not None:
            comments = [comment.body for comment in fields.comment.comments]

        attachments = 0
        if hasattr(fields, 'attachment') and fields.attachment is not None:
            attachments = len(fields.attachment)

        dictionary = {
            'key': issue.key,
            'n_attachments': attachments,
            'n_comments': len(comments),
            'comments': comments,
            'n_components': len(get_issue_var(fields, ['components'], list)),
            'components': get_issue_var(fields, ['components'], list),
            'description': get_issue_var(fields, ['description'], str),
            'n_issuelinks': len(get_issue_var(fields, ['issuelinks'], list)),
            'issuetype': get_issue_var(fields, ['issuetype', 'name'], str),
            'n_labels': len(get_issue_var(fields, ['labels'], list)),
            'labels': get_issue_var(fields, ['labels'], list),
            'parent': get_issue_var(fields, ['parent'], bool),
            'priority': get_issue_var(fields, ['priority', 'name'], str),
            'resolution': get_issue_var(fields, ['resolution', 'name'], str),
            'status': get_issue_var(fields, ['status', 'name'], str),
            'n_subtasks': len(get_issue_var(fields, ['subtasks'], list)),
            'summary': get_issue_var(fields, ['summary'], str),
            'n_votes': get_issue_var(fields, ['votes', 'votes'], int),
            'n_watches': get_issue_var(fields, ['watches', 'watchCount'], int),
        }
        json_list.append(dictionary)

    with open('../data/issuedata/EBSE_issues_raw.json', 'w') as json_file:
        json.dump(json_list, json_file, indent=4)


if __name__ == '__main__':
    main()
