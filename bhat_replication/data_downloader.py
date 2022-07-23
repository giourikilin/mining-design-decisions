import json
import re
import typing
import subprocess


class Issue(typing.NamedTuple):
    summary: str
    description: str
    id: str
    is_arch: bool
    arch_type: str

STEP = 1

if STEP == 1:
    with open('todo.txt') as file:
        entities = json.load(file)


    final = []
    for entity in entities:
        if 'SPARK' in entity['name'] or 'HADOOP' in entity['name']:
            final.append(entity)

    print(len(final))

    with open('reduced.txt', 'w') as file:
        json.dump(final, file)

    STEP = 2


if STEP == 2:
    with open('reduced.txt') as file:
        entities = json.load(file)

    pattern = re.compile(r'(SPARK|HADOOP)-\d+')
    seen = set()
    final = []

    for entity in entities:
        if not pattern.fullmatch(entity['name']):
            continue
        if entity['name'] in seen:
            continue
        seen.add(entity['name'])
        final.append(entity)

    print(len(final))

    with open('reduced2.txt', 'w') as file:
        json.dump(final, file)

    STEP = 3


if STEP == 3:
    with open('reduced2.txt') as file:
        entities = json.load(file)

    results = []

    def __remove_ws(x):
        import string
        return ''.join(
            filter(lambda y: y not in string.whitespace, x)
        )

    def find_field(data, attr):
        for row in data:
            if __remove_ws(row['name'].lower()) == attr:
                return row

    def __get(url):
        print(url)
        info = subprocess.run(f'curl {url}', stdout=subprocess.PIPE)
        return json.loads(info.stdout.decode())

    for entity in entities:
        response = __get(entity['href'])
        if 'attributes' not in response:
            continue
        attributes = response['attributes']
        summary = find_field(attributes, 'summary')
        description = find_field(attributes, 'description')
        binary_type = find_field(attributes, 'designdecision')
        binary_type = binary_type['values'][0] if binary_type['values'] else None
        if binary_type:
            exact_type = find_field(attributes, 'decisioncategory')
            exact_type = exact_type['values'][0]['name'] if exact_type['values'] else None
        else:
            exact_type = 'N/A'
        if summary is None or description is None or binary_type is None or exact_type is None:
            continue
        issue = Issue(
            summary='. '.join(summary['values']),
            description='. '.join(description['values']),
            id=entity['name'],
            is_arch=binary_type,
            arch_type=exact_type
        )
        results.append(issue)

    print(len(results))

    with open('issues.csv', 'w') as file:
        file.write('"id","summary","description","is-design","design-type"\n')
        for result in results:
            file.write(f'"{result.id}","{result.summary}","{result.description}",{result.is_arch},{result.arch_type}\n')

    STEP = 4


