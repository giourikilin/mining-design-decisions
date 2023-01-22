# This is a quick-and-dirty script to download the Bhat dataset

import json
import re
import shlex
import typing
import os
import subprocess


class Issue(typing.NamedTuple):
    summary: str
    description: str
    id: str
    is_arch: bool
    arch_type: str


# Debugging variable, used to go through the program in steps,
# or to skip steps.
STEP = 0

if STEP == 0:
    # The basic link on where to find the dataset is given by:
    #
    # https://server.sociocortex.com/typeDefinitions/1vk4hqzziw3jp/Task
    #
    # However, the "download as CSV" option does not work. Hence,
    # we use the API exposed by the website. The basic API
    # route is https://server.sociocortex.com/api/v1/
    #
    # First, we can get a list of workspace IDs using
    # curl https://server.sociocortex.com/api/v1/workspaces
    #
    # This tells us that the Amelie workspace
    # (the workspace containing the dataset)
    # is located at
    # https://server.sociocortex.com/api/v1/workspaces/1iksmphpafkxq
    #
    # We can now get a list of entities belonging to the workspace
    # using
    # curl https://server.sociocortex.com/api/v1/workspaces/1iksmphpafkxq/entities
    #
    # The response of this requested is to be saved in todo.txt
    if not os.path.exists('todo.txt'):
        subprocess.run(
            shlex.split(
                'curl -k https://server.sociocortex.com/api/v1/workspaces/1iksmphpafkxq/entities > todo.txt'
            ),
            shell=True
        )
    STEP = 1


if STEP == 1:
    # todo.txt contains a list of all the entities in the Amelie workspace.
    # We now have to filter our all items in the dataset.
    # We initially trim the amount of entities down by looking for items
    # with "SPARK" or "HADOOP" in their name.
    # In the next step, we make sure these entity names all follow
    # the (SPARK|HADOOP)-\d+ pattern.
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
        info = subprocess.run(shlex.split(f'curl -k {url}'), stdout=subprocess.PIPE)
        return json.loads(info.stdout.decode())

    for entity in entities:
        response = __get(entity['href'])
        if 'attributes' not in response:
            continue
        # Check that the issue belongs to the correct workspace
        if response['entityType']['id'] != '1vk4hqzziw3jp':
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


