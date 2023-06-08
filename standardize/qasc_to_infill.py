prefix = '''Input: Reproduction is the process by which living things what?
Output: Reproduction is the process by which living things _.

Input: What can cause rocks to break down?
Output: _ can cause rocks to break down.

Input: Removing what from food will preserve it?
Output: Removing _ from food will preserve it.

Input: Which organ helps break down food into nutrients for our bodies?
Output: _ helps break down food into nutrients for our bodies.

Input: Who pollinates?
Output: _ pollinates.

Input: When does water expand?
Output: Water expands _.

Input: Where do platypus females lay their eggs?
Output: Platypus females lay their eggs _.

Input: How do you feel heat on your skin?
Output: You feel heat on your skin by _.

Input: Heat can change something that has how many states?
Output: Heat can change something that has _ states.

Input: What kind of animals has a water vascular system with tubed feet?
Output: _ has a water vascular system with tubed feet.

Input: What type of water formation is formed by clouds?
Output: _ is formed by clouds.

Input: What do most mussles have?
Output: Most mussles have _.

Input: What does the hemisphere have to be tilted away from to receive less sunlight?
Output: The hemisphere has to be tilted away from _ to receive less sunlight.

Input: what can wood be used for?
Output: Wood can be used for _.

Input: What happens when there is an increase in heat energy?
Output: _ when there is an increase in heat energy.

Input: To learn more about the average weather, it is essential to:
Output: To learn more about the average weather, it is essential to _.

Input: a long tail can be used for _ by an animal
Output: A long tail can be used for _ by an animal.

Input: %s
Output:'''

import os,sys,time
import json
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.gpt3_generation import request

with open('../data/qasc/test.qasc.json') as f:
    ds = json.load(f)

for item in tqdm(ds):
    prompt = prefix % item['query']
    transformed_query = request(prompt, temperature=0.0)[0]
    if transformed_query.count('_') == 1:
        item['query'] = transformed_query.replace('_', '<extra_id_0>')
    else:
        print(item['query'])

with open('../data/qasc/test.qasc_infill.json', 'w') as f:
    json.dump(ds, f, indent=4)

