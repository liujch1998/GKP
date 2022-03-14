from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('config.yml')
OPENAI_API_KEY = ''
try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')

NUMERSENSE_ANSWERS = ['no', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
DATASET_TO_QUERY_KEY = {
    'numersense': 'query',
    'CSQA': 'query',
    'CSQA2': 'query',
    'qasc': 'query',
}
