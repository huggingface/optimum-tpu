import os
import sys
import yaml

# Check that both files exist
examples_file = 'docs/scripts/examples_list.yml'
toctree_file = 'docs/source/_toctree.yml'

if not os.path.exists(examples_file):
    print(f"Error: {examples_file} does not exist")
    sys.exit(1)

if not os.path.exists(toctree_file):
    print(f"Error: {toctree_file} does not exist") 
    sys.exit(1)

# Read the examples list
with open(examples_file, 'r') as f:
    examples = yaml.safe_load(f)

# Read the main toctree
with open(toctree_file, 'r') as f:
    toc = yaml.safe_load(f)

# Find the howto section and insert before more_examples
# Iterate through the list to find the sections with howto
for item in toc:
    if isinstance(item, dict) and 'sections' in item:
        for section in item['sections']:
            if isinstance(section, dict) and 'sections' in section:
                howto_items = section['sections']
                for i, subitem in enumerate(howto_items):
                    if subitem.get('local') == 'howto/more_examples':
                        # Insert the new examples before this position
                        for example in reversed(examples):
                            howto_items.insert(i, example)
                        break

# Write back the modified toctree
with open(toctree_file, 'w') as f:
    yaml.dump(toc, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

print("Added examples to the howto section of the toctree")

# Print the updated toctree contents
with open(toctree_file, 'r') as f:
    print("\nUpdated _toctree.yml contents:")
    print(f.read())
