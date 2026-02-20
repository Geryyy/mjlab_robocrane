import xml.etree.ElementTree as ET
import re

# Load XML file
tree = ET.parse('./robocrane.xml')
root = tree.getroot()

def round_sig(x, sig=4):
    try:
        return format(float(x), f'.{sig}g')
    except ValueError:
        return x  # Keep as-is if not a float

# Pattern to match whitespace-separated numbers
number_pattern = re.compile(r'^-?\d+(\.\d+)?(e[+-]?\d+)?$')

def process_attr(val):
    # Round each number if it's a float
    return ' '.join(round_sig(x) if number_pattern.match(x) else x for x in val.strip().split())

# Iterate through all XML elements and attributes
for elem in root.iter():
    for attr in elem.attrib:
        if any(c.isdigit() for c in elem.attrib[attr]):
            elem.attrib[attr] = process_attr(elem.attrib[attr])

# Write output
tree.write('./robocrane_simplified.xml')
