import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse_arff_file(filename):
    """
    Parse ARFF file and return features and labels
    """
    features = []
    labels = []
    feature_names = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Parse attributes
    in_data_section = False
    for line in lines:
        line = line.strip()
        
        if line.startswith('@attribute'):
            # Extract attribute name and values
            match = re.match(r'@attribute\s+(\w+)\s+\{([^}]+)\}', line)
            if match:
                attr_name = match.group(1)
                attr_values = match.group(2).split(',')
                feature_names.append(attr_name)
        
        elif line == '@data':
            in_data_section = True
            continue
        
        elif in_data_section and line and not line.startswith('%'):
            # Parse data line
            values = line.split(',')
            if len(values) == len(feature_names):
                # Last column is the label
                feature_values = [int(x) for x in values[:-1]]
                label = int(values[-1])
                
                features.append(feature_values)
                labels.append(label)
    
    return np.array(features), np.array(labels), feature_names

def get_feature_description():
    """
    Return descriptions of the features in the dataset
    """
    feature_descriptions = {
        'having_IP_Address': 'Whether the URL has an IP address',
        'URL_Length': 'Length of the URL (1=long, 0=medium, -1=short)',
        'Shortining_Service': 'Whether URL uses shortening service',
        'having_At_Symbol': 'Whether URL contains @ symbol',
        'double_slash_redirecting': 'Whether URL has double slash redirecting',
        'Prefix_Suffix': 'Whether URL has prefix/suffix',
        'having_Sub_Domain': 'Number of subdomains (-1=none, 0=1, 1=multiple)',
        'SSLfinal_State': 'SSL certificate state',
        'Domain_registeration_length': 'Domain registration length',
        'Favicon': 'Whether favicon is present',
        'port': 'Whether non-standard port is used',
        'HTTPS_token': 'Whether HTTPS token is present',
        'Request_URL': 'Whether request URL is present',
        'URL_of_Anchor': 'URL of anchor analysis',
        'Links_in_tags': 'Links in tags analysis',
        'SFH': 'Server Form Handler analysis',
        'Submitting_to_email': 'Whether form submits to email',
        'Abnormal_URL': 'Whether URL is abnormal',
        'Redirect': 'Number of redirects',
        'on_mouseover': 'Whether onmouseover is used',
        'RightClick': 'Whether right-click is disabled',
        'popUpWidnow': 'Whether popup window is used',
        'Iframe': 'Whether iframe is used',
        'age_of_domain': 'Age of domain',
        'DNSRecord': 'DNS record analysis',
        'web_traffic': 'Web traffic analysis',
        'Page_Rank': 'Page rank analysis',
        'Google_Index': 'Whether indexed by Google',
        'Links_pointing_to_page': 'Links pointing to page',
        'Statistical_report': 'Statistical report analysis'
    }
    return feature_descriptions

if __name__ == "__main__":
    # Test the parser
    features, labels, feature_names = parse_arff_file('Training Dataset.arff')
    print(f"Dataset shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(features)}")
    print(f"Label distribution: {np.bincount(labels + 1)}")  # +1 to convert -1,1 to 0,1
    print(f"Feature names: {feature_names[:5]}...")  # First 5 features
