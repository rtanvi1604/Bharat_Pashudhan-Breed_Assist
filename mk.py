# NEW CODE: Save the class indices to a JSON file
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved as 'class_indices.json'")