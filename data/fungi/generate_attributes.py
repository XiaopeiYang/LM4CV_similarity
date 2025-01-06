import os
from openai import OpenAI
import json
import itertools

client = OpenAI(api_key='sk-proj-xkNCqlUVPb8KiWoVYHs7T3BlbkFJEE6IZlJKkJjr9d3e9019')  # Ensure you replace 'YOUR_API_KEY' with your actual API key

def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]

def generate_prompt(category_name: str):
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} under a microscope?
A: There are several useful visual features to tell there is a {category_name} under a microscope:
-
"""

def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def obtain_descriptors_and_save(filename, class_list):
    responses = []
    descriptors = {}

    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]

    for prompt_partition in partition(prompts, 20):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt} for prompt in prompt_partition],
            temperature=0.0,
            max_tokens=100,
        )
        responses.extend(response.choices)

    response_texts = [r.message.content for r in responses]

    # Logging responses for debugging
    for i, response_text in enumerate(response_texts):
        print(f"Response for {class_list[i]}: {response_text}")

    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

obtain_descriptors_and_save('example', ["Caspofungin", "Carbendazim", "Mycelium", "Germination Inhibition", "GWT1", "Tebuconazole"])

# sk-proj-xkNCqlUVPb8KiWoVYHs7T3BlbkFJEE6IZlJKkJjr9d3e9019
# export OPENAI_API_KEY = "sk-proj-xkNCqlUVPb8KiWoVYHs7T3BlbkFJEE6IZlJKkJjr9d3e9019"