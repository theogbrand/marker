# sending
import requests

url = 'http://example.com/api'  # Replace with your API endpoint
file_path = '/path/to/your/file.pdf'  # Replace with your file path

# Open the file in binary mode
with open(file_path, 'rb') as f:
    # Define the multipart form data dictionary
    files = {'file': f}

    # Send the request
    response = requests.post(url, files=files)

# Print the response
print(response.text)


json_response = {
    "content": "base64_encoded_pdf_content",
    "contentType": "application/pdf",
    "fileName": "the_name_of_the_file.pdf"
}

# receiving
import requests
import json
import base64

# Make the API request
response = requests.get('http://example.com/api')  # Replace with your API endpoint

# Parse the JSON response
data = response.json()

# The PDF content is likely base64 encoded, so we need to decode it
pdf_content = base64.b64decode(data['content'])

# Write the decoded PDF content to a file
with open('output.pdf', 'wb') as f:
    f.write(pdf_content)