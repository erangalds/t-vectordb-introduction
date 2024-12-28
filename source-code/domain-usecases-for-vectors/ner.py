from transformers import pipeline

# Initialize the NER pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Example emails
emails = [
    "Dear John Doe, Thank you for purchasing the UltraWidget 3000. Your Customer ID is 12345. We hope you are enjoying the product in Canada. Best regards, The UltraWidget Team",
    "Hi Jane Smith, We're thrilled you chose to try out the MegaGadget 2.0. Your Customer ID is 67890. We appreciate your business in Australia. Best, The MegaGadget Crew",
    "Hello Michael Brown, Your new SuperTool X has been shipped! Your Customer ID is 13579. We look forward to your feedback from the USA. Warm wishes, The SuperTool Team",
    "Greetings Linda Green, Thank you for your order of the PowerDevice Pro. Your Customer ID is 24680. Enjoy your new purchase in the United Kingdom. Cheers, The PowerDevice Support",
    "Dear Robert White, We're excited you've chosen the HyperWidget Z for your needs. Your Customer ID is 11223. Happy shopping in New Zealand. Regards, The HyperWidget Team",
]

# Function to extract entities
def extract_entities(email, ner_model):
    entities = ner_model(email)
    product_name = ""
    customer_id = ""
    customer_name = ""
    country = ""
    
    for entity in entities:
        if entity['entity'] == 'B-ORG' or entity['entity'] == 'I-ORG':
            product_name += email[entity['start']:entity['end']] + " "
        elif entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER':
            customer_name += email[entity['start']:entity['end']] + " "
        elif entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC':
            country += email[entity['start']:entity['end']] + " "
        elif entity['entity'] == 'B-MISC':
            customer_id += email[entity['start']:entity['end']] + " "
    
    return {
        "Product Name": product_name.strip(),
        "Customer ID": customer_id.strip(),
        "Customer Name": customer_name.strip(),
        "Country": country.strip(),
    }

# Extract entities from emails
for email in emails:
    extracted_info = extract_entities(email, ner)
    print(extracted_info)
