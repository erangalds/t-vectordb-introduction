import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

# Example data with 50 examples
data = {
    "text": [
        "I am unable to receive any emails in my inbox.",
        "I cannot log into my account, it keeps saying the password is incorrect.",
        "My database connection fails every time I try to access it.",
        "The application crashes whenever I try to save a file.",
        "I can't access my email folders, they seem to have disappeared.",
        "My email is not syncing with the server.",
        "I forgot my password and can't reset it.",
        "Database connection timed out.",
        "App shuts down unexpectedly.",
        "I lost access to my emails.",
        "Emails are not being sent.",
        "Can't sign in with my username.",
        "Database query is taking too long.",
        "Application hangs on start.",
        "Email attachments are missing.",
        "Receiving error messages when sending emails.",
        "Unable to authenticate with my account.",
        "Database keeps disconnecting.",
        "App is not responding.",
        "Emails are disappearing from my inbox.",
        "Emails are bouncing back.",
        "Login screen freezes.",
        "Database is corrupted.",
        "App is crashing on startup.",
        "Cannot access email settings.",
        "Email filters are not working.",
        "Account lockout issues.",
        "Database migration failed.",
        "App is slow and unresponsive.",
        "Cannot find archived emails.",
        "Emails are getting delayed.",
        "Login attempts are being rejected.",
        "Database backup failed.",
        "App freezes during use.",
        "Email signature is not displaying correctly.",
        "Cannot recover deleted emails.",
        "Password reset link is not working.",
        "Database performance issues.",
        "App is not updating.",
        "Emails are being marked as spam.",
        "Two-factor authentication is failing.",
        "Database restore issues.",
        "App is not installing correctly.",
        "Cannot move emails to folders.",
        "Receiving duplicate emails.",
        "Login credentials are being rejected.",
        "Database user permissions issue.",
        "App is not launching.",
        "Emails are not showing up in the sent folder.",
        "Unable to change email password."
    ],
    "label": [
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues",
        "Email issues", "Login issues", "Database issues", "Application issues", "Email issues"
    ]
}

df = pd.DataFrame(data)

# Encode labels
label_map = {label: i for i, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_map)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))

# Tokenize the inputs
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

# Create a Dataset object
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels.tolist())
val_dataset = EmailDataset(val_encodings, val_labels.tolist())

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./email_issue_model")
tokenizer.save_pretrained("./email_issue_model")

# Load the fine-tuned model and classify new emails
classifier = pipeline("text-classification", model="./email_issue_model", tokenizer=tokenizer)

# Classify new emails
new_emails = [
    "My email account is locked, and I cannot reset the password.",
    "I get an error when trying to connect to the database.",
    "The app freezes when I try to upload a document."
]

for email in new_emails:
    result = classifier(email)[0]
    # Extract the integer label from the string label format
    label = list(label_map.keys())[int(result['label'].split('_')[-1])]
    print(f"Email: {email}\nPredicted Issue: {label}, Confidence: {result['score']:.4f}\n")
