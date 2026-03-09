import pandas as pd
import json

documents = []

# -------- Load FAQ JSON dataset --------
with open("data/Ecommerce_FAQ_Chatbot_dataset.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# handle category based structure
for category, faqs in faq_data.items():
    for item in faqs:
        question = item.get("question", "")
        answer = item.get("answer", "")

        text = f"""
Category: {category}
Question: {question}
Answer: {answer}
"""
        documents.append(text)


# -------- Load Product CSV dataset --------
products = pd.read_csv("data/train.csv")

for i, row in products.head(300).iterrows():
    name = str(row.get("product_name", ""))
    description = str(row.get("description", ""))
    price = str(row.get("price", ""))

    text = f"""
Product Information
Name: {name}
Description: {description}
Price: {price}
"""
    documents.append(text)


# -------- Save merged data --------
with open("data/ecommerce_data.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("Dataset prepared successfully!")