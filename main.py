import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define a larger set of FAQs and their answers
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase for a full refund.",
    "How can I track my order?": "You can track your order using the tracking number sent to your email.",
    "Do you offer international shipping?": "Yes, we offer international shipping to most countries.",
    "How can I contact customer service?": "You can contact customer service via email at support@example.com.",
    "What are your store hours?": "Our store hours are Monday to Friday, 9 AM to 5 PM.",
    "Can I change my shipping address after placing an order?": "Yes, you can change your shipping address within 24 hours of placing your order. Contact our support team for assistance.",
    "What payment methods do you accept?": "We accept major credit cards, PayPal, and bank transfers.",
    "Do you have a loyalty program?": "Yes, we have a loyalty program where you earn points with every purchase. Points can be redeemed for discounts.",
    "How do I use a discount code?": "You can enter your discount code at checkout to apply the discount to your order.",
    "Can I cancel my order?": "Orders can be canceled within 1 hour of placement. Contact our support team for cancellation requests.",
    "What is your warranty policy?": "We offer a 1-year warranty on all products. For more details, please check the warranty section on our website.",
    "How do I return a damaged item?": "If you receive a damaged item, please contact our support team within 7 days of receipt to arrange a return and replacement.",
    "Where can I find the size chart?": "Size charts are available on the product pages of our website. Check the 'Size Guide' section for detailed information.",
    "Do you offer gift cards?": "Yes, we offer gift cards in various denominations. You can purchase them on our website.",
    "Can I track the status of my return?": "Yes, once your return is processed, you will receive an email with tracking information for the return shipment.",
    "How long does shipping take?": "Shipping times vary based on your location and the shipping method chosen. Standard shipping typically takes 5-7 business days.",
    "What should I do if I forgot my password?": "Use the 'Forgot Password' link on the login page to reset your password. Follow the instructions sent to your email.",
    "How can I update my account information?": "Log in to your account and go to the 'Account Settings' section to update your information.",
    "Do you offer free shipping?": "We offer free standard shipping on orders over $50. Some restrictions may apply.",
    "What is your privacy policy?": "Our privacy policy outlines how we collect, use, and protect your personal information. You can review it on our website.",
    "Can I subscribe to your newsletter?": "Yes, you can subscribe to our newsletter on our website to receive updates and promotions.",
    "How do I provide feedback?": "You can provide feedback through our contact form on the website or by sending an email to feedback@example.com.",
    "Are there any current promotions?": "Check our website's 'Offers' section for the latest promotions and discounts.",
    "What should I do if I receive the wrong item?": "Contact our support team immediately to resolve the issue and arrange for a return or replacement.",
    "Can I purchase items in bulk?": "Yes, we offer bulk purchasing options. Please contact our sales team for more information."
}

# Preprocess the text using SpaCy
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Create a function to get the most relevant FAQ
def get_faq_answer(user_input):
    processed_input = preprocess(user_input)
    
    # Create a list of FAQs
    faqs = list(faq_data.keys())
    processed_faqs = [preprocess(faq) for faq in faqs]
    
    # Use TF-IDF Vectorizer to convert text data into vectors
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(processed_faqs)
    input_vector = vectorizer.transform([processed_input])
    
    # Calculate cosine similarities between input and FAQs
    similarities = cosine_similarity(input_vector, faq_vectors)
    best_match_index = similarities.argmax()
    
    # Return the answer for the best matching FAQ
    best_faq = faqs[best_match_index]
    return faq_data[best_faq]

# Chatbot loop
def chatbot():
    print("Hello! I'm here to answer your questions. Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        answer = get_faq_answer(user_input)
        print("Chatbot:", answer)

if __name__ == "__main__":
    chatbot()