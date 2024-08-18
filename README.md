# FAQ Chatbot with GUI

## Overview

This project implements a simple FAQ chatbot using natural language processing techniques. The chatbot answers frequently asked questions (FAQs) using SpaCy for text processing and `scikit-learn` for text similarity. The chatbot is integrated with a modern and fluent graphical user interface (GUI) built with `tkinter`.

## Features

- **FAQ Handling**: Answers a set of predefined questions with corresponding answers.
- **Natural Language Processing**: Uses SpaCy for text preprocessing and `scikit-learn` for calculating text similarity.
- **Graphical User Interface**: Provides a user-friendly interface using `tkinter`.

## Installation

### Prerequisites

Ensure you have Python installed on your system. This project requires Python 3.x.

### Dependencies

Install the required Python libraries using pip:

```bash
pip install spacy scikit-learn
python -m spacy download en_core_web_sm
```

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/faq-chatbot.git
cd faq-chatbot
```

## Usage

### Running the Chatbot

To run the chatbot with the graphical user interface, execute the following command:

```bash
python faq_chatbot_gui.py
```

### How It Works

1. **Text Processing**: User input is processed using SpaCy to remove stop words and punctuation.
2. **FAQ Matching**: The processed user input is compared against a set of predefined FAQs using TF-IDF and cosine similarity.
3. **Response Display**: The chatbot's response is displayed in the GUI.

### Customization

You can customize the set of FAQs and their answers in the `faq_data` dictionary within the `faq_chatbot_gui.py` file. Modify the dictionary to add or update questions and answers.

## Example

Here's an example of how the chatbot responds:

```
You: What is your return policy?
Chatbot: You can return any item within 30 days of purchase for a full refund.
```

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

### Guidelines

- Ensure code follows the existing style and structure.
- Add tests for new features or bug fixes.
- Update the documentation as necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, you can reach me at:

- **Email**: abdulrafay517@hotmail.com
