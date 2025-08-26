# Llama2 Saudi Bot

The **Llama2 Saudi Bot** is a powerful AI-driven tool designed to provide intelligent responses to user queries using state-of-the-art language models and vector stores. This project leverages the **Mistral-7B-Instruct** model for high-quality natural language understanding and generation.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results and Documentation](#results-and-documentation)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

The **Llama2 Saudi Bot** is built to handle complex queries and provide accurate, context-aware answers. It uses the **Mistral-7B-Instruct-v0.1.Q4_K_M.gguf** model for its language processing capabilities and integrates with FAISS for efficient vector-based search.

This bot is particularly useful for tasks such as:
- Answering domain-specific queries (e.g., medical, technical, or legal).
- Providing summarized responses from large datasets.
- Supporting multilingual interactions with i18n capabilities.

---

## Features

- **Advanced Language Model**: Powered by the Mistral-7B-Instruct model.
- **Vector Search**: Uses FAISS for efficient document retrieval.
- **Multilingual Support**: Includes translations for multiple languages.
- **Customizable**: Easily adaptable to different domains and use cases.
- **Interactive UI**: Built with Chainlit for a responsive and user-friendly interface.

---

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.6 or higher.
- Required Python packages (installable via `requirements.txt`):
  - `langchain`
  - `chainlit`
  - `sentence-transformers`
  - `faiss`
  - `PyPDF2`
  - `ctransformers`
  - `deep-translator`

---

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/ahmad1adel/newSaudiCode.git
    cd llama2-saudi-bot
    ```

2. Create a Python virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the **Mistral-7B-Instruct-v0.1.Q4_K_M.gguf** model from [Hugging Face](https://huggingface.co/mistralai/mistral-7b-instruct-v0.1).

5. Configure the project by updating the `DB_FAISS_PATH` variable in `model.py` and other necessary settings.

---

## Getting Started

1. Prepare your environment and ensure all dependencies are installed.
2. Run the bot using the provided script:
    ```bash
    python app.py
    ```
3. Interact with the bot through the Chainlit interface.

---

## Usage

The bot can be used for various tasks, including:

- Querying domain-specific knowledge.
- Retrieving and summarizing documents.
- Translating and localizing content.

Simply input your query, and the bot will provide a detailed response based on the available data.

---

## Results and Documentation

### PDF Reports

The following PDF reports are available for reference:

- [Saudi Code](Results_pdf/Saudi%20Code.pdf)
- [الجديد تقرير فحص الأعمال الكهربائية ](Results_pdf/الجديد%20تقرير%20فحص%20الأعمال%20الكهربائية%20.pdf)
- [جدول فحص الأساسات ](Results_pdf/جدول%20فحص%20الأساسات%20.pdf)
- [جدول فحص الأساسات بدون الملاحظات](Results_pdf/جدول%20فحص%20الأساسات%20بدون%20الملاحظات.pdf)
- [الجديد تقرير فحص الأعمال الكهربائية](Results_pdf/جدول%20فحص%20الأعمال%20الكهربائية.pdf)
### Screenshots

Below are some screenshots showcasing the bot's interface and functionality:

#### Result Display 1
![Home Screen](screenshots/image_1.png)

#### Result Display 2
![Query Example](screenshots/image_2.png)

#### Result Display 3
![Results Display](screenshots/image_3.png)

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request with a detailed explanation of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Links

- [Mistral-7B-Instruct Model](https://huggingface.co/mistralai/mistral-7b-instruct-v0.1)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Chainlit](https://github.com/Chainlit/chainlit)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

Happy coding with **Llama2 Saudi Bot**! 🚀