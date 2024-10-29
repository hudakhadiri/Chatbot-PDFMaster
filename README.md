# Chatbot-PDFMaster
Chatbot-PDFMaster is a Python app that lets you interact with multiple PDFs using natural language. Ask questions, and the chatbot will generate responses based on the documents' content using a language model. It only answers queries related to the loaded PDFs, providing accurate, relevant information efficiently.

Are Embeddings Used?
Yes, embeddings are used internally. When we load the pre-trained model bert-large-uncased-whole-word-masking-finetuned-squad, it includes an embedding layer. This layer converts input tokens (words, subwords, or characters) into continuous vectors.

Are we Using a Transformer Model?
Yes, BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model. The code uses a pre-trained BERT model fine-tuned for the question-answering task.

Are we Using LLMs?
Yes, BERT is a type of Large Language Model (LLM), especially when it comes in large variations like bert-large.

Relation Between LLM and Transformer
Transformers: An architectural framework for building models that can process sequences of data (like text).
LLMs: Large Language Models built using transformer architectures. They are trained on massive amounts of text data to perform language-related tasks (e.g., GPT-3, BERT).

How Does It Work Without Embeddings?
Even though you didn't explicitly implement embeddings in your code, the pre-trained transformer model you used has built-in embedding layers. These embeddings are essential for the model to process the input text and generate meaningful outputs.

In summary, the code leverages the embeddings and transformers implicitly through the use of the pre-trained BERT model. This is why we didn't need to implement embeddings explicitly—the model handles it for us internally.
