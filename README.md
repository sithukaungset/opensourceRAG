# Retrieval Augmented Generation (RAG) Chatbot

This project involves developing a chatbot that leverages Retrieval Augmented Generation (RAG) to accurately and truthfully answer queries related to the paper titled "Llama 2: Open Foundation and Fine-Tuned Chat Models" by MetaAI. The chatbot uses a combination of vector embeddings and a language model to process and generate responses based on the provided paper.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Modules](#modules)
4. [Key Classes & Methods](#key-classes-and-methods)
5. [Flow](#flow)
6. [Notes](#notes)
7. [Experimentation](#experimentation)
8. [Contributing](#contributing)
9. [License](#license)

## Features

- RESTful API to handle user queries and return responses.
- Integration with `llama.cpp` or `ctranslate2` to run language models on CPU.
- Storage of chatbot responses in a database for retrieval and analysis.
- Latency measurement for each query response.
- Experimentation with various text chunking strategies for optimal information retrieval.

## Architecture

The project is structured into multiple components, each responsible for specific functionalities, such as query processing, interaction with the LLM, and database management.

## Modules

- **Flask**: Web framework to create the RESTful API.
- **llama-cpp-python**: Interface to run the Llama 2 model.
- **pymupdf**: For reading and processing PDF files.
- **sqlite3**: Database to store chat history and responses.
- **time**: To measure latency of responses.
- **numpy**: For handling vector embeddings.
- **transformers**: Hugging Face library for language model interaction.

## Key Classes and Methods

- **ChatbotAPI**: Handles API requests, processes queries, and returns responses.
- **TextChunker**: Implements various strategies for segmenting the paper into text chunks.
- **LatencyMonitor**: Measures and logs the time taken to generate responses.
- **DatabaseManager**: Manages storing and retrieving chat history and responses in a SQLite database.

## Flow

1. **Setup**: Initialize the REST API and load the Llama 2 model.
2. **Handling Queries**: 
   - Encode user queries into vector embeddings.
   - Retrieve relevant text chunks from the paper.
   - Generate a response using the language model.
   - Measure and log latency for each response.
3. **Response Storage**: Store each query and corresponding response in the database for future retrieval.

## Notes

- The chatbot is designed to run on CPU using efficient libraries.
- Various text chunking strategies are implemented to optimize information retrieval.
- Latency is measured and logged for each interaction with the chatbot.

## Experimentation

- **Text Chunking**: Experiment with different methods to split the paper into chunks (e.g., sentence-based, paragraph-based).
- **Latency Optimization**: Analyze and optimize the response time of the chatbot.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and create a pull request. For major changes, open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. Please see the LICENSE file for more information.