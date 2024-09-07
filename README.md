# Chroma RAG System

This project implements a command-line interface (CLI) for querying and populating a retrieval-augmented generation (RAG) system. It uses LangChain's Chroma vector store to search for relevant information in a document database and generates responses based on the retrieved context.

## Project Structure

### Files

- **main.py**: Handles querying and CLI functionality for text-based questions using the RAG system.
- **populate_db.py**: Populates the Chroma database with documents split into chunks.
- **test_query.py**: Contains unit tests for validating the query results against expected responses.
- **get_embedding_function.py**: This script contains the function that generates embeddings for the documents, used by Chroma.

### Key Functions

#### main.py

- `main()`: CLI function for querying text from the Chroma database.
- `query_rag(query_text: str)`: Searches the Chroma database for relevant context, formats a prompt for the model, and returns the model's response.

#### populate_db.py

- `main()`: CLI function to handle database population, with an option to reset the database.
- `load_documents()`: Loads PDF documents from a specified directory.
- `split_documents(documents: List[Document])`: Splits loaded documents into chunks for storage in the vector store.
- `add_to_chroma(chunks: List[Document])`: Adds or updates the document chunks in the Chroma database.
- `clear_database()`: Clears the Chroma database by removing the directory.

#### test_query.py

- `test_monopoly_rules()`: Tests the query function with a specific question about Monopoly game rules.
- `test_ticket_to_ride_rules()`: Tests the query function with a specific question about Ticket to Ride game rules.
- `query_and_validate(question: str, expected_response: str)`: Runs a query and compares the response with an expected result.

## Usage

### 1. Querying the RAG System

To query the RAG system, run the following command:

```bash
python main.py "<your_query_here>"
```


### Example

To query the RAG system with a specific question, use the following command:

```bash
python main.py "How much total money does a player start with in Monopoly?"
```


### 2. Populating the Database

To populate the Chroma database with document chunks, use the following command:

```bash
python populate_db.py
```

If you want to clear the existing database before repopulating, use the `--reset` flag:

```bash
python populate_db.py --reset
```

### 3. Running Tests

Unit tests are included to validate the querying system. To run the tests, use the following command:

```bash
python test_query.py
```

## Requirements

- Python 3.8+
- Libraries:
  - `langchain`
  - `argparse`
  - `PyPDF2`
  - `Chroma`
  - `Ollama`

## How it Works

1. **Document Loading & Chunking**:The PDF documents are loaded from the `data/` directory and split into smaller chunks. Each chunk is assigned a unique ID for reference.
2. **Embedding and Vector Store**:The `get_embedding_function()` generates embeddings for the document chunks, which are stored in the Chroma vector store for fast similarity-based retrieval.
3. **Querying the System**:When a query is provided, the system retrieves the top 5 most similar document chunks and constructs a context. The query and context are passed to the Ollama model to generate a response.
4. **Testing**:
   Tests validate that the model's responses match predefined expected responses using a simple true/false prompt.

## Future Enhancements

- Implement support for more document formats (e.g., DOCX, TXT).
- Add more robust error handling and logging.
- Enable the use of different models beyond `mistral` in the Ollama integration.

```

This version provides a clean and structured Markdown format with code blocks and bullet points to enhance readability.
```
