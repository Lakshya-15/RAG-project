import argparse
import os
import shutil
from typing import List
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main() -> None:
    """
    Main function to handle the command-line interface (CLI) for populating the database.
    """
    try:
        # Check if the database should be cleared (using the --reset flag).
        parser = argparse.ArgumentParser()
        parser.add_argument("--reset", action="store_true", help="Reset the database.")
        args = parser.parse_args()
        if args.reset:
            print("✨ Clearing Database")
            clear_database()

        # Create (or update) the data store.
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")


def load_documents() -> List[Document]:
    """
    Load documents from the specified data path.

    Returns:
        List[Document]: A list of loaded documents.
    """
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents (List[Document]): A list of documents to be split.

    Returns:
        List[Document]: A list of document chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"An error occurred while splitting documents: {e}")
        return []


def add_to_chroma(chunks: List[Document]) -> None:
    """
    Add or update document chunks in the Chroma database.

    Args:
        chunks (List[Document]): A list of document chunks to be added or updated.
    """
    try:
        # Load the existing database.
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")
    except Exception as e:
        print(f"An error occurred while adding to Chroma: {e}")


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for each document chunk.

    Args:
        chunks (List[Document]): A list of document chunks.

    Returns:
        List[Document]: A list of document chunks with unique IDs.
    """
    try:
        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks
    except Exception as e:
        print(f"An error occurred while calculating chunk IDs: {e}")
        return []


def clear_database() -> None:
    """
    Clear the Chroma database by removing the directory.
    """
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
    except Exception as e:
        print(f"An error occurred while clearing the database: {e}")


if __name__ == "__main__":
    main()
