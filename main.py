import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()


def load_markdown_files():
    vault_path = "/home/echidna/Notes"
    loader = DirectoryLoader(
        vault_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from the vault.")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = []
    for doc in docs:
        splits = md_splitter.split_text(doc.page_content)
        for s in splits:
            s.metadata.update(doc.metadata)
        md_header_splits.extend(splits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
    final_chunks = text_splitter.split_documents(md_header_splits)
    return final_chunks


def embedd_and_store(split_docs):
    cohere_embeddings = CohereEmbeddings(
        model="embed-v4.0", cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    vectorstore = Chroma.from_documents(
        split_docs,
        cohere_embeddings,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"},
        collection_name="obsidian_recall",
    )
    print(f"Stored {len(vectorstore.get()['ids'])} chunks in Chroma.")
    return vectorstore


def get_file_path(target_filename, vectorstore):
    all_metadatas = vectorstore.get()["metadatas"]
    if not all_metadatas:
        return None

    unique_paths = list(set([m["source"] for m in all_metadatas if "source" in m]))
    for path in unique_paths:
        if target_filename.lower() in path.lower():
            return path
    return None


def handle_summarize_file(filename, vectorstore, llm):

    full_path = get_file_path(filename, vectorstore)

    if not full_path:
        print(f"Could not find a note matching '{filename}'.")
        return

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"source": full_path}, "k": 10}
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant. Summarize the following document clearly and concisely.
    Focus on the main themes and key takeaways.

    Document Context:
    {context}

    Summary:
    """)

    chain = {"context": retriever} | prompt | llm | StrOutputParser()

    print(f"Summarizing {os.path.basename(full_path)}...")
    response = chain.invoke("Summarize this document")
    return response


def handle_summarize_topic(topic, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant. Summarize the key points about '{topic}' based on the provided context.

    Context from Vault:
    {context}

    Summary:
    """)

    chain = (
        {"context": retriever, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"Gathering notes on '{topic}'...")
    response = chain.invoke(topic)
    return response


def handle_flashcards(filename, vectorstore, llm):

    full_path = get_file_path(filename, vectorstore)

    if not full_path:
        print(f"Could not find a note matching '{filename}'.")
        return

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"source": full_path}, "k": 12}
    )

    flashcard_template = """
    You are a specialized study assistant. Based on the technical notes provided below,
    generate 5 high-quality flashcards.

    Each flashcard must have:
    - A 'Question' that tests a core concept, definition, or process.
    - A 'Concise Answer' based strictly on the provided text.

    FORMAT:
    Q: [Question]
    A: [Answer]
    ---

    NOTES FROM {filename}:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(flashcard_template)

    chain = (
        {"context": retriever, "filename": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"Generating flashcards from {os.path.basename(full_path)}...")
    response = chain.invoke(os.path.basename(full_path))
    return response


def getfile(description, vectorstore):
    results = vectorstore.similarity_search(description, k=3)
    if not results:
        return []
    found_files = set()
    for doc in results:
        full_path = doc.metadata.get("source", "Unknown")
        found_files.add(os.path.basename(full_path))
    return sorted(list(found_files))


def get_vectorstore():

    cohere_embeddings = CohereEmbeddings(
        model="embed-v4.0", cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=cohere_embeddings,
        collection_name="obsidian_recall",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def update_vault(vectorstore):
    existing_metadatas = vectorstore.get()["metadatas"]
    existing_sources = set([m["source"] for m in existing_metadatas if "source" in m])

    vault_path = "/home/echidna/Notes"
    loader = DirectoryLoader(
        vault_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    all_docs = loader.load()

    new_docs = [
        doc for doc in all_docs if doc.metadata["source"] not in existing_sources
    ]

    if not new_docs:
        print("✅ No new notes found. Vault is up to date.")
        return vectorstore

    print(f"🆕 Found {len(new_docs)} new documents. Processing...")

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    new_chunks = []
    for doc in new_docs:
        splits = md_splitter.split_text(doc.page_content)
        for s in splits:
            s.metadata.update(doc.metadata)
        new_chunks.extend(splits)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
    final_new_chunks = text_splitter.split_documents(new_chunks)

    vectorstore.add_documents(final_new_chunks)
    print(f"🚀 Successfully added {len(final_new_chunks)} new chunks.")
    return vectorstore


def chat_loop(vectorstore):
    llm = ChatOllama(model="gemma2:2b", temperature=0.7)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt_template = """
    You are a helpful assistant that answers questions based on the following retrieved
    chunks from an Obsidian vault. Use only the provided context. If you don't know, say so.

    Context:
    {retrieved_chunks}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"retrieved_chunks": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Obsidian Recall Active (Type 'exit' to quit) ---")

    while True:
        user_input = input("\n You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down Copilot.")
            break

        if user_input.startswith("/"):
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]

            if command == "/summarize":
                if args:
                    handle_summarize_topic(" ".join(args), vectorstore, llm)
                else:
                    print(
                        " Please provide a topic. Example: /summarize Neural Networks"
                    )

            elif command == "/summarize_file":
                if args:
                    handle_summarize_file(args[0], vectorstore, llm)
                else:
                    print(
                        "Please provide a filename. Example: /summarize_file Project.md"
                    )

            elif command == "/find":
                if args:
                    getfile(" ".join(args), vectorstore)
                else:
                    print(
                        "⚠️ Please provide a description. Example: /find architecture of mlops"
                    )

            elif command == "/flashcards":
                if args:
                    handle_flashcards(args[0], vectorstore, llm)
                else:
                    print("Please provide a filename. Example: /flashcards Biology.md")

            elif command == "/list":
                all_metadatas = vectorstore.get()["metadatas"]
                if not all_metadatas:
                    print(" No documents found in the vault.")
                    continue

                unique_paths = list(
                    set([m["source"] for m in all_metadatas if "source" in m])
                )
                print("Notes in Vault:")
                for path in unique_paths:
                    print(f" - {os.path.basename(path)}")
            else:
                print(f" Unknown command: {command}")

        else:
            print("Thinking...")
            response = chain.invoke(user_input)
            print(f"AI: {response}")


if __name__ == "__main__":
    db = get_vectorstore()
    db = update_vault(db)
    chat_loop(db)
