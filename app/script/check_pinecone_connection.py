from app.vector_store.store import get_vector_store


def main() -> None:
    vector_store = get_vector_store()
    print("LangChain vector store initialized successfully.")


if __name__ == "__main__":
    main()