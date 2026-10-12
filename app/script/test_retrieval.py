import asyncio

from app.rag.retriever import retrieve_docs

async def main() -> None:
    query = input("Enter your question: ")
    documents = await retrieve_docs(query)

    if not documents:
        print("no docs available")
        return
    
    for index, document in enumerate(documents, start=1):
        print(f"\n --- result {index} ---")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(document.page_content)

if __name__ == "__main__":
    asyncio.run(main())

    