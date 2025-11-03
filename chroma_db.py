import chromadb
from chromadb.utils import embedding_functions

ef =  embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-V2"
)

client = chromadb.Client()
collection_name ="my_grocery_collection"

# Define the main function to interact with the Chroma DB
def main():
    try:   
        collection = client.create_collection(
            name=collection_name,
            metadata={"description":"A collection of strong grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        
        )

        print("Collection created:", collection.name)
        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        # Create a list of unique IDs for each text item in the 'texts' array
        # Each ID follows the format 'food_<index>', where <index> starts from 1
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        # Add documents and their corresponding IDs to the collection
        # The `add` method inserts the data into the collection
        # The documents are the actual text items, and the IDs are unique identifiers
        # ChromaDB will automatically generate embeddings using the configured embedding function
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        all_items = collection.get()

        # Log the retrieved items to the console for inspection
        # This will print out all the documents, IDs, and metadata stored in the collection
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        def perform_similarity_search(collection,all_trems):
            try: 

                query_term = "apple"
                results = collection.query(
                    query_texts=[query_term],
                    n_results=3
                )
                print(f"Query results for '{query_term}':")
                print(results)

                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print("No document found for similarity search", query_term)

                print(f'Top 3 similar documents to "{query_term}":')

                for i in range(min(3,len(results['ids'][0]))):
                    doc_id=results['ids'][0][i]
                    score = results['distances'][0][i]
                    text = results['documents'][0][i]
                    if not text:
                        print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

            except Exception as error:
                print("Error in similarity search:", error)

        perform_similarity_search(collection, all_items)

    except Exception as error:  # Catch any errors and log them to the console
        print(f"Error: {error}")





if __name__ == "__main__":
    main()
        



