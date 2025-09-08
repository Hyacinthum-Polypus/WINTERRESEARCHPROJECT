from arango import ArangoClient
import os

def main():
    # Read configuration (with sensible defaults)
    hosts = os.getenv("ARANGO_HOSTS", "http://host.docker.internal:8529")
    root_user = os.getenv("ARANGO_ROOT_USER", "root")
    root_password = os.getenv("ROOT_DB_PASSWORD", "3mkIeiUqD2oEFmAF")
    db_name = os.getenv("ARANGO_DB_NAME", "sample_db")

    # Initialize the ArangoDB client
    client = ArangoClient(hosts=hosts)
    
    # Connect to '_system' database as root user
    # In a production environment, you should use a different user with appropriate permissions
    try:
        sys_db = client.db('_system', username=root_user, password=root_password)
    except Exception as e:
        print(f"Error connecting to ArangoDB: {e}")
        print("Make sure ArangoDB is running at http://host.docker.internal:8529")
        return
    
    # Create a new database if it doesn't exist
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        print(f"Database '{db_name}' created")
    
    # Connect to the database
    # Connect to the database using the same credentials
    db = client.db(db_name, username=root_user, password=root_password)
    
    # Create a collection if it doesn't exist
    collection_name = "users"
    if not db.has_collection(collection_name):
        users = db.create_collection(collection_name)
        print(f"Collection '{collection_name}' created")
    else:
        users = db.collection(collection_name)
    
    # Insert a document
    doc = {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    }
    meta = users.insert(doc)
    print(f"Document inserted with key: {meta['_key']}")
    
    # Query for the document
    query = "FOR doc IN users FILTER doc.name == @name RETURN doc"
    cursor = db.aql.execute(query, bind_vars={"name": "John Doe"})
    print("Query results:")
    for doc in cursor:
        print(f"  {doc}")
    
    # Update the document (provide _key with fields to change)
    users.update({"_key": meta["_key"], "age": 31, "updated": True})
    print(f"Document {meta['_key']} updated")
    
    # Retrieve the updated document
    updated_doc = users.get(meta['_key'])
    print(f"Updated document: {updated_doc}")
    
    # Delete by key to avoid _rev conflicts
    users.delete(meta["_key"])
    print(f"Document {meta['_key']} deleted")
    
    print("ArangoDB operations completed successfully")

if __name__ == "__main__":
    main()
