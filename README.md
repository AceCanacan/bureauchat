# Bureau-Chat 🏢

Bureau-Chat is a specialized chatbot designed to provide streamlined access to a vast database of government documents, including citizen's charters, manuals, and handbooks from multiple government organizations. Utilizing advanced AI technologies such as OpenAI's embeddings and llama_index, Bureau-Chat enables efficient querying and data retrieval without offering direct advice. This application empowers users to access essential information regarding government processes, ensuring they can conduct informed discussions and gain a better understanding of bureaucratic procedures. [Visit the App](https://bureauchat-ai.streamlit.app/)


## Features 🌟

- **Data Collection and Ingestion**
  - Government agency manuals and citizens charters were manually collected by downloading the documents and storing them in a specific folder.

- **Data Processing and Index Creation**
  - **Loading Data:** Documents were loaded using the PDF loader instead of a basic directory scanner.
  - **Embedding Generation:**
    - **Transformation:** Employing the llama_index with OpenAI's model text-embedding-3-small, the textual data is transformed into semantic embeddings to enhance understanding beyond simple keyword searches.
  - **Vector Storage:**
    - **ChromaDB Integration:** Embeddings are stored in ChromaDB hosted on an AWS EC2 instance, optimizing the retrieval process for user queries.

- **Cloud Setup and Vector Store Management**
  - **AWS Configuration:**
    - **Resource Deployment:** Utilizing AWS CloudFormation, the necessary infrastructure, including the EC2 instance for ChromaDB, is deployed and configured.

- **Query Processing**
  - **User Interface Interaction:**
    - **Interface Design:** A Streamlit-based interface allows for straightforward and engaging user interaction. The interface includes a sidebar where users can select a specific government agency, which then loads the related document collection into the system.
    - **Query Submission:** Users submit their legal queries through the chat interface, which then retrieves relevant information from the loaded collection.
  - **Query Handling and Response Generation:**
    - **Document Retrieval:** Upon receiving a query, llama_index’s vector store index evaluates the semantic similarity between the query and stored document embeddings to retrieve the most pertinent documents.
    - **Response Synthesis:** Employing a retrieval-augmented generation approach, the system synthesizes information from both the retrieved documents and pre-encoded AI knowledge to produce detailed and contextually relevant responses.
    - **Infrastructure Utilization:** This process leverages the scalable resources of AWS, ensuring the efficient management of concurrent user queries.
