# Chat With PDF - Generative AI App 📄🤖  

Welcome to the **Chat With PDF GenAI App** folder, a part of my **NLP Projects** repository! This folder provides a brief overview of the project, which combines the power of Generative AI, Amazon Bedrock, Langchain, and vector databases to enable interactive conversations with PDFs. To explore the project in detail, including all files, architecture, and implementation, please visit the [**Original Repository**](https://github.com/amark720/Chat_With_PDF_App).  

#### 📂 **View Full Chat With PDF Project:**  <a href="https://github.com/amark720/Chat_With_PDF_App" target="_blank"><img align="center" src="https://github.com/amark720/Amar-kumar/blob/master/ScreenShots/Click-Here-btn.gif" width="150" height="35" ></a>  

## About the Original Repository  
The **Chat With PDF GenAI App** allows users to interact with PDF documents in a conversational manner using the Retrieval-Augmented Generation (RAG) approach. It utilizes Amazon Bedrock for embedding and Large Language Models (LLMs), providing intelligent and context-aware responses based on user queries. The project consists of two key applications:  

### Admin Application  
- Allows admins to upload PDFs for processing.  
- Splits PDF text into chunks and generates vector embeddings using Amazon Titan Embedding Model.  
- Stores vector indices locally using FAISS and uploads them to Amazon S3.  

### User Application  
- Provides users with an interface to query and chat with PDFs.  
- Downloads vector indices from S3 and builds a local FAISS index.  
- Uses Langchain's RetrievalQA to fetch relevant chunks, provide context to the LLM, and generate meaningful responses.  

#### 🔗 **[Explore the Full Project—Click Here! 👈](https://github.com/amark720/Chat_With_PDF_App)**  

## Highlights of the Original Repository  
- 🌟 **Generative AI & LLMs:** Combines Anthropic Claude 2.1 and Amazon Titan Embedding G1 models for robust conversational capabilities.  
- 🧠 **Retrieval-Augmented Generation:** Ensures accurate responses by combining user queries with document context.  
- 📦 **Dockerized Applications:** Fully containerized for seamless deployment across environments.  
- ☁️ **Cloud Integration:** Leverages Amazon S3 for cost-effective vector storage.  
- 🖥️ **End-to-End Workflow:** Covers both admin-side and user-side functionalities, providing a complete solution.  

## Technologies Used  
- **Amazon Bedrock**: For embedding generation.  
- **Langchain**: For RetrievalQA and prompt engineering.  
- **Streamlit**: For building user-friendly web applications.  
- **FAISS**: For efficient vector storage and similarity search.  
- **Docker**: For containerizing admin and user applications.  

#### 🔗 **[View Complete Project—Click Here to Learn More! 👈](https://github.com/amark720/Chat_With_PDF_App)**  

## Stay Connected  
Feel free to explore, fork, or contribute to the projects! If you have any questions or feedback, don’t hesitate to reach out.  

📧 **Email:** amark720@gmail.com  
🌐 **Website:** [**AmarKumar.in**](https://AmarKumar.in)  

🙏 Thank you for visiting this folder! Head over to the [**Original Repository**](https://github.com/amark720/Chat_With_PDF_App) to dive deeper into this innovative project. 📄🤖  
