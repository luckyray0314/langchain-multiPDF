import os
import sys
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

app = Flask(__name__)
# app.debug = True
# app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app)
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """"
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer.
{context} 
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = "data/"
    documents = []
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                loader = PyMuPDFLoader(pdf_path)
                pdf_paths.append(pdf_path)
                documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunked_documents = text_splitter.split_documents(documents)
    pdf_ids = ['pdf' + str(i) for i in range(len(chunked_documents))]
    print("--->", pdf_paths)
    # new_client = chromadb.EphemeralClient()
    # client = chromadb.Client()
    # if client.list_collections():
    #     consent_collection = client.create_collection("consent_collection")
    # else:
    #     print("Collection already exists")
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        client=new_client,
        collection_name="openai_collection",
        persist_directory="chroma_db"
        # ids=pdf_ids,
    )
    # vectordb.persist()
    return vectordb


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

vectorstore = load_chunk_persist_pdf()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})
    # create the chain for allowing us to chat with the document

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever
        )
        | qa_prompt
        | llm
    )

    chat_history = []

@app.route('/api/proprietary-assistant', methods = ['POST'])
def proprietary_assistant():
    query = request.json["prompt"]
    # print(prompt)
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    ai_msg = rag_chain.invoke({'question': query, 'chat_history': chat_history})

    print(ai_msg)

    chat_history.extend([HumanMessage(content=query), ai_msg])
    
    # client.create_message(prompt)
    # client.create_run()
    # gpt_output = client.output()
    # conversation.append(gpt_output)
    # markdown = Markdown(ai_msg, code_theme="one-dark")
    # print('-------------------MARK-----------------', markdown)
    
    return jsonify({'result': str(ai_msg)[8:]})

@app.route('/', methods = ['GET'])
def index():
    return "Hello"

if __name__ == "__main__":
    print("kkk")
    
    app.run(host='0.0.0.0', port=5099)