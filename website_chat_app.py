from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_docs(url):
    """
    Load a web document that will provide additional context to LLM
    """
    loader = WebBaseLoader(url)
    data = loader.load()
    return data


def get_text_chunks(text):
    """
    Split document into smaller sized chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=100)
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Embed and store document chunks in a vector store.
    We can retreive from tgis vectorstore based on similarity search
    """
    vectorstore = Chroma.from_documents(documents=text_chunks,
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


def get_rag_chain(retriever):
    """
    Create a chain that takes a question, retrieves relevant documents,
    constructs a prompt, passes that to a model"""

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    template = """SYSTEM: You are a question answer bot. 
                Be factual in your response.
               Respond to the following question: {question} only from the below
               context :{context}. 
               If you don't know the answer, just say that you don't know.
               """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain = (
            {"context": retriever
                        | format_docs, "question": RunnablePassthrough()}
            | rag_prompt_custom
            | llm
            | StrOutputParser()
    )
    return rag_chain


def gen_answer(url, question):
    data = load_docs(url)
    chunks = get_text_chunks(data)
    vector = get_vector_store(chunks)
    ragchain = get_rag_chain(vector)
    answer = ragchain.invoke(question)
    return answer


def main():
    st.title("Website BOT")
    st.write("Enter a website URL and a question to get an answer.")

    # Input fields for URL and question
    url = st.text_input("Enter Website URL")
    question = st.text_input("Enter Your Question")

    # Button to trigger answer generation
    if st.button("Get Answer"):
        if url.strip() == "":
            st.warning("Please enter the URL.")
        elif question.strip() == "":
            st.warning("Please enter the Question.")
        else:
            # Generate the answer
            answer = gen_answer(url, question)
            # Display the answer
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()