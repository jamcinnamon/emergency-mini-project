import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from first_aid import open_json
import streamlit as st

from langchain_core.output_parsers import StrOutputParser
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

load_dotenv()
api_key= os.getenv("OPENAI_API_KEY")


# BM25 Retriever 생성
embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)


document_vectorstore = FAISS.load_local("faiss_file", embedding_model, allow_dangerous_deserialization=True)
retriever = document_vectorstore.as_retriever(search_kwargs={"k": 2})

symptoms_vectorstore = FAISS.load_local("symptoms_faiss_file", embedding_model, allow_dangerous_deserialization=True)
symptoms_retriever = symptoms_vectorstore.as_retriever(search_kwargs={"k": 2}) # label: not #1


# RetrievalQA 설정
def get_chain(retriever):
    return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever, 
            return_source_documents=True
    )



# 1단계
def get_emergency_label(response_text):
 
    prompt =  """
    다음 응답을 보고 응급 여부를 라벨링해주세요:
    - 1: 응급 상황
    - 2: 애매함
    - 3: 비응급
    응답: '{response_text}'
    라벨만 숫자로 반환해주세요.
    """

    label_chain = llm | StrOutputParser()
    label = label_chain.invoke([{"role": "user", "content": prompt}])
    return label


# 2단계
def get_symptoms_info(chain, label, detail_symptos):

    prompt = """
        <goal>
            구체적인 응급처리 방법을 알려주세요
        </goal>

        <emergency_tag>
            label: {label}
        </emergency_tag>

        <detail_symptoms>
            {detail_symptos}
        </detail_symptoms>

        <result_format>
            symptoms 있나요? 이런 증상이 있다면 응급 상황일 수 있습니다. 아니면, 응급처치로 이런걸 해보세요
        </result_format>
    """
    rst = chain.invoke({"query": prompt})

    return rst['result']


retriever_chain = get_chain(retriever)
symptoms_chain = get_chain(symptoms_retriever)

user_symptoms_text = input("증상이 무엇인가요?") # 0
print(user_symptoms_text)
answer_promt = f"(응급에 가까우면 1 중경증이면 2 경증에 가까우면 3 으로 마지막에 태그로 붙이면서 라벨링)  {user_symptoms_text}"

user_symptoms = retriever_chain.invoke({"query": f"{answer_promt}"})['result']
print(user_symptoms)
label = get_emergency_label(user_symptoms)
