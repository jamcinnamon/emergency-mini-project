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


# 세 번째 FAISS 불러오기
hospital_vectorstore = FAISS.load_local("hospital_faiss_file", embedding_model, allow_dangerous_deserialization=True)
hospital_retriever = hospital_vectorstore.as_retriever(search_kwargs={"k": 200}) # label: #1


# RetrievalQA 설정
def get_chain(retriever):
    return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever, 
            return_source_documents=True
    )


# 3단계
def get_hospital_info(chain, geo_name): # 2
    # 지역 정보 입력 받은 후 체인으로 넘어감
    prompt =  f"""
        사용자가 대답한 구({geo_name})에 있는 병원정보를 알려주세요.
        ------------
        예))
        user: 관악구 근처요.
        answer: 해당 지역의 병원을 알려드릴게요! 너무 걱정되시면 119로 전화해 바로 물어보는 것도 좋은 방법이에요. 저는 챗봇이기 때문에 정답이 될 수는 없거든요.

        해당 
        병원: 의료법인서울효천의료재단에이치플러스양지병원
        전화번호: 02-1877-8875
        주소: 서울특별시 관악구 남부순환로 1636, 양지병원 (신림동)

        도움이 되셨기를 바래요. 경증으로 판단되어 (비응급) 비용이 걱정되신다면, 추가적 증상을 입력주시고 처치방법이 있는지 물어봐 주세요~
        -------------
    """
    
    rst = chain.invoke({"query": prompt})
    print(rst['result'])
    return rst['result']

hospital_chain = get_chain(hospital_retriever)



geo_name = input('where?') # 2
rst = get_hospital_info(hospital_chain, geo_name)
print(rst)



