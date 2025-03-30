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

nltk.download('punkt')

# BM25 Retriever 생성
embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)


#  첫 번째 FAISS 불러오기
document_vectorstore = FAISS.load_local("faiss_file", embedding_model, allow_dangerous_deserialization=True)
retriever = document_vectorstore.as_retriever(search_kwargs={"k": 2})

# 두 번째 FAISS 불러오기
symptoms_vectorstore = FAISS.load_local("symptoms_faiss_file", embedding_model, allow_dangerous_deserialization=True)
symptoms_retriever = symptoms_vectorstore.as_retriever(search_kwargs={"k": 2}) # label: not #1

# 세 번째 FAISS 불러오기
hospital_vectorstore = FAISS.load_local("hospital_faiss_file", embedding_model, allow_dangerous_deserialization=True)
hospital_retriever = hospital_vectorstore.as_retriever(search_kwargs={"k": 200}) # label: #1

#  prompt QA 한국어로 존중체이며 친밀한 모드로 대답하고 이모지와 함께 답변
qa_prompt =  """
you are an assistant for severe emergency situation. \
use the following pieces of retrieved context to answer the mention. \
If you don't know the answer, just say that you don't know. \
Keep the answer perfect. please use emoji with the answer. \
Please answer in Korean and use respectfully as well as friendly mode.\

{context}
"""

output_parser = StrOutputParser()

promptTemplate = ([
    ("system", qa_prompt),
])

promptChain = LLMChain(llm=llm, prompt=ChatPromptTemplate(promptTemplate), output_parser=output_parser)

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
    prompt =  """
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
    return rst['result']

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
hospital_chain = get_chain(hospital_retriever)

# user_symptoms_text = input("증상이 무엇인가요?") # 0
# print(user_symptoms_text)
# answer_promt = f"(응급에 가까우면 1 중경증이면 2 경증에 가까우면 3 으로 마지막에 태그로 붙이면서 라벨링)  {user_symptoms_text}"

# for cli
# user_symptoms = retriever_chain.invoke({"query": f"{answer_promt}"})['result']
# label = get_emergency_label(user_symptoms)

# if not label == 1:
#     detail_symptos = input('구체적인 증상?') # 1
#     detail_symptos_info = retriever_chain.invoke({"query": f"{detail_symptos}"})
#     rst = get_symptoms_info(symptoms_chain, label, detail_symptos_info['result'])
#     print('비응급 정보')
#     print(rst)

# geo_name = input('where?') # 2
# rst = get_hospital_info(hospital_chain, geo_name)
# print(rst)




st.header("스마트응급가이드 11911")
        
if "messages" not in st.session_state:
    st.session_state.state = 1
    st.session_state.messages = [{"role": "assistant","content": "너무 아픈데 확 오른 응급실 의료비, 응급상황인지 궁금할 때 찾게되는 챗봇"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt_message := st.chat_input("얼마나 걱정되는 상황인가요? 증상을 자세히 입력하며 응급인지 물어봐주세요! ): "):
    print('prompt_message')
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.spinner("Thinking..."):
        
        if st.session_state.state == 1:
            with st.chat_message(""):
                answer_promt = f"(응급에 가까우면 1 중경증이면 2 경증에 가까우면 3 으로 마지막에 태그로 붙이면서 라벨링)  {prompt_message}"
                user_symptoms = retriever_chain.invoke({"query": f"{answer_promt}"})['result']
                label = get_emergency_label(user_symptoms)
                st.session_state.label = label
                if not label == 1:
                    st.session_state.state = 2
                    st.session_state.messages.append({"role": "assistant", "content": "구체적인 증상이 무엇인가요?"})
                    st.markdown('구체적인 증상이 무엇인가요?')
                else:
                    st.session_state.state = 4
                    st.session_state.messages.append({"role": "assistant", "content": "현재 계신 구가 어떻게 되나요?"})
                    st.markdown('현재 계신 구가 어떻게 되나요?')

        elif st.session_state.state == 2:
            with st.chat_message(""):
                st.session_state.state = 3
                detail_symptos_info = retriever_chain.invoke({"query": f"{prompt_message}"})

                rst = get_symptoms_info(symptoms_chain, st.session_state.label, detail_symptos_info['result'])
                
                a = promptChain.invoke({
                    "context": detail_symptos_info['result']
                })

                st.session_state.messages.append({"role": "assistant", "content": a["text"]})
                st.markdown(a["text"])
                st.session_state.messages.append({"role": "assistant", "content": "현재 계신 구가 어떻게 되나요?"})
                st.markdown('현재 계신 구가 어떻게 되나요?')

        elif st.session_state.state == 3:
            with st.chat_message(""):
                rst = get_hospital_info(hospital_chain, prompt_message)

                a = promptChain.invoke({
                    "context": rst
                })

                st.session_state.state = 1
                st.session_state.messages.append({"role": "assistant", "content": a["text"]})
                st.markdown(a["text"])

        else:
            with st.chat_message(""):
                rst = get_hospital_info(hospital_chain, prompt_message)

                st.session_state.state = 1
                st.session_state.messages.append({"role": "assistant", "content": rst})
                st.markdown(rst)

