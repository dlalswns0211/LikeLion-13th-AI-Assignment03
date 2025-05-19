# 필수 라이브러리 임포트
import os
import json
from dotenv import load_dotenv, find_dotenv  # 환경변수 로드
from openai import OpenAI  # Together API 호환용 OpenAI 클라이언트
import tiktoken  # 토큰 수 계산용 라이브러리

# .env 파일에서 환경변수 로드
load_dotenv(find_dotenv())

# 환경변수에서 API 키와 시스템 메시지 불러오기
API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]

# 기본 설정
BASE_URL = "https://api.together.xyz"  # Together API URL
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # 기본 LLM 모델
FILENAME = "message_history.json"  # 대화 기록 저장 파일
INPUT_TOKEN_LIMIT = 2048  # 입력 토큰 제한

# OpenAI 클라이언트 인스턴스 생성
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 일반 응답 (비스트리밍)
def chat_completion(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False,  # 비스트리밍
        **kwargs,
    )
    return response.choices[0].message.content

# 스트리밍 응답 함수
def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,  # 스트리밍 응답
        **kwargs,
    )

    response_content = ""

    # 스트리밍된 응답을 한 글자씩 출력
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None:
            print(chunk_content, end="")  # 실시간 출력
            response_content += chunk_content

    print()  # 줄바꿈
    return response_content

# 텍스트의 토큰 수 계산
def count_tokens(text, model):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

# 메시지 리스트의 전체 토큰 수 계산
def count_total_tokens(messages, model):
    total = 0
    for message in messages:
        total += count_tokens(message["content"], model)
    return total

# 토큰 제한을 초과하면 오래된 메시지를 제거
def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL):
    while count_total_tokens(messages, model) > token_limit:
        if len(messages) > 1:
            messages.pop(1)  # 사용자 메시지(시스템 메시지는 유지)
        else:
            break

# JSON 파일로 저장
def save_to_json_file(obj, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)

# JSON 파일 불러오기
def load_from_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"{filename} 파일을 읽는 중 오류 발생: {e}")
        return None

# 메인 챗봇 함수
def chatbot():
    # 기존 메시지 로드 또는 새로 시작
    messages = load_from_json_file(FILENAME)
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    print("Chatbot: 안녕하세요! 어떤 요리를 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break  # 종료 조건

        # 사용자 입력 추가
        messages.append({"role": "user", "content": user_input})

        # 현재 토큰 수 출력
        total_tokens = count_total_tokens(messages, DEFAULT_MODEL)
        print(f"[현재 토큰 수: {total_tokens} / {INPUT_TOKEN_LIMIT}]")

        # 토큰 수가 제한 초과 시 오래된 메시지 제거
        enforce_token_limit(messages, INPUT_TOKEN_LIMIT)

        # 챗봇 응답 출력 (스트리밍 방식)
        print("Chatbot: ", end="")
        response = chat_completion_stream(messages)
        print()

        # 챗봇 응답 저장
        messages.append({"role": "assistant", "content": response})

        # 대화 내역 저장
        save_to_json_file(messages, FILENAME)

# 챗봇 실행
chatbot()
