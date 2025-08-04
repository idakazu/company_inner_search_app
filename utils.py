"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def load_employee_csv(file_path):
    """
    社員名簿CSVファイルを統合されたドキュメントとして読み込む

    Args:
        file_path: CSVファイルのパス

    Returns:
        統合されたドキュメントのリスト
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 部署ごとにグループ化して、検索しやすいテキスト形式に変換
        departments = df['部署'].unique()
        documents = []
        
        for dept in departments:
            dept_employees = df[df['部署'] == dept]
            
            # 部署ごとの統合テキストを作成
            content_lines = [f"【{dept}所属の従業員情報】"]
            content_lines.append(f"部署名: {dept}")
            content_lines.append(f"所属人数: {len(dept_employees)}人")
            content_lines.append("")
            
            # 各従業員の詳細情報を追加
            for _, employee in dept_employees.iterrows():
                employee_info = [
                    f"■従業員{employee['社員ID']}: {employee['氏名（フルネーム）']}",
                    f"  性別: {employee['性別']}",
                    f"  年齢: {employee['年齢']}歳",
                    f"  メールアドレス: {employee['メールアドレス']}",
                    f"  従業員区分: {employee['従業員区分']}",
                    f"  入社日: {employee['入社日']}",
                    f"  部署: {employee['部署']}",
                    f"  役職: {employee['役職']}",
                    f"  スキルセット: {employee['スキルセット']}",
                    f"  保有資格: {employee['保有資格']}",
                    f"  大学名: {employee['大学名']}",
                    f"  学部・学科: {employee['学部・学科']}",
                    f"  卒業年月日: {employee['卒業年月日']}",
                    ""
                ]
                content_lines.extend(employee_info)
            
            # 部署別の検索キーワードも追加
            content_lines.append("【検索キーワード】")
            content_lines.append(f"{dept}, {dept}部, {dept}所属, 従業員, 社員, 名簿, スタッフ")
            
            # 役職別の分類も追加
            roles = dept_employees['役職'].unique()
            content_lines.append(f"役職: {', '.join(roles)}")
            
            # 従業員区分別の分類も追加
            emp_types = dept_employees['従業員区分'].unique()
            content_lines.append(f"従業員区分: {', '.join(emp_types)}")
            
            # 統合されたテキストを作成
            content = "\n".join(content_lines)
            
            # ドキュメントオブジェクトを作成
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "department": dept,
                    "employee_count": len(dept_employees)
                }
            )
            documents.append(doc)
        
        # 全社員を対象とした統合ドキュメントも作成
        all_content_lines = [
            "【全社員名簿・従業員情報一覧】",
            f"総従業員数: {len(df)}人",
            f"部署数: {len(departments)}部署",
            f"部署一覧: {', '.join(departments)}",
            ""
        ]
        
        # 部署別の概要を追加
        for dept in departments:
            dept_count = len(df[df['部署'] == dept])
            all_content_lines.append(f"{dept}: {dept_count}人")
        
        all_content_lines.append("")
        all_content_lines.append("【検索キーワード】")
        all_content_lines.append("社員名簿, 従業員名簿, 全社員, 人事情報, 組織図, 部署別, 従業員一覧, スタッフ一覧")
        
        all_content = "\n".join(all_content_lines)
        
        # 全社員用ドキュメントを作成
        all_doc = Document(
            page_content=all_content,
            metadata={
                "source": file_path,
                "department": "全社",
                "employee_count": len(df)
            }
        )
        documents.append(all_doc)
        
        return documents
        
    except Exception as e:
        # エラーが発生した場合は空のリストを返す
        print(f"Error loading employee CSV: {e}")
        return []


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response