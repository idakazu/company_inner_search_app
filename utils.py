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
            content_lines = [
                f"【{dept}所属の従業員情報一覧】",
                f"部署名: {dept}",
                f"所属人数: {len(dept_employees)}人",
                ""
            ]
            
            # 各従業員の詳細情報を追加
            content_lines.append("【従業員一覧】")
            for i, (_, employee) in enumerate(dept_employees.iterrows(), 1):
                employee_info = [
                    f"{i}. 【{employee['氏名（フルネーム）']}】({employee['社員ID']})",
                    f"   性別: {employee['性別']} | 年齢: {employee['年齢']}歳",
                    f"   役職: {employee['役職']} | 従業員区分: {employee['従業員区分']}",
                    f"   入社日: {employee['入社日']}",
                    f"   メールアドレス: {employee['メールアドレス']}",
                    f"   スキルセット: {employee['スキルセット']}",
                    f"   保有資格: {employee['保有資格']}",
                    f"   学歴: {employee['大学名']} {employee['学部・学科']} ({employee['卒業年月日']}卒業)",
                    ""
                ]
                content_lines.extend(employee_info)
            
            # 部署の統計情報を追加
            content_lines.extend([
                "【部署統計】",
                f"役職別内訳: {dict(dept_employees['役職'].value_counts())}",
                f"従業員区分別内訳: {dict(dept_employees['従業員区分'].value_counts())}",
                f"性別内訳: {dict(dept_employees['性別'].value_counts())}",
                ""
            ])
            
            # 検索キーワードを強化（人事部の場合は特別に強化）
            if dept == "人事部":
                content_lines.extend([
                    "【検索キーワード】",
                    f"{dept}, {dept}部, {dept}所属, 従業員, 社員, 名簿, スタッフ, 人事情報, 組織, メンバー",
                    f"役職: {', '.join(dept_employees['役職'].unique())}",
                    f"従業員区分: {', '.join(dept_employees['従業員区分'].unique())}",
                    "人事部, 人事, HR, 人材管理, 採用, 労務, 給与, 人事担当, 人事スタッフ",
                    "人事部に所属, 人事部の従業員, 人事部の社員, 人事部のスタッフ, 人事部のメンバー",
                    "人事部員, 人事部一覧, 人事チーム, HR部門, 人材管理部門",
                    f"人事部詳細情報, 人事部{len(dept_employees)}名, 人事部全員",
                    ""
                ])
            else:
                content_lines.extend([
                    "【検索キーワード】",
                    f"{dept}, {dept}部, {dept}所属, 従業員, 社員, 名簿, スタッフ, 人事情報, 組織, メンバー",
                    f"役職: {', '.join(dept_employees['役職'].unique())}",
                    f"従業員区分: {', '.join(dept_employees['従業員区分'].unique())}",
                    ""
                ])
            
            # 統合されたテキストを作成
            content = "\n".join(content_lines)
            
            # ドキュメントオブジェクトを作成
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "department": dept,
                    "employee_count": len(dept_employees),
                    "document_type": "department_roster"
                }
            )
            documents.append(doc)
        
        # 全社員統合ドキュメント（人事部情報を強調）
        all_content_lines = [
            "【全社員名簿・従業員情報一覧】",
            f"総従業員数: {len(df)}人",
            f"部署数: {len(departments)}部署",
            f"部署一覧: {', '.join(departments)}",
            "",
            "【重要: 人事部従業員完全リスト】",
            f"人事部所属者数: {len(df[df['部署'] == '人事部'])}名",
            "人事部に所属している全従業員は以下の通りです:",
        ]
        
        # 人事部メンバーを個別に詳細列挙
        hr_members = df[df['部署'] == '人事部']
        for i, (_, emp) in enumerate(hr_members.iterrows(), 1):
            all_content_lines.extend([
                f"\n{i}. 【人事部】{emp['氏名（フルネーム）']}",
                f"   - 社員ID: {emp['社員ID']}",
                f"   - 役職: {emp['役職']}",
                f"   - 従業員区分: {emp['従業員区分']}",
                f"   - 年齢: {emp['年齢']}歳",
                f"   - 入社日: {emp['入社日']}"
            ])
        
        all_content_lines.extend([
            "",
            f"※人事部は{len(hr_members)}名の重要な組織です",
            "",
            "【部署別人数】"
        ])
        
        # 部署別の概要を追加
        for dept in departments:
            dept_count = len(df[df['部署'] == dept])
            all_content_lines.append(f"・{dept}: {dept_count}人")
        
        all_content_lines.extend([
            "",
            "【検索キーワード】",
            "社員名簿, 従業員名簿, 全社員, 人事情報, 組織図, 部署別, 従業員一覧, スタッフ一覧",
            "人事部, 各部署, 組織構成, メンバー構成, 全従業員, 社員情報"
        ])
        
        all_content = "\n".join(all_content_lines)
        
        # 全社員用ドキュメントを作成
        all_doc = Document(
            page_content=all_content,
            metadata={
                "source": file_path,
                "department": "全社",
                "employee_count": len(df),
                "document_type": "company_roster"
            }
        )
        documents.append(all_doc)
        
        # 人事部専用の追加ドキュメントを作成（検索精度向上のため）
        hr_dept_employees = df[df['部署'] == '人事部']
        if len(hr_dept_employees) > 0:
            hr_content_lines = [
                "【人事部従業員情報詳細一覧】",
                f"人事部総人数: {len(hr_dept_employees)}名",
                "",
                "【人事部全メンバー詳細情報】"
            ]
            
            for i, (_, emp) in enumerate(hr_dept_employees.iterrows(), 1):
                hr_content_lines.extend([
                    f"\n■ 人事部メンバー {i}人目",
                    f"【{emp['氏名（フルネーム）']}】",
                    f"・社員ID: {emp['社員ID']}",
                    f"・部署: {emp['部署']}（人事部所属）",
                    f"・役職: {emp['役職']}",
                    f"・従業員区分: {emp['従業員区分']}",
                    f"・性別: {emp['性別']}",
                    f"・年齢: {emp['年齢']}歳",
                    f"・入社日: {emp['入社日']}",
                    f"・メールアドレス: {emp['メールアドレス']}",
                    f"・スキルセット: {emp['スキルセット']}",
                    f"・保有資格: {emp['保有資格']}",
                    f"・学歴: {emp['大学名']} {emp['学部・学科']} ({emp['卒業年月日']}卒業)",
                    ""
                ])
            
            hr_content_lines.extend([
                "\n【人事部検索用キーワード】",
                "人事部, 人事, HR, 人材管理, 採用, 労務, 給与, 人事担当, 人事スタッフ",
                "従業員情報, 社員情報, 人事メンバー, 人事チーム, 人事組織",
                f"人事部員{len(hr_dept_employees)}名, 人事部所属, 人事部一覧",
                "人事部に所属, 人事部の従業員, 人事部の社員, 人事部のスタッフ, 人事部のメンバー"
            ])
            
            hr_content = "\n".join(hr_content_lines)
            
            hr_doc = Document(
                page_content=hr_content,
                metadata={
                    "source": file_path,
                    "department": "人事部",
                    "employee_count": len(hr_dept_employees),
                    "document_type": "hr_department_detailed"
                }
            )
            documents.append(hr_doc)
            
            # 人事部専用テーブル形式ドキュメントも追加
            hr_table_lines = [
                "【人事部従業員一覧表】",
                f"人事部所属者: {len(hr_dept_employees)}名",
                "",
                "| No. | 氏名 | 社員ID | 役職 | 従業員区分 | 年齢 | 入社日 |",
                "|-----|------|--------|------|------------|------|--------|"
            ]
            
            for i, (_, emp) in enumerate(hr_dept_employees.iterrows(), 1):
                hr_table_lines.append(
                    f"| {i} | {emp['氏名（フルネーム）']} | {emp['社員ID']} | {emp['役職']} | {emp['従業員区分']} | {emp['年齢']}歳 | {emp['入社日']} |"
                )
            
            hr_table_lines.extend([
                "",
                "【詳細情報】",
                "上記は人事部に所属している全従業員の一覧です。",
                f"合計{len(hr_dept_employees)}名が人事部に配属されています。",
                "",
                "【人事部関連キーワード】",
                "人事部所属, 人事部員, 人事部の従業員, 人事部一覧, 人事部メンバー",
                "人事担当者, 人事スタッフ, 人事チーム, HR部門, 人材管理部門"
            ])
            
            hr_table_content = "\n".join(hr_table_lines)
            
            hr_table_doc = Document(
                page_content=hr_table_content,
                metadata={
                    "source": file_path,
                    "department": "人事部",
                    "employee_count": len(hr_dept_employees),
                    "document_type": "hr_department_table"
                }
            )
            documents.append(hr_table_doc)
            
            # 人事部専用の簡潔な名前リストドキュメントも追加（より高い検索精度のため）
            hr_simple_lines = [
                "【人事部メンバー簡潔リスト】",
                f"人事部には{len(hr_dept_employees)}名の従業員が所属しています。",
                "",
                "人事部所属の全従業員："
            ]
            
            for i, (_, emp) in enumerate(hr_dept_employees.iterrows(), 1):
                hr_simple_lines.append(f"{i}. {emp['氏名（フルネーム）']} - {emp['役職']} ({emp['従業員区分']})")
            
            hr_simple_lines.extend([
                "",
                "これが人事部に所属している全従業員の完全なリストです。",
                "人事部, 人事部所属, 人事部員, 人事部の従業員, 人事部一覧, 人事部メンバー",
                f"人事部総数{len(hr_dept_employees)}名"
            ])
            
            hr_simple_content = "\n".join(hr_simple_lines)
            
            hr_simple_doc = Document(
                page_content=hr_simple_content,
                metadata={
                    "source": file_path,
                    "department": "人事部",
                    "employee_count": len(hr_dept_employees),
                    "document_type": "hr_department_simple"
                }
            )
            documents.append(hr_simple_doc)
        
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