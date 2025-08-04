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
            
            # 部署の概要情報を追加
            if dept == "人事部":
                content_lines.extend([
                    "【人事部について】",
                    "人事部は従業員の採用、研修、評価、給与管理などを担当する重要な部署です。",
                    f"現在{len(dept_employees)}名の従業員が在籍しています。",
                    ""
                ])
            
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
            
            # 検索キーワードを強化
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
        
        # 人事部専用の詳細ドキュメントを作成
        hr_employees = df[df['部署'] == '人事部']
        if len(hr_employees) > 0:
            hr_content_lines = [
                "【人事部従業員詳細情報】",
                "=== 人事部メンバー完全リスト ===",
                f"人事部総人数: {len(hr_employees)}名",
                "",
                "【全メンバー詳細】"
            ]
            
            for i, (_, emp) in enumerate(hr_employees.iterrows(), 1):
                hr_content_lines.extend([
                    f"【{i}番目】{emp['氏名（フルネーム）']} ({emp['社員ID']})",
                    f"  ・役職: {emp['役職']}",
                    f"  ・年齢: {emp['年齢']}歳 ({emp['性別']})",
                    f"  ・雇用形態: {emp['従業員区分']}",
                    f"  ・入社: {emp['入社日']}",
                    f"  ・連絡先: {emp['メールアドレス']}",
                    f"  ・専門スキル: {emp['スキルセット']}",
                    f"  ・資格: {emp['保有資格']}",
                    f"  ・学歴: {emp['大学名']} {emp['学部・学科']}",
                    ""
                ])
            
            hr_content_lines.extend([
                "【人事部検索用キーワード】",
                "人事部, 人事課, HR, Human Resources, 人材管理, 採用, 研修, 給与, 評価",
                "従業員管理, 労務, 人事担当, 人事スタッフ, 人事メンバー, 人事チーム",
                f"人事部メンバー: {', '.join(hr_employees['氏名（フルネーム）'].tolist())}",
                f"人事部役職: {', '.join(hr_employees['役職'].unique())}"
            ])
            
            hr_doc = Document(
                page_content="\n".join(hr_content_lines),
                metadata={
                    "source": file_path,
                    "department": "人事部",
                    "employee_count": len(hr_employees),
                    "document_type": "hr_detailed_roster"
                }
            )
            documents.append(hr_doc)
        
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
            all_content_lines.append(f"・{dept}: {dept_count}人")
        
        all_content_lines.extend([
            "",
            "【人事部特別情報】",
            f"人事部所属者: {len(df[df['部署'] == '人事部'])}名",
            f"人事部メンバー名: {', '.join(df[df['部署'] == '人事部']['氏名（フルネーム）'].tolist())}",
            "",
            "【検索キーワード】",
            "社員名簿, 従業員名簿, 全社員, 人事情報, 組織図, 部署別, 従業員一覧, スタッフ一覧",
            "人事部, 各部署, 組織構成, メンバー構成"
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
        
        return documents
        
    except Exception as e:
        # エラーが発生した場合は空のリストを返す
        print(f"Error loading employee CSV: {e}")
        return []
        
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