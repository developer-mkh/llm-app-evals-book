from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.generate.types import GraphState
from app.llm import get_model

_SYSTEM_PROMPT_TEMPLATE = """\
あなたは株式会社サンプルエージェントのカスタマーサポート担当です。
お客様からのお問い合わせに対して、丁寧なビジネスメール形式で返信を作成してください。

<rules>
- 敬語を使用すること
- 宛名（会社名・お客様名）を含めること
- 挨拶文から始めること
- 具体的で役立つ返信を提供すること
- 締めの挨拶で終わること
- 返信件名はお問い合わせ内容に基づいた適切な件名にすること
</rules>

<forbidden_items>
以下の内容が含まれないように注意してください。
1. 見積り金額の提示: 具体的な金額、料金、費用の数値を提示しないこと。「お見積もりを作成します」のような案内は問題ありません。
2. 未確定情報の断定: 確認が必要な事項を断定的に述べていないこと。「確認いたします」「担当より回答いたします」といった表現を用い、「必ず対応可能です」などの断定を避けること。
3. 競合他社への言及: 他社の製品名、サービス名、会社名に具体的に言及しないこと。
</forbidden_items>"""

_USER_PROMPT_TEMPLATE = """\
以下のお問い合わせに対して返信メールを作成してください。件名と本文を分けて出力してください。

<inquiry>
<topic>{topic}</topic>
<customer_name>{customer_name}</customer_name>
<company_name>{company_name}</company_name>
<content>
{content}
</content>
</inquiry>"""


class GeneratedResponse(BaseModel):
    """生成された返信メール"""

    response_subject: str = Field(description="返信メールの件名（お問い合わせ内容から適切な件名を生成）")
    response_body: str = Field(description="返信メールの本文")


async def generate_response(state: GraphState) -> GraphState:
    """問い合わせ内容に基づいて返信メールを生成する。"""
    model = get_model(thinking=True)
    model_with_structure = model.with_structured_output(GeneratedResponse, method="json_schema")

    user_content = _USER_PROMPT_TEMPLATE.format(
        topic=state["topic"],
        customer_name=state["customer_name"],
        company_name=state["company_name"] or "（なし）",
        content=state["content"],
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=user_content),
    ]
    result: GeneratedResponse = await model_with_structure.ainvoke(messages)  # type: ignore[assignment]

    return {
        "response_subject": result.response_subject,
        "response_body": result.response_body,
    }
