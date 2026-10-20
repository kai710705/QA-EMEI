import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
QA_FILE = ROOT / "QA.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.45
FALLBACK_ANSWER = (
    "目前 FAQ 資料中沒有足夠相近的答案，建議直接聯絡診所客服確認。"
    "如有嚴重或急迫症狀，請立即就醫。"
)


@dataclass(frozen=True)
class QaEntry:
    question: str
    answer: str


def load_qa(path: Path) -> list[QaEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"QA dataset not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["question", "answer"]:
            raise ValueError("QA.csv must contain question and answer columns.")
        entries = [
            QaEntry(question=row["question"].strip(), answer=row["answer"].strip())
            for row in reader
        ]

    if not entries or any(not entry.question or not entry.answer for entry in entries):
        raise ValueError("QA.csv must contain non-empty questions and answers.")
    return entries


class QaAssistant:
    def __init__(self, entries: list[QaEntry]) -> None:
        self.entries = entries
        self.model = SentenceTransformer(MODEL_NAME)
        texts = [f"{entry.question}\n{entry.answer}" for entry in entries]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def answer(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "請先輸入問題。"

        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = np.asarray(self.embeddings) @ np.asarray(query_embedding)
        best_index = int(np.argmax(similarities))
        if float(similarities[best_index]) < MIN_SIMILARITY:
            return FALLBACK_ANSWER

        match = self.entries[best_index]
        return f"{match.answer}\n\n參考問題：{match.question}"


@st.cache_resource
def get_assistant() -> QaAssistant:
    return QaAssistant(load_qa(QA_FILE))


def main() -> None:
    st.set_page_config(page_title="醫美診所線上客服", page_icon="💬")
    st.title("醫美診所線上客服")
    st.write(
        "輸入問題，系統會從診所 QA 資料中找出語意最相近的回答。"
        "此工具僅提供一般資訊，不能取代醫療人員診斷或面診。"
    )

    assistant = get_assistant()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("請輸入你的問題"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在查找相關資訊..."):
                answer = assistant.answer(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
