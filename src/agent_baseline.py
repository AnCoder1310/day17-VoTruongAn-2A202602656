from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Characteristics:
    - Within-session short-term memory only (scoped strictly to thread_id)
    - No persistent User.md profile storage
    - Forgets long-term facts when queried in new threads or sessions
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Dispatch message processing to live agent or deterministic offline mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_msg = result["messages"][-1].content
                out_text = output_msg if isinstance(output_msg, str) else str(output_msg)

                prompt_tokens = estimate_tokens(message)
                reply_tokens = estimate_tokens(out_text)

                if thread_id not in self.sessions:
                    self.sessions[thread_id] = SessionState()
                session = self.sessions[thread_id]
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": out_text})
                session.prompt_tokens_processed += prompt_tokens
                session.token_usage += reply_tokens

                return {
                    "reply": out_text,
                    "response": out_text,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": reply_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent token count for one thread or across all threads."""
        if thread_id is not None:
            session = self.sessions.get(thread_id)
            return session.token_usage if session else 0
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        if thread_id is not None:
            session = self.sessions.get(thread_id)
            return session.prompt_tokens_processed if session else 0
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline has no compact memory layer."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline baseline behavior.

        Stores user message in the thread session, estimates accumulated prompt tokens
        per turn, and generates a response without access to any cross-thread state.
        """
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Accumulate prompt tokens per turn: all prior messages + current user message
        history_lines = [f"{m['role']}: {m['content']}" for m in session.messages]
        history_lines.append(f"user: {message}")
        prompt_text = "\n".join(history_lines)
        prompt_tokens = estimate_tokens(prompt_text)
        session.prompt_tokens_processed += prompt_tokens

        msg_lower = message.lower()
        is_query = bool(
            "?" in message
            or re.search(
                r"\b(nhắc lại|nhớ lại|có biết|là ai không|thử nhớ|xem đồ uống|mình tên gì|ở đâu|làm nghề gì|nghề hiện tại)\b",
                msg_lower,
            )
        )

        # Baseline forgets across threads: if asking for facts not present in this thread
        if is_query and not session.messages:
            reply_text = (
                "Chào bạn! Rất tiếc là trong phiên làm việc này mình chưa có thông tin trước đó của bạn. "
                "Bạn có thể chia sẻ lại thông tin để mình hỗ trợ tốt hơn được không?"
            )
        else:
            reply_text = (
                "Chào bạn! Mình đã ghi nhận thông tin của bạn trong phiên trò chuyện này."
            )

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens

        return {
            "reply": reply_text,
            "response": reply_text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": reply_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        """Optionally wire LangChain / LangGraph agent with InMemorySaver."""
        if not self.config.model.api_key:
            return
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                model=model,
                tools=[],
                checkpointer=checkpointer,
            )
        except Exception:
            self.langchain_agent = None
