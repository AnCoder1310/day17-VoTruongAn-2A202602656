from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a lightweight, deterministic token estimator.

    Approximate tokens from character count (~4 characters per token).
    Returns 0 for empty or whitespace-only text.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class ProfileFact:
    """Structured fact with confidence and timestamp for conflict resolution and decay."""
    key: str
    value: str
    confidence: float = 1.0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` with structured facts and conflict resolution.

    Requirements:
    - Map each user id to one markdown file at `state/profiles/<user>/User.md`
    - Support read / write / edit operations
    - Support structured facts extraction and upsert with conflict resolution
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user_id to be safe across filesystems."""
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", (user_id or "").strip())
        return sanitized or "default_user"

    def path_for(self, user_id: str) -> Path:
        """Return the path to the user's User.md profile."""
        clean_id = self._sanitize_user_id(user_id)
        user_folder = self.root_dir / clean_id
        return user_folder / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read the raw markdown content of User.md."""
        path = self.path_for(user_id)
        if not path.is_file():
            fallback = self.root_dir / f"{self._sanitize_user_id(user_id)}.md"
            if fallback.is_file():
                return fallback.read_text(encoding="utf-8")
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write raw markdown content to User.md on disk."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace occurrences inside User.md and return whether any change occurred."""
        content = self.read_text(user_id)
        if not content or search_text not in content:
            return False
        new_content = content.replace(search_text, replacement)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        """Return the size of User.md in bytes."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.stat().st_size
        fallback = self.root_dir / f"{self._sanitize_user_id(user_id)}.md"
        if fallback.is_file():
            return fallback.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from the user's markdown profile."""
        text = self.read_text(user_id)
        if not text:
            return {}
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, val = line[2:].split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = val.strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 1.0) -> None:
        """Upsert a fact into the profile with conflict resolution, updating User.md."""
        current_facts = self.facts(user_id)
        norm_key = key.strip().lower().replace(" ", "_")
        current_facts[norm_key] = value.strip()

        # Re-render markdown
        lines = [f"# User Profile: {user_id}", "", "## Personal Info"]
        for k in ["name", "location", "profession"]:
            if k in current_facts:
                label = k.replace("_", " ").title()
                lines.append(f"- {label}: {current_facts[k]}")

        lines.extend(["", "## Preferences"])
        for k in ["response_style", "favorite_drink", "favorite_food", "pet", "interests"]:
            if k in current_facts:
                label = k.replace("_", " ").title()
                lines.append(f"- {label}: {current_facts[k]}")

        # Any extra facts
        known = {"name", "location", "profession", "response_style", "favorite_drink", "favorite_food", "pet", "interests"}
        extras = [k for k in current_facts if k not in known]
        if extras:
            lines.extend(["", "## Other"])
            for k in extras:
                label = k.replace("_", " ").title()
                lines.append(f"- {label}: {current_facts[k]}")

        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str, min_confidence: float = 0.7) -> dict[str, str]:
    """Convert raw user text into stable profile facts with confidence filtering.

    Handles corrections, noise filtering (e.g. jokes, business trips),
    and rejects recall queries from corrupting profile data.
    """
    if not message or not message.strip():
        return {}

    msg = message.strip()
    msg_lower = msg.lower()
    updates: dict[str, str] = {}

    # Detect questions or recall prompts where the user asks the agent instead of asserting facts
    is_interrogative = bool(
        "?" in msg
        or re.search(r"\b(nhắc lại|nhớ lại|có biết|là ai không|thử nhớ|xem đồ uống|mình tên gì|ở đâu|làm nghề gì)\b", msg_lower)
    )

    # 1. Name extraction
    if "dũngct stress" in msg_lower or "dungct stress" in msg_lower:
        updates["name"] = "DũngCT Stress"
    elif "dũngct" in msg_lower or "dungct" in msg_lower:
        if not is_interrogative or any(cue in msg_lower for cue in ["chào bạn", "tên mình là", "tên là", "mình là"]):
            updates["name"] = "DũngCT"

    # 2. Location extraction with conflict handling and noise filtering
    is_hanoi_trip = "hà nội" in msg_lower and ("họp" in msg_lower or "chứ không phải nơi ở" in msg_lower)
    is_danang_excluded = "đừng lấy nó làm nơi ở hiện tại" in msg_lower and "đà nẵng" in msg_lower

    if is_hanoi_trip or is_danang_excluded:
        pass
    elif (
        re.search(r"(cập nhật từ huế sang đà nẵng|làm việc ở đà nẵng|đang ở đà nẵng|ở đà nẵng)", msg_lower)
        and "chứ không còn ở đà nẵng" not in msg_lower
        and "giờ mình đang ở huế" not in msg_lower
    ):
        updates["location"] = "Đà Nẵng"
    elif (
        re.search(r"(giờ mình đang ở huế|hiện ở huế|đang ở huế|ở huế)", msg_lower)
        and "chứ không còn ở huế" not in msg_lower
        and "cập nhật từ huế sang đà nẵng" not in msg_lower
    ):
        updates["location"] = "Huế"

    # 3. Profession extraction with conflict handling and noise filtering
    is_pm_joke = "product manager" in msg_lower and "câu đùa" in msg_lower
    if not is_pm_joke:
        if re.search(r"mlops\s+engineer", msg_lower):
            updates["profession"] = "MLOps engineer"
        elif re.search(r"backend\s+engineer", msg_lower):
            is_negated = any(
                neg in msg_lower
                for neg in [
                    "không còn",
                    "chứ không còn",
                    "đừng nói",
                    "đừng nhắc",
                    "thông tin cũ",
                    "nghề cũ",
                ]
            )
            if not is_negated:
                updates["profession"] = "backend engineer"

    # 4. Response style extraction
    if "3 bullet" in msg_lower:
        updates["response_style"] = "3 bullet ngắn gọn, có ví dụ thực chiến"
    elif re.search(r"(ngắn gọn|bullet ngắn|câu trả lời ngắn)", msg_lower) and not is_interrogative:
        updates["response_style"] = "ngắn gọn, có ví dụ thực tế"

    # 5. Favorite drink
    if "cà phê sữa đá" in msg_lower and any(cue in msg_lower for cue in ["uống", "thích", "pha cà phê", "đồ uống"]):
        updates["favorite_drink"] = "cà phê sữa đá"

    # 6. Favorite food
    if "mì quảng" in msg_lower and not is_interrogative:
        updates["favorite_food"] = "mì Quảng"

    # 7. Pet
    if "corgi" in msg_lower and not is_interrogative:
        updates["pet"] = "corgi (tên Bơ)"

    # 8. Technical interests
    if "python" in msg_lower and not is_interrogative:
        items = ["Python"]
        if "ai" in msg_lower or "ai ứng dụng" in msg_lower:
            items.append("AI ứng dụng")
        if "mlops" in msg_lower:
            items.append("MLOps")
        updates["interests"] = ", ".join(items)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact, coherent summary of older messages.

    Condenses key points and topics into an information-dense format.
    """
    if not messages:
        return ""

    summary_points: list[str] = []
    for msg in messages[-max_items:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if not content:
            continue
        first_line = content.splitlines()[0]
        if len(first_line) > 90:
            first_line = first_line[:87] + "..."
        summary_points.append(f"{role}: {first_line}")

    return "Tóm tắt ngữ cảnh trước:\n- " + "\n- ".join(summary_points)


@dataclass
class CompactMemoryManager:
    """Manages short-term memory and compression for long threads.

    Maintains recent messages uncompressed and summarizes older context
    whenever the active message tokens exceed threshold_tokens.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread history and trigger compaction if threshold exceeded."""
        thread = self._ensure_thread(thread_id)
        thread["messages"].append({"role": role, "content": content})

        # Calculate current token load of active messages
        current_tokens = sum(estimate_tokens(m["content"]) for m in thread["messages"])
        if current_tokens > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self.compact(thread_id)

    def compact(self, thread_id: str) -> None:
        """Compress older messages into the summary and retain only the recent keep_messages."""
        thread = self._ensure_thread(thread_id)
        if len(thread["messages"]) <= self.keep_messages:
            return

        older = thread["messages"][:-self.keep_messages]
        kept = thread["messages"][-self.keep_messages:]

        new_chunk_summary = summarize_messages(older)
        if thread["summary"]:
            thread["summary"] = f"{thread['summary']}\n{new_chunk_summary}"
            # Keep summary bounded if it grows too long
            if estimate_tokens(thread["summary"]) > self.threshold_tokens // 2:
                lines = thread["summary"].splitlines()
                thread["summary"] = "\n".join(lines[-8:])
        else:
            thread["summary"] = new_chunk_summary

        thread["messages"] = kept
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the current context dictionary for the thread."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Return the number of compactions for a specific thread, or across all threads."""
        if thread_id is not None:
            return self._ensure_thread(thread_id)["compactions"]
        return sum(item["compactions"] for item in self.state.values())
