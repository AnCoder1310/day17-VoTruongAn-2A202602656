from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit and integration testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # nhỏ để compact kích hoạt sớm
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Initial state: empty profile
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # Write initial markdown profile
    initial_text = "# User Profile: test_user\n\n## Personal Info\n- Name: Alice\n- Location: Đà Nẵng\n"
    written_path = store.write_text(user_id, initial_text)
    assert written_path.is_file()
    assert store.file_size(user_id) > 0
    assert "Alice" in store.read_text(user_id)

    # Edit text
    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    updated_text = store.read_text(user_id)
    assert "Huế" in updated_text
    assert "Đà Nẵng" not in updated_text

    # Edit nonexistent text should return False
    assert store.edit_text(user_id, "NonexistentCity", "Hà Nội") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction and summarize older context."""
    # Threshold 40 tokens, retain 2 messages
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "stress_test_thread"

    # Append turns that exceed the token threshold
    manager.append(thread_id, "user", "Báo cáo đầu tiên về Artemis III rất dài và nhiều nội dung chi tiết.")
    manager.append(thread_id, "assistant", "Đã nhận được thông tin báo cáo một.")
    manager.append(thread_id, "user", "Báo cáo thứ hai về máy bay siêu thanh X-59 với nhiều thông số bay.")
    manager.append(thread_id, "assistant", "Đã nhận được thông tin báo cáo hai.")
    manager.append(thread_id, "user", "Báo cáo thứ ba về khí hậu toàn cầu và hiện tượng El Nino.")

    # Compaction should have fired at least once
    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    assert ctx["summary"] != ""
    assert "user:" in ctx["summary"] or "assistant:" in ctx["summary"]
    # Active messages should be bounded to recent messages
    assert len(ctx["messages"]) <= 3


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced agent remembers facts across sessions/threads, while baseline forgets."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_recall_test"
    thread_1 = "thread_session_1"
    thread_2 = "thread_session_2"

    intro = "Chào bạn, mình tên là DũngCT, hiện ở Huế và đang làm MLOps engineer."
    baseline.reply(user_id, thread_1, intro)
    advanced.reply(user_id, thread_1, intro)

    # Ask in a fresh thread
    question = "Nhắc lại giúp mình: mình tên gì, hiện ở đâu và làm nghề gì?"
    base_res = baseline.reply(user_id, thread_2, question)
    adv_res = advanced.reply(user_id, thread_2, question)

    # Baseline must forget
    base_text = base_res["reply"]
    assert "DũngCT" not in base_text
    assert "MLOps engineer" not in base_text
    assert "chưa có thông tin" in base_text.lower() or "rất tiếc" in base_text.lower()

    # Advanced must remember from User.md
    adv_text = adv_res["reply"]
    assert "DũngCT" in adv_text
    assert "Huế" in adv_text
    assert "MLOps engineer" in adv_text


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long conversation thread."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "long_context_user"
    thread_id = "long_test_thread"

    # Send 8 substantial turns into the same thread
    for i in range(8):
        msg = f"Đoạn tin tức công nghệ dài thứ {i} với các thông số phân tích hệ thống AI chi tiết và phân bổ tài nguyên bộ nhớ phức tạp."
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    # Advanced agent must have triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # Prompt tokens processed for advanced must be significantly lower than baseline
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)
    assert adv_prompt_tokens < base_prompt_tokens


def test_conflict_handling_and_noise_filtering(tmp_path: Path) -> None:
    """Verify correction handling (updating facts) and noise rejection (jokes, trips)."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "filter_test_user"

    # Turn 1: initial facts
    agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và làm backend engineer.")
    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Đà Nẵng"
    assert facts.get("profession") == "backend engineer"

    # Turn 2: correction of location to Huế
    agent.reply(user_id, "t2", "À mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Huế"

    # Turn 3: correction of profession to MLOps engineer
    agent.reply(user_id, "t3", "Mình chuyển sang làm MLOps engineer rồi.")
    facts = agent.profile_store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"

    # Turn 4: noise - trip to Hanoi and joke about product manager
    agent.reply(user_id, "t4", "Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày, và chuyển sang product manager chỉ là câu đùa.")
    facts = agent.profile_store.facts(user_id)
    # Location must remain Huế (not Hanoi), profession must remain MLOps engineer (not PM)
    assert facts.get("location") == "Huế"
    assert facts.get("profession") == "MLOps engineer"


def test_all_providers_instantiation() -> None:
    """Verify all 6 required providers instantiate properly."""
    providers = ["openai", "custom", "gemini", "anthropic", "ollama", "openrouter"]
    for p in providers:
        norm = normalize_provider(p)
        assert norm == p
        cfg = ProviderConfig(provider=p, model_name="test-model", api_key="test-key")
        model = build_chat_model(cfg)
        assert model is not None
