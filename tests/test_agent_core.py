import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from agent_core import graph
from agent_core.tools import llm, text as text_tools


def test_truncated_reply_is_never_emptied():
    """A capped reply without sentence punctuation is kept, not discarded."""
    run_on = "The Federal Reserve held rates steady while officials debated the budget"
    assert text_tools.cut_at_sentence_boundary(run_on) == run_on
    assert text_tools.cut_at_sentence_boundary(
        "One thing happened. Another thing happened. And then the tokens ran out"
    ) == "One thing happened. Another thing happened."


def test_english_ratio_flags_a_source_language_reply():
    assert text_tools.english_ratio("Iran is prepared for war, its minister said.") == 1.0
    assert text_tools.english_ratio("伊朗外长阿拉格齐表示已做好重新开战的准备。") < 0.1


def test_summarize_keeps_the_previous_draft_when_the_rewrite_is_unusable(monkeypatch):
    """A rewrite the model answers in Chinese must not end the run empty."""
    previous = "A wildfire broke out in southern California on Tuesday."
    monkeypatch.setattr(text_tools, "read_source", lambda *_args, **_kwargs: "source")
    monkeypatch.setattr(llm, "write_script", lambda *args, **kwargs: "伊朗外长表示已做好准备。")

    result = graph.summarize_node(
        {"source_file": "source.txt", "feedback": "too long", "attempts": 1,
         "summary": previous, "sentences": [{"index": 1, "text": previous}]}
    )

    # no summary in the update: langgraph leaves the previous draft in the state
    assert "summary" not in result
    assert result["attempts"] == 2
    assert result["errors"] and "not English" in result["errors"][0]


def test_summarize_retries_with_a_corrective_note_when_there_is_no_draft(monkeypatch):
    from agent_core import config

    monkeypatch.setattr(text_tools, "read_source", lambda *_args, **_kwargs: "source")
    monkeypatch.setattr(llm, "write_script", lambda *args, **kwargs: "伊朗外长表示已做好准备。")

    result = graph.summarize_node({"source_file": "source.txt", "feedback": "", "attempts": 0})

    assert result["summary"] == ""
    assert result["feedback"] == config.RETRY_FEEDBACK
    assert graph.route_after_summarize(result) == "retry"


def test_summary_generation_exposes_text(monkeypatch):
    source_text = "A wildfire broke out in southern California on Tuesday. Fire crews moved quickly to contain the blaze near a residential area."
    fake_summary = "A wildfire broke out in southern California on Tuesday. Fire crews moved quickly to contain the blaze near a residential area. Officials said the evacuation order remained in place overnight."

    monkeypatch.setattr(text_tools, "read_source", lambda *_args, **_kwargs: source_text)
    monkeypatch.setattr(llm, "write_script", lambda *args, **kwargs: fake_summary)
    monkeypatch.setattr(text_tools, "split_sentences", lambda text: [
        "A wildfire broke out in southern California on Tuesday.",
        "Fire crews moved quickly to contain the blaze near a residential area.",
        "Officials said the evacuation order remained in place overnight.",
    ])
    monkeypatch.setattr(text_tools, "write_sentence_files", lambda sentences: [
        {"index": 1, "text": sentences[0], "text_file": "news_001.txt"},
        {"index": 2, "text": sentences[1], "text_file": "news_002.txt"},
        {"index": 3, "text": sentences[2], "text_file": "news_003.txt"},
    ])

    result = graph.summarize_node({"source_file": "source.txt", "feedback": ""})
    print("SUMMARY_TEXT:")
    print(result["summary"])

    assert result["summary"] == fake_summary
    assert "California" in result["summary"]
    assert len(result["sentences"]) == 3


def test_feedback_generation_exposes_text(monkeypatch):
    source_text = "A wildfire broke out in southern California on Tuesday. Fire crews moved quickly to contain the blaze near a residential area."
    summary = "A wildfire broke out in southern California on Tuesday. Fire crews moved quickly to contain the blaze near a residential area."
    feedback = "Please keep the summary to a single factual paragraph and remove repetition."

    monkeypatch.setattr(text_tools, "read_source", lambda *_args, **_kwargs: source_text)
    # inside the MIN_WORDS..MAX_WORDS range, so review_node reaches the editor
    monkeypatch.setattr(text_tools, "count_words", lambda text: 300)
    monkeypatch.setattr(llm, "review_script", lambda script, source_text, model_manage=None: (False, feedback))

    result = graph.review_node({"summary": summary, "source_file": "source.txt"})
    print("FEEDBACK_TEXT:")
    print(result["feedback"])

    assert result["approved"] is False
    assert result["feedback"] == feedback
    assert "single factual paragraph" in result["feedback"]
