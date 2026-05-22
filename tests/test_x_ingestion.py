from datetime import datetime, timezone

from sqlalchemy import func, select

from tests.test_storage import make_analysis
from tnmi.contracts import SourceType, XHandleSource, XPost
from tnmi.storage import (
    AIAnalysisRecord,
    RawItemRecord,
    create_session_factory,
    get_source_checkpoint,
    init_db,
    save_source_checkpoint,
)
from tnmi.x_ingestion import DailyXPipeline, InMemoryXClient, normalize_x_post


class CountingAnalyzer:
    model_name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, item):
        self.calls += 1
        return make_analysis().model_copy(update={"summary_original": item.clean_text_original[:80]})


def make_post(post_id: str = "101", text: str = "Tamil Nadu government scheme update") -> XPost:
    return XPost(
        id=post_id,
        handle="ExampleTNNews",
        text=text,
        created_at=datetime(2026, 5, 22, 8, 0, tzinfo=timezone.utc),
        lang="en",
        public_metrics={"like_count": 10, "reply_count": 2, "retweet_count": 3},
        metadata={"edit_history_tweet_ids": [post_id]},
    )


def test_normalize_x_post_preserves_metrics_and_source_url():
    source = XHandleSource(handle="ExampleTNNews")
    item = normalize_x_post(make_post(), source)

    assert item.source_type == SourceType.X
    assert item.source_name == "@ExampleTNNews"
    assert item.source_url == "https://x.com/ExampleTNNews/status/101"
    assert item.language == "en"
    assert item.clean_text_original == "Tamil Nadu government scheme update"
    assert item.metadata["public_metrics"]["like_count"] == 10
    assert item.metadata["external_id"] == "101"


def test_daily_x_pipeline_saves_posts_analyzes_and_updates_since_id(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    init_db(session_factory)
    client = InMemoryXClient(posts_by_handle={"ExampleTNNews": [make_post("100"), make_post("101")]})
    analyzer = CountingAnalyzer()
    pipeline = DailyXPipeline(session_factory=session_factory, x_client=client, analyzer=analyzer)

    result = pipeline.run([XHandleSource(handle="ExampleTNNews")], max_results=50)

    with session_factory() as session:
        raw_count = session.scalar(select(func.count()).select_from(RawItemRecord))
        analysis_count = session.scalar(select(func.count()).select_from(AIAnalysisRecord))
        checkpoint = get_source_checkpoint(
            session,
            source_type="x",
            source_key="@ExampleTNNews",
            cursor_name="since_id",
        )

    assert result.handles_seen == 1
    assert result.posts_seen == 2
    assert result.items_saved == 2
    assert result.analyses_saved == 2
    assert raw_count == 2
    assert analysis_count == 2
    assert checkpoint is not None
    assert checkpoint.cursor_value == "101"
    assert analyzer.calls == 2


def test_daily_x_pipeline_rerun_uses_since_id_and_does_not_reanalyze_existing_rows(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    init_db(session_factory)
    client = InMemoryXClient(posts_by_handle={"ExampleTNNews": [make_post("100"), make_post("101")]})
    analyzer = CountingAnalyzer()
    pipeline = DailyXPipeline(session_factory=session_factory, x_client=client, analyzer=analyzer)
    sources = [XHandleSource(handle="ExampleTNNews")]

    first = pipeline.run(sources, max_results=50)
    second = pipeline.run(sources, max_results=50)

    with session_factory() as session:
        raw_count = session.scalar(select(func.count()).select_from(RawItemRecord))
        analysis_count = session.scalar(select(func.count()).select_from(AIAnalysisRecord))

    assert first.analyses_saved == 2
    assert second.posts_seen == 0
    assert second.analyses_saved == 0
    assert analyzer.calls == 2
    assert raw_count == 2
    assert analysis_count == 2
    assert client.requests[-1] == {"handle": "ExampleTNNews", "since_id": "101", "max_results": 50}


def test_daily_x_pipeline_uses_existing_checkpoint_before_first_request(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    init_db(session_factory)
    with session_factory() as session:
        save_source_checkpoint(
            session,
            source_type="x",
            source_key="@ExampleTNNews",
            cursor_name="since_id",
            cursor_value="100",
        )
        session.commit()
    client = InMemoryXClient(posts_by_handle={"ExampleTNNews": [make_post("99"), make_post("101")]})
    pipeline = DailyXPipeline(session_factory=session_factory, x_client=client, analyzer=CountingAnalyzer())

    result = pipeline.run([XHandleSource(handle="ExampleTNNews")], max_results=50)

    assert result.posts_seen == 1
    assert client.requests[0] == {"handle": "ExampleTNNews", "since_id": "100", "max_results": 50}


def test_daily_x_pipeline_skips_inactive_handles(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'x.db'}")
    init_db(session_factory)
    pipeline = DailyXPipeline(
        session_factory=session_factory,
        x_client=InMemoryXClient(posts_by_handle={}),
        analyzer=CountingAnalyzer(),
    )

    result = pipeline.run([XHandleSource(handle="ExampleTNNews", active=False)], max_results=50)

    assert result.handles_seen == 1
    assert result.handles_skipped == 1
