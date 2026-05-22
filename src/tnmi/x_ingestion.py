from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from tnmi.ai import AIAnalyzer
from tnmi.contracts import NormalizedItem, SourceType, XHandleSource, XPost
from tnmi.language import detect_language
from tnmi.storage import (
    get_ai_analysis,
    get_source_checkpoint,
    save_ai_analysis,
    save_raw_item,
    save_source_checkpoint,
)


X_PROMPT_VERSION = "x-stance-v1"


class XClient(Protocol):
    def search_recent_posts(self, handle: str, *, since_id: str | None = None, max_results: int = 50) -> list[XPost]:
        ...


class InMemoryXClient:
    def __init__(self, *, posts_by_handle: dict[str, list[XPost]]) -> None:
        self.posts_by_handle = posts_by_handle
        self.requests: list[dict[str, object]] = []

    def search_recent_posts(self, handle: str, *, since_id: str | None = None, max_results: int = 50) -> list[XPost]:
        self.requests.append({"handle": handle, "since_id": since_id, "max_results": max_results})
        posts = self.posts_by_handle.get(handle, [])
        if since_id is not None:
            posts = [post for post in posts if _post_id_as_int(post.id) > _post_id_as_int(since_id)]
        return sorted(posts, key=lambda post: _post_id_as_int(post.id))[:max_results]


class TweepyXClient:
    def __init__(self, bearer_token: str) -> None:
        try:
            import tweepy
        except ImportError as exc:
            raise RuntimeError("tweepy is required for live X ingestion") from exc
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    def search_recent_posts(self, handle: str, *, since_id: str | None = None, max_results: int = 50) -> list[XPost]:
        response = self.client.search_recent_tweets(
            query=f"from:{handle} -is:retweet",
            since_id=since_id,
            max_results=max(10, min(max_results, 100)),
            tweet_fields=["created_at", "lang", "public_metrics"],
        )
        return [_tweet_to_post(handle, tweet) for tweet in (response.data or [])]


def _tweet_to_post(handle: str, tweet: Any) -> XPost:
    public_metrics = _value(tweet, "public_metrics") or {}
    metadata: dict[str, Any] = {}
    edit_history = _value(tweet, "edit_history_tweet_ids")
    if edit_history:
        metadata["edit_history_tweet_ids"] = edit_history
    return XPost(
        id=str(_value(tweet, "id")),
        handle=handle,
        text=str(_value(tweet, "text") or ""),
        created_at=_value(tweet, "created_at"),
        lang=_value(tweet, "lang"),
        public_metrics={str(key): int(value) for key, value in dict(public_metrics).items()},
        metadata=metadata,
    )


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "get"):
        value = obj.get(key)
        if value is not None:
            return value
    return getattr(obj, key, None)


def canonical_x_post_url(handle: str, post_id: str) -> str:
    return f"https://x.com/{handle}/status/{post_id}"


def normalize_x_post(post: XPost, source: XHandleSource) -> NormalizedItem:
    text = post.text.strip()
    return NormalizedItem(
        source_type=SourceType.X,
        source_name=source.source_name,
        source_url=post.url or canonical_x_post_url(source.handle, post.id),
        published_at=post.created_at,
        language=post.lang or detect_language(text),
        title=f"X post by {source.source_name}",
        raw_text_original=post.text,
        clean_text_original=text,
        metadata={
            **post.metadata,
            "platform": "x",
            "external_id": post.id,
            "handle": source.handle,
            "public_metrics": post.public_metrics,
        },
    )


@dataclass(frozen=True)
class XIngestionResult:
    handles_seen: int = 0
    handles_skipped: int = 0
    posts_seen: int = 0
    items_saved: int = 0
    analyses_saved: int = 0
    failures: int = 0


class DailyXPipeline:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        x_client: XClient,
        analyzer: AIAnalyzer,
    ) -> None:
        self.session_factory = session_factory
        self.x_client = x_client
        self.analyzer = analyzer

    def run(self, sources: list[XHandleSource], *, max_results: int = 50) -> XIngestionResult:
        handles_seen = 0
        handles_skipped = 0
        posts_seen = 0
        items_saved = 0
        analyses_saved = 0
        failures = 0

        with self.session_factory() as session:
            for source in sources:
                handles_seen += 1
                if not source.active:
                    handles_skipped += 1
                    continue

                source_key = source.source_name
                checkpoint = get_source_checkpoint(
                    session,
                    source_type=SourceType.X.value,
                    source_key=source_key,
                    cursor_name="since_id",
                )
                since_id = checkpoint.cursor_value if checkpoint else None
                handle_failures_before = failures
                processed_ids: list[str] = []

                try:
                    posts = self.x_client.search_recent_posts(
                        source.handle,
                        since_id=since_id,
                        max_results=max_results,
                    )
                except Exception:
                    failures += 1
                    continue

                for post in posts:
                    posts_seen += 1
                    try:
                        with session.begin_nested():
                            item = normalize_x_post(post, source)
                            if not item.clean_text_original:
                                raise ValueError("X post has no text")
                            raw = save_raw_item(session, item)
                            existing_analysis = get_ai_analysis(
                                session,
                                raw.id,
                                model_name=self.analyzer.model_name,
                                prompt_version=X_PROMPT_VERSION,
                            )
                            if existing_analysis:
                                items_saved += 1
                                processed_ids.append(post.id)
                                continue
                            analysis = self.analyzer.analyze(item)
                            save_ai_analysis(
                                session,
                                raw.id,
                                analysis,
                                model_name=self.analyzer.model_name,
                                prompt_version=X_PROMPT_VERSION,
                            )
                            items_saved += 1
                            analyses_saved += 1
                            processed_ids.append(post.id)
                    except Exception:
                        failures += 1

                if processed_ids and failures == handle_failures_before:
                    save_source_checkpoint(
                        session,
                        source_type=SourceType.X.value,
                        source_key=source_key,
                        cursor_name="since_id",
                        cursor_value=_max_post_id(processed_ids),
                        metadata={"posts_seen": len(posts)},
                    )

            session.commit()

        return XIngestionResult(
            handles_seen=handles_seen,
            handles_skipped=handles_skipped,
            posts_seen=posts_seen,
            items_saved=items_saved,
            analyses_saved=analyses_saved,
            failures=failures,
        )


def _post_id_as_int(post_id: str) -> int:
    try:
        return int(post_id)
    except ValueError:
        return 0


def _max_post_id(post_ids: list[str]) -> str:
    return str(max((_post_id_as_int(post_id) for post_id in post_ids), default=0))
