"""Apply explicit demo curation for first-screen articles.

The small local LLM is useful for a demo, but it can over-classify political
stories. These overrides keep the first screen operational and evidence-based.
They update only the current prompt-version rows for known raw_item IDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sqlalchemy import select

from tnmi.ai import PROMPT_VERSION
from tnmi.config import Settings
from tnmi.storage import AIAnalysisRecord, create_session_factory, init_db


OVERRIDES: dict[int, dict[str, object]] = {
    301: {
        "government_relevance": "medium",
        "stance_toward_government": "neutral",
        "tvk_relevance": "low",
        "tvk_portrayal": "neutral",
        "sentiment": "neutral",
        "target": "Political alliance discourse",
        "political_actors": ["MDMK", "DMK alliance"],
        "department": "Political monitoring",
        "district": "Coimbatore",
        "topic": "MDMK symbol dispute",
        "issue_category": "political_strategy",
        "people_issue": False,
        "public_issue": "alliance symbol dispute",
        "severity": "low",
        "party_action": "",
        "people_impact": "",
        "summary_english": (
            "MDMK leader Durai Vaiko said MDMK candidates should not contest on another "
            "party symbol."
        ),
        "root_cause": (
            "Evidence reports Durai Vaiko's position that MDMK candidates should not contest "
            "on another party symbol."
        ),
        "recommended_step": (
            "Media monitoring desk: track whether the DMK-alliance symbol debate creates a TVK "
            "narrative opening; no public response now."
        ),
        "action_owner": "Media monitoring desk",
        "action_type": "monitor",
        "action_priority": "low",
        "needs_human_review": False,
    },
    300: {
        "government_relevance": "high",
        "stance_toward_government": "negative",
        "tvk_relevance": "medium",
        "tvk_portrayal": "neutral",
        "sentiment": "negative",
        "target": "Tamil Nadu Electricity Board",
        "political_actors": ["Electricity board officials"],
        "department": "Electricity board / anti-corruption",
        "district": "Chennai",
        "topic": "Electricity board hard disk theft",
        "issue_category": "civic_services",
        "people_issue": True,
        "public_issue": "electricity board corruption concern",
        "severity": "high",
        "party_action": "",
        "people_impact": (
            "People are already affected by unannounced power cuts while sensitive TNEB records "
            "are reportedly missing."
        ),
        "summary_english": (
            "A TNEB office reportedly lost hard disks linked to tenders and inquiries amid "
            "power-cut complaints."
        ),
        "root_cause": (
            "Evidence links public power-cut distress with alleged theft of hard disks containing "
            "tender, procurement and inquiry records."
        ),
        "recommended_step": (
            "Policy research team: verify police complaint, missing hard disks, tender records and "
            "consumer outage impact before TVK statement."
        ),
        "action_owner": "Policy research team",
        "action_type": "evidence_review",
        "action_priority": "high",
        "needs_human_review": True,
    },
    299: {
        "government_relevance": "medium",
        "stance_toward_government": "mixed",
        "tvk_relevance": "high",
        "tvk_portrayal": "mixed",
        "sentiment": "neutral",
        "target": "TVK MLA Pallavi and Chennai civic leadership",
        "political_actors": ["TVK", "MLA", "DMK (opposition)"],
        "department": "School education / civic administration",
        "district": "Chennai",
        "topic": "TVK-DMK school opening protocol dispute",
        "issue_category": "party_governance",
        "people_issue": False,
        "public_issue": "party optics and protocol dispute",
        "severity": "medium",
        "party_action": (
            "Confirm event protocol and MLA account before deciding whether TVK should clarify "
            "or stay silent."
        ),
        "people_impact": "",
        "summary_english": (
            "A school-opening protocol dispute between Chennai Mayor Priya and TVK MLA Pallavi "
            "created party optics risk."
        ),
        "root_cause": (
            "Evidence reports a protocol tussle between Chennai Mayor Priya and TVK MLA Pallavi; "
            "the article frames it as a DMK-versus-TVK representatives' ego clash."
        ),
        "recommended_step": (
            "TVK leadership office: speak with MLA Pallavi and the event team, verify invitation "
            "and protocol chronology with the school, then choose silent monitoring or a short "
            "fact-based clarification."
        ),
        "action_owner": "TVK leadership office",
        "action_type": "internal_review",
        "action_priority": "medium",
        "needs_human_review": True,
    },
    298: {
        "government_relevance": "none",
        "stance_toward_government": "neutral",
        "tvk_relevance": "none",
        "tvk_portrayal": "neutral",
        "sentiment": "neutral",
        "target": "Out of scope",
        "political_actors": [],
        "department": "Media monitoring",
        "district": "Tamil Nadu",
        "topic": "Income tax cash reporting rule",
        "issue_category": "out-of-scope",
        "people_issue": False,
        "public_issue": "",
        "severity": "low",
        "party_action": "",
        "people_impact": "",
        "summary_english": "Income Tax Department cash-reporting guidance is not a Tamil Nadu public issue.",
        "root_cause": "",
        "recommended_step": "",
        "action_owner": "Media monitoring desk",
        "action_type": "monitor",
        "action_priority": "low",
        "needs_human_review": False,
    },
    297: {
        "government_relevance": "medium",
        "stance_toward_government": "neutral",
        "tvk_relevance": "medium",
        "tvk_portrayal": "neutral",
        "sentiment": "negative",
        "target": "Chennai civic safety",
        "political_actors": ["Chennai civic administration"],
        "department": "Public safety / fire and rescue",
        "district": "Chennai",
        "topic": "Pallikaranai dump-yard fire",
        "issue_category": "public_safety",
        "people_issue": True,
        "public_issue": "public safety incident",
        "severity": "high",
        "party_action": "",
        "people_impact": (
            "Smoke and fire risk affected the Pallikaranai area after seized vehicles and RDF "
            "plastic waste reportedly caught fire."
        ),
        "summary_english": (
            "A fire broke out at Pallikaranai dump yard where RDF plastic waste and seized "
            "vehicles were stored."
        ),
        "root_cause": (
            "Evidence points to RDF plastic waste and stored seized vehicles at the municipal "
            "dump yard as the safety risk."
        ),
        "recommended_step": (
            "Chennai district field team: verify fire cause, nearby resident impact, waste-storage "
            "controls and civic response before statement."
        ),
        "action_owner": "Chennai district field team",
        "action_type": "field_verification",
        "action_priority": "high",
        "needs_human_review": True,
    },
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    args = parser.parse_args(argv)

    session_factory = create_session_factory(Settings().database_url)
    init_db(session_factory)
    updated = 0
    with session_factory() as session:
        for raw_item_id, fields in OVERRIDES.items():
            rows = session.scalars(
                select(AIAnalysisRecord).where(
                    AIAnalysisRecord.raw_item_id == raw_item_id,
                    AIAnalysisRecord.prompt_version == args.prompt_version,
                )
            ).all()
            for row in rows:
                for key, value in fields.items():
                    setattr(row, key, value)
                updated += 1
        session.commit()
    print(f"updated={updated} prompt_version={args.prompt_version}")


if __name__ == "__main__":
    main()
