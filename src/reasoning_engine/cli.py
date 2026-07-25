from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import get_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open Reasoning Engine CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest a document or directory")
    ingest.add_argument("path", type=Path)

    ask = commands.add_parser("ask", help="Ask the knowledge base")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = get_service()

    if args.command == "ingest":
        path: Path = args.path
        if not path.exists():
            raise SystemExit(f"Path does not exist: {path}")
        count = (
            service.knowledge_base.ingest_directory(path)
            if path.is_dir()
            else service.knowledge_base.ingest_paths([path])
        )
        print(f"Ingested {count} document(s) into {service.settings.collection_name}")
        return

    response = service.engine.query(args.question, top_k=args.top_k)
    if args.as_json:
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(response.answer)
        if response.citations:
            print("\nSources:")
            for citation in response.citations:
                print(f"[{citation.index}] {citation.source} (score={citation.score})")


if __name__ == "__main__":
    main()

