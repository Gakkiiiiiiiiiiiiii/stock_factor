from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from stock_factor.adapters.postgres.models import Base


class Database:
    def __init__(self, url: str | None = None) -> None:
        resolved = url or os.getenv("FACTOR_DATABASE_URL", "sqlite:///./stock_factor.db")
        arguments = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
        self.engine = create_engine(resolved, pool_pre_ping=True, connect_args=arguments)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        # Split deployments may already have the original compact paper table.
        # Apply additive columns without rebuilding or losing paper history.
        columns = {column["name"] for column in inspect(self.engine).get_columns("paper_state")}
        additions = {
            "order_history": "JSON NOT NULL DEFAULT '[]'",
            "fill_history": "JSON NOT NULL DEFAULT '[]'",
            "risk_events": "JSON NOT NULL DEFAULT '[]'",
        }
        with self.engine.begin() as connection:
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE paper_state ADD COLUMN {name} {definition}"))
