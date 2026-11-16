"""Pin a selected narrator to each game session."""
import sqlalchemy as sa
from alembic import op

revision = "0015_session_narrator"
down_revision = "0014_turn_retry_diagnostics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "game_sessions",
        sa.Column("narrator", sa.String(32), server_default="local_qwen", nullable=False),
    )
    op.create_check_constraint(
        "ck_game_session_narrator",
        "game_sessions",
        "narrator IN ('local_qwen', 'gpt6_luna_high')",
    )


def downgrade() -> None:
    has_luna_sessions = op.get_bind().exec_driver_sql(
        "SELECT EXISTS (SELECT 1 FROM game_sessions WHERE narrator <> 'local_qwen')"
    ).scalar_one()
    if has_luna_sessions:
        raise RuntimeError("cannot discard narrator selections from existing sessions")
    op.drop_constraint("ck_game_session_narrator", "game_sessions", type_="check")
    op.drop_column("game_sessions", "narrator")
