"""002_json_to_jsonb_variant

Revision ID: 257932de4454
Revises: 947c26fdf063
Create Date: 2026-09-23 14:11:39.712879

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '257932de4454'
down_revision: Union[str, None] = '947c26fdf063'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Cross-database JSON type definition (JSON for SQLite, JSONB for PostgreSQL)
JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table('message_analyses', schema=None) as batch_op:
        batch_op.alter_column('safety_probabilities', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=False)
        batch_op.alter_column('distortion_probabilities', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=False)
        batch_op.alter_column('entities', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=False)
        batch_op.alter_column('model_metadata', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=False)

    with op.batch_alter_table('ai_responses', schema=None) as batch_op:
        batch_op.alter_column('cbt_data', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=True)
        batch_op.alter_column('llm_metadata', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=True)
        batch_op.alter_column('metadata', type_=JSON_TYPE, existing_type=sa.JSON(), existing_nullable=True)


def downgrade() -> None:
    with op.batch_alter_table('ai_responses', schema=None) as batch_op:
        batch_op.alter_column('metadata', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=True)
        batch_op.alter_column('llm_metadata', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=True)
        batch_op.alter_column('cbt_data', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=True)

    with op.batch_alter_table('message_analyses', schema=None) as batch_op:
        batch_op.alter_column('model_metadata', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=False)
        batch_op.alter_column('entities', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=False)
        batch_op.alter_column('distortion_probabilities', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=False)
        batch_op.alter_column('safety_probabilities', type_=sa.JSON(), existing_type=JSON_TYPE, existing_nullable=False)
