"""003_add_user_role

Revision ID: 3d8a1c9e4b21
Revises: 257932de4454
Create Date: 2026-11-14 21:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3d8a1c9e4b21'
down_revision: Union[str, None] = '257932de4454'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('role', sa.String(length=20), server_default='user', nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('role')
