-- ==============================================================================
-- BETTER ME FYP: PHASE 2 BECK'S 5-COLUMN THOUGHT RECORD SCHEMA (SUPABASE POSTGRES)
-- ==============================================================================
-- Run this script in your Supabase SQL Editor:
-- https://supabase.com/dashboard/project/csceryxjmluvegbdekxe/sql/new

CREATE TABLE IF NOT EXISTS public.cbt_thought_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    conversation_id UUID,
    situation TEXT NOT NULL,
    automatic_thought TEXT NOT NULL,
    initial_belief_rating INTEGER NOT NULL DEFAULT 80 CHECK (initial_belief_rating >= 0 AND initial_belief_rating <= 100),
    emotions JSONB NOT NULL DEFAULT '{}'::jsonb,
    distortion_type VARCHAR(100) NOT NULL,
    evidence_for TEXT NOT NULL,
    evidence_against TEXT NOT NULL,
    balanced_thought TEXT NOT NULL,
    outcome_belief_rating INTEGER NOT NULL DEFAULT 20 CHECK (outcome_belief_rating >= 0 AND outcome_belief_rating <= 100),
    outcome_emotions JSONB NOT NULL DEFAULT '{}'::jsonb,
    behavioral_action TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Indexes for rapid dashboard querying
CREATE INDEX IF NOT EXISTS idx_thought_records_user_id ON public.cbt_thought_records(user_id);
CREATE INDEX IF NOT EXISTS idx_thought_records_created_at ON public.cbt_thought_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_thought_records_distortion ON public.cbt_thought_records(distortion_type);

-- Row Level Security (RLS)
ALTER TABLE public.cbt_thought_records ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own thought records" ON public.cbt_thought_records;
CREATE POLICY "Users can view their own thought records"
ON public.cbt_thought_records FOR SELECT
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own thought records" ON public.cbt_thought_records;
CREATE POLICY "Users can insert their own thought records"
ON public.cbt_thought_records FOR INSERT
WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own thought records" ON public.cbt_thought_records;
CREATE POLICY "Users can update their own thought records"
ON public.cbt_thought_records FOR UPDATE
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own thought records" ON public.cbt_thought_records;
CREATE POLICY "Users can delete their own thought records"
ON public.cbt_thought_records FOR DELETE
USING (auth.uid() = user_id);
