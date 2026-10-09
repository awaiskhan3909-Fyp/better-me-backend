-- ==============================================================================
-- BETTER ME FYP: PHASE 1 CLINICAL LONGITUDINAL MEMORY SCHEMA (SUPABASE POSTGRES)
-- ==============================================================================
-- Run this script in your Supabase SQL Editor:
-- https://supabase.com/dashboard/project/csceryxjmluvegbdekxe/sql/new

-- 1. Patient Clinical Profile (Longitudinal CBT Attributes)
CREATE TABLE IF NOT EXISTS public.patient_clinical_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    primary_triggers JSONB NOT NULL DEFAULT '[]'::jsonb,
    dominant_distortions JSONB NOT NULL DEFAULT '{}'::jsonb,
    core_beliefs JSONB NOT NULL DEFAULT '[]'::jsonb,
    effective_reframes JSONB NOT NULL DEFAULT '[]'::jsonb,
    active_homework TEXT,
    last_session_summary TEXT,
    total_sessions_completed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_patient_clinical_profile_user UNIQUE (user_id)
);

-- Indexes for lightning-fast retrieval
CREATE INDEX IF NOT EXISTS idx_clinical_profiles_user_id ON public.patient_clinical_profiles(user_id);

-- Enable Row Level Security (RLS)
ALTER TABLE public.patient_clinical_profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own clinical profile" ON public.patient_clinical_profiles;
CREATE POLICY "Users can view their own clinical profile"
ON public.patient_clinical_profiles FOR SELECT
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own clinical profile" ON public.patient_clinical_profiles;
CREATE POLICY "Users can insert their own clinical profile"
ON public.patient_clinical_profiles FOR INSERT
WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own clinical profile" ON public.patient_clinical_profiles;
CREATE POLICY "Users can update their own clinical profile"
ON public.patient_clinical_profiles FOR UPDATE
USING (auth.uid() = user_id);


-- 2. Episodic Therapy Memories (Past Reframes & Breakthrough Episodes)
CREATE TABLE IF NOT EXISTS public.episodic_therapy_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    conversation_id UUID,
    situation_context TEXT NOT NULL,
    distorted_thought TEXT NOT NULL,
    distortion_type VARCHAR(100) NOT NULL,
    rational_reframe TEXT NOT NULL,
    breakthrough_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id ON public.episodic_therapy_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_distortion ON public.episodic_therapy_memories(distortion_type);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_created_at ON public.episodic_therapy_memories(created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE public.episodic_therapy_memories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own therapy memories" ON public.episodic_therapy_memories;
CREATE POLICY "Users can view their own therapy memories"
ON public.episodic_therapy_memories FOR SELECT
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own therapy memories" ON public.episodic_therapy_memories;
CREATE POLICY "Users can insert their own therapy memories"
ON public.episodic_therapy_memories FOR INSERT
WITH CHECK (auth.uid() = user_id);
