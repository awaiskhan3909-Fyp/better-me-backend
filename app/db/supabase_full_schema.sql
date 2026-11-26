-- =============================================================================
-- Better Me: Single Master Supabase PostgreSQL Schema & Auth Sync Query
-- Safe & 100% Idempotent (can be saved as your ONLY query in Supabase SQL Editor)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Core Users Table (UUID matches Supabase auth.users.id)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL DEFAULT '',
    full_name VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- Backfill existing Supabase Auth users into public.users with exact same UUID
INSERT INTO public.users (id, email, password_hash, full_name, role, is_active, created_at, updated_at)
SELECT
    au.id,
    LOWER(au.email),
    'supabase_managed',
    COALESCE(au.raw_user_meta_data->>'full_name', split_part(au.email, '@', 1)),
    'user',
    true,
    COALESCE(au.created_at, timezone('utc'::text, now())),
    timezone('utc'::text, now())
FROM auth.users au
WHERE au.email IS NOT NULL
ON CONFLICT (email) DO NOTHING;

-- Auto-sync trigger: any new signup/update in auth.users automatically mirrors to public.users
CREATE OR REPLACE FUNCTION public.handle_supabase_auth_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF NEW.email IS NULL THEN
        RETURN NEW;
    END IF;
    INSERT INTO public.users (id, email, password_hash, full_name, role, is_active, created_at, updated_at)
    VALUES (
        NEW.id,
        LOWER(NEW.email),
        'supabase_managed',
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
        'user',
        true,
        COALESCE(NEW.created_at, timezone('utc'::text, now())),
        timezone('utc'::text, now())
    )
    ON CONFLICT (email) DO UPDATE
    SET
        full_name = COALESCE(EXCLUDED.full_name, public.users.full_name),
        updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT OR UPDATE ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_supabase_auth_user();

-- 2. Conversations, Messages, Analyses, AI Responses & Session States
CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL DEFAULT 'New Session',
    current_risk_level VARCHAR(20) NOT NULL DEFAULT 'Safe',
    is_archived BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON public.conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON public.conversations(updated_at DESC);

CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    sender_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_messages_conversation_sequence UNIQUE (conversation_id, sequence_number)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON public.messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON public.messages(created_at DESC);

CREATE TABLE IF NOT EXISTS public.message_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL UNIQUE REFERENCES public.messages(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    safety_risk_level VARCHAR(20) NOT NULL,
    needs_safety_alert BOOLEAN NOT NULL DEFAULT false,
    safety_probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    distortion_class VARCHAR(50) NOT NULL,
    distortion_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    distortion_probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_message_analyses_conversation_id ON public.message_analyses(conversation_id);
CREATE INDEX IF NOT EXISTS idx_message_analyses_distortion ON public.message_analyses(distortion_class);

CREATE TABLE IF NOT EXISTS public.ai_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    user_message_id UUID REFERENCES public.messages(id) ON DELETE SET NULL,
    ai_message_id UUID REFERENCES public.messages(id) ON DELETE CASCADE,
    response_source VARCHAR(50) NOT NULL,
    response_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    response_content TEXT NOT NULL,
    cbt_data JSONB,
    llm_metadata JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_ai_responses_conversation_id ON public.ai_responses(conversation_id);

CREATE TABLE IF NOT EXISTS public.conversation_session_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL UNIQUE REFERENCES public.conversations(id) ON DELETE CASCADE,
    current_stage VARCHAR(50) NOT NULL DEFAULT 'check_in',
    stage_turn_count INTEGER NOT NULL DEFAULT 0,
    total_turns INTEGER NOT NULL DEFAULT 0,
    identified_trigger TEXT,
    automatic_thought TEXT,
    identified_distortion VARCHAR(100),
    distortion_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    distortion_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_gated BOOLEAN NOT NULL DEFAULT false,
    balanced_perspective TEXT,
    action_plan TEXT,
    asked_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
    session_goals JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 3. User Intake Assessments
CREATE TABLE IF NOT EXISTS public.user_intake_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    user_email TEXT,
    primary_focus JSONB NOT NULL DEFAULT '[]'::jsonb,
    distress_baseline INTEGER NOT NULL DEFAULT 0,
    familiar_distortions JSONB NOT NULL DEFAULT '[]'::jsonb,
    primary_goal VARCHAR(255) NOT NULL DEFAULT 'Reframing negative thoughts',
    safety_acknowledged BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_intake_user_id ON public.user_intake_assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_intake_user_email ON public.user_intake_assessments(user_email);

-- 4. Longitudinal Clinical Memory (Profiles + Episodic Memories)
CREATE TABLE IF NOT EXISTS public.patient_clinical_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    primary_triggers JSONB NOT NULL DEFAULT '[]'::jsonb,
    dominant_distortions JSONB NOT NULL DEFAULT '{}'::jsonb,
    core_beliefs JSONB NOT NULL DEFAULT '[]'::jsonb,
    effective_reframes JSONB NOT NULL DEFAULT '[]'::jsonb,
    active_homework TEXT,
    last_session_summary TEXT,
    total_sessions_completed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_clinical_profiles_user_id ON public.patient_clinical_profiles(user_id);

CREATE TABLE IF NOT EXISTS public.episodic_therapy_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES public.conversations(id) ON DELETE CASCADE,
    situation_context TEXT NOT NULL,
    distorted_thought TEXT NOT NULL,
    distortion_type VARCHAR(100) NOT NULL,
    rational_reframe TEXT NOT NULL,
    breakthrough_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id ON public.episodic_therapy_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_conv_id ON public.episodic_therapy_memories(conversation_id);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_created_at ON public.episodic_therapy_memories(created_at DESC);

-- 5. Beck's 5-Column Thought Records
CREATE TABLE IF NOT EXISTS public.cbt_thought_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES public.conversations(id) ON DELETE SET NULL,
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
CREATE INDEX IF NOT EXISTS idx_thought_records_user_id ON public.cbt_thought_records(user_id);
CREATE INDEX IF NOT EXISTS idx_thought_records_created_at ON public.cbt_thought_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_thought_records_distortion ON public.cbt_thought_records(distortion_type);

-- 6. CBT Practice Exercises & Stress Relief Activities
CREATE TABLE IF NOT EXISTS public.cbt_exercise_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    distortion_category VARCHAR(100) NOT NULL,
    exercise_id VARCHAR(100) NOT NULL,
    exercise_title VARCHAR(255) NOT NULL,
    scenario TEXT NOT NULL,
    answers JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_completed BOOLEAN NOT NULL DEFAULT true,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_user_exercise_submission UNIQUE (user_id, exercise_id)
);
CREATE INDEX IF NOT EXISTS idx_cbt_exercises_user_id ON public.cbt_exercise_submissions(user_id);

CREATE TABLE IF NOT EXISTS public.stress_relief_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    activity_name VARCHAR(100) NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    is_completed BOOLEAN NOT NULL DEFAULT true,
    notes TEXT,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);
CREATE INDEX IF NOT EXISTS idx_stress_relief_user_id ON public.stress_relief_activities(user_id);

-- 7. Automatic Legacy Schema Cleanup (via pg_constraint — catches auth.users FKs & TEXT user_id)
DO $$
DECLARE
    r RECORD;
BEGIN
    -- Migrate legacy TEXT user_id on user_intake_assessments if present
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'user_intake_assessments'
          AND column_name = 'user_id' AND data_type <> 'uuid'
    ) THEN
        DELETE FROM public.user_intake_assessments
        WHERE user_id !~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$';
        ALTER TABLE public.user_intake_assessments ALTER COLUMN user_id TYPE UUID USING user_id::uuid;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'user_intake_assessments' AND column_name = 'user_email'
    ) THEN
        ALTER TABLE public.user_intake_assessments ADD COLUMN user_email TEXT;
    END IF;

    -- Drop any legacy foreign keys in public schema still pointing to auth.users
    FOR r IN (
        SELECT con.conname, rel.relname
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace ns ON ns.oid = rel.relnamespace
        JOIN pg_class frel ON frel.oid = con.confrelid
        JOIN pg_namespace fns ON fns.oid = frel.relnamespace
        WHERE con.contype = 'f' AND ns.nspname = 'public' AND fns.nspname = 'auth'
    ) LOOP
        EXECUTE format('ALTER TABLE public.%I DROP CONSTRAINT IF EXISTS %I', r.relname, r.conname);
    END LOOP;
END $$;

-- 8. Row Level Security (Per-user isolation for direct Supabase client access)
ALTER TABLE public.user_intake_assessments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow read and write access for intake assessments" ON public.user_intake_assessments;
DROP POLICY IF EXISTS "Users manage own intake assessments" ON public.user_intake_assessments;
CREATE POLICY "Users manage own intake assessments" ON public.user_intake_assessments
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

ALTER TABLE public.cbt_thought_records ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view their own thought records" ON public.cbt_thought_records;
DROP POLICY IF EXISTS "Users can insert their own thought records" ON public.cbt_thought_records;
DROP POLICY IF EXISTS "Users can update their own thought records" ON public.cbt_thought_records;
DROP POLICY IF EXISTS "Users can delete their own thought records" ON public.cbt_thought_records;
CREATE POLICY "Users can view their own thought records" ON public.cbt_thought_records FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own thought records" ON public.cbt_thought_records FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own thought records" ON public.cbt_thought_records FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own thought records" ON public.cbt_thought_records FOR DELETE USING (auth.uid() = user_id);

ALTER TABLE public.patient_clinical_profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view their own clinical profile" ON public.patient_clinical_profiles;
DROP POLICY IF EXISTS "Users can insert their own clinical profile" ON public.patient_clinical_profiles;
DROP POLICY IF EXISTS "Users can update their own clinical profile" ON public.patient_clinical_profiles;
CREATE POLICY "Users can view their own clinical profile" ON public.patient_clinical_profiles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own clinical profile" ON public.patient_clinical_profiles FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own clinical profile" ON public.patient_clinical_profiles FOR UPDATE USING (auth.uid() = user_id);

ALTER TABLE public.episodic_therapy_memories ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view their own therapy memories" ON public.episodic_therapy_memories;
DROP POLICY IF EXISTS "Users can insert their own therapy memories" ON public.episodic_therapy_memories;
CREATE POLICY "Users can view their own therapy memories" ON public.episodic_therapy_memories FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own therapy memories" ON public.episodic_therapy_memories FOR INSERT WITH CHECK (auth.uid() = user_id);
