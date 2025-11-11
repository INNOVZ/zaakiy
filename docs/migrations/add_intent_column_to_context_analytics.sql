-- Migration: Add intent column to context_analytics table
-- This allows for better querying and indexing of intent data
-- Run this migration in your Supabase SQL editor

-- Add intent column as JSONB (allows flexible schema and indexing)
ALTER TABLE context_analytics
ADD COLUMN IF NOT EXISTS intent JSONB;

-- Add GIN index for fast JSONB queries on intent
CREATE INDEX IF NOT EXISTS idx_context_analytics_intent_gin
ON context_analytics USING GIN (intent);

-- Add index for primary_intent queries (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_context_analytics_intent_primary
ON context_analytics ((intent->>'primary_intent'))
WHERE intent IS NOT NULL;

-- Add index for intent confidence queries
CREATE INDEX IF NOT EXISTS idx_context_analytics_intent_confidence
ON context_analytics ((intent->>'confidence'))
WHERE intent IS NOT NULL;

-- Optional: Migrate existing intent data from retrieval_stats to intent column
-- This updates existing records that have intent data in retrieval_stats
UPDATE context_analytics
SET intent = retrieval_stats->'intent'
WHERE retrieval_stats->'intent' IS NOT NULL
  AND intent IS NULL;

-- Add comment to column for documentation
COMMENT ON COLUMN context_analytics.intent IS 'User intent detection data including primary_intent, confidence, secondary_intents, and detection_method';
