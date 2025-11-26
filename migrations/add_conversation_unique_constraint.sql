-- Migration: Add unique constraint to prevent duplicate conversations
-- This constraint is required for the UPSERT fix to work properly

-- Add unique constraint on (session_id, org_id, chatbot_id)
-- This ensures only one conversation exists per session/org/chatbot combination
ALTER TABLE conversations
ADD CONSTRAINT unique_session_org_chatbot
UNIQUE (session_id, org_id, chatbot_id);

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_conversations_session_org_chatbot
ON conversations (session_id, org_id, chatbot_id);

-- Note: If you get an error about duplicate rows, you need to clean up duplicates first:
--
-- Step 1: Find duplicates
-- SELECT session_id, org_id, chatbot_id, COUNT(*) as count
-- FROM conversations
-- GROUP BY session_id, org_id, chatbot_id
-- HAVING COUNT(*) > 1;
--
-- Step 2: Keep only the oldest conversation for each duplicate group
-- DELETE FROM conversations
-- WHERE id NOT IN (
--     SELECT MIN(id)
--     FROM conversations
--     GROUP BY session_id, org_id, chatbot_id
-- );
--
-- Step 3: Then run this migration
