-- Manually mark upload as pending in Supabase
UPDATE uploads
SET status = 'pending', error_message = NULL
WHERE id = '2e256d5c-52e4-47f5-aeeb-1bb6dc7f1604';
