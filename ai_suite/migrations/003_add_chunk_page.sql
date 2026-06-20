-- Add page-number tracking to chunks for page-level citations
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page INTEGER;
