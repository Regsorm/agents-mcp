-- 006_agent_calls_result_path.sql
-- Путь файла-итога agent_run. Старые строки и обычный invoke хранят NULL.
ALTER TABLE agents_mcp.agent_calls
    ADD COLUMN IF NOT EXISTS result_path text;
