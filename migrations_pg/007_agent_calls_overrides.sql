-- 007_agent_calls_overrides.sql
-- Перекрытия настроек вызова (overrides) — канонический JSON. Старые строки и
-- вызовы без перекрытий хранят NULL.
ALTER TABLE agents_mcp.agent_calls
    ADD COLUMN IF NOT EXISTS overrides text;
