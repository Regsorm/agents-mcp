-- 003_agent_calls_overrides.sql
-- Перекрытия настроек вызова (overrides) — канонический JSON. Старые строки и
-- вызовы без перекрытий хранят NULL.
ALTER TABLE agent_calls
    ADD COLUMN overrides TEXT;
