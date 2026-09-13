-- 002_agent_calls_result_path.sql
-- Путь файла-итога agent_run. Нужен стартовой пометке осиротевшего вызова:
-- ожидатель должен получить конверт ровно там, где его ждёт.
ALTER TABLE agent_calls ADD COLUMN result_path TEXT;
