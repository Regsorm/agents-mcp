"""Чистая копия проекта для исполнителя на внешней модели и отдельный индекс по ней.

Исполнитель на внешней модели не должен видеть
того, что лежит в проекте вне контроля версий: .env с доступами, словари,
отчёты прогонов с настоящими значениями. И не должен ходить в общий индекс кода:
тот знает все репозитории машины и перечисляет их с путями в get_stats,
health и в ошибке «неизвестный алиас» — а всё это уходит внешней модели.

Поэтому:

* ``prepare`` — git worktree проекта в <AGENT_WORK_DIR>/<имя> от HEAD (в копии
  только файлы под контролем версий), отдельные демон и ``bsl-indexer serve``
  со своим CODE_INDEX_HOME на 127.0.0.1:8037 с единственным репозиторием под
  алиасом ``work`` — демон следит за копией, так что правки агента видны в
  индексе сразу; затем проверка, что в ответах индекса нет чужих путей;
* ``apply`` — перенос изменений копии (с новыми файлами) в рабочий каталог
  проекта через git apply, без фиксации: тесты и фиксация — в самом проекте;
* ``remove`` — остановка индекса и удаление копии.

Экземпляр индекса один на машину: новая ``prepare`` останавливает прежний.

Настройка переменными окружения:

* ``CODE_INDEX_EXE`` — путь к индексатору ``bsl-indexer`` (code-index); не
  задан — ищется в PATH;
* ``CODE_INDEX_MAIN_HOME`` — каталог конфигов общего индекса (daemon.toml,
  serve.toml) для проверки изоляции; не задан — каталог индексатора;
* ``AGENT_WORK_DIR`` — каталог копий; не задан — ``C:/Temp/agent-work``.

Запуск::

    python scripts/clean_copy.py prepare C:/Projects/my-app задача1 [--language python]
    python scripts/clean_copy.py apply   C:/Projects/my-app задача1
    python scripts/clean_copy.py remove  C:/Projects/my-app задача1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Индексатор code-index: переменная CODE_INDEX_EXE, иначе bsl-indexer из PATH.
EXE = os.environ.get("CODE_INDEX_EXE") or shutil.which("bsl-indexer") or ""
# Конфиги общего индекса — только чтобы собрать список чужих репозиториев для
# проверки изоляции; сам экземпляр копии их не читает. По умолчанию — каталог
# индексатора.
MAIN_HOME = Path(os.environ.get("CODE_INDEX_MAIN_HOME") or (Path(EXE).parent if EXE else "."))
КАТАЛОГ_КОПИЙ = Path("C:/Temp/agent-work") if os.name == "nt" else Path(tempfile.gettempdir()) / "agent-work"
КОПИИ = Path(os.environ.get("AGENT_WORK_DIR") or КАТАЛОГ_КОПИЙ)
# Свой CODE_INDEX_HOME экземпляра: свой демон и свой serve. Serve отдаёт
# содержимое только по путям, которые ведёт демон этого же дома («путь не
# отслеживается демоном» — проверено 11.09.2026), а общий демон трогать нельзя.
# Лежит рядом с копиями, но вне каталога любой из них: рабочий каталог
# агента — только его копия.
ДОМ = КОПИИ / "_index"
# Статусы, которыми serve отвечает, пока индекс не готов (TRANSIENT_STATUSES
# в исходниках code-index).
НЕ_ГОТОВ = ("not_started", "indexing", "error", "daemon_offline", "unknown_repo")
PORT = 8037
ALIAS = "work"
CREATE_NO_WINDOW = 0x08000000


def git(*args: str, cwd: Path | str, env: dict | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def git_байты(*args: str, cwd: Path | str, env: dict | None = None) -> bytes:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, env=env)
    if r.returncode != 0:
        ошибка = (r.stderr or r.stdout).decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git {' '.join(args)}: {ошибка}")
    return r.stdout


# ── экземпляр индекса ────────────────────────────────────────────────────────

def среда() -> dict:
    return dict(os.environ, CODE_INDEX_HOME=str(ДОМ))


def время_старта(pid: int) -> str:
    """Момент запуска процесса. Пара «PID + момент запуска» указывает ровно на один
    процесс: PID, освободившийся и доставшийся другому процессу (даже другому
    экземпляру индексатора, например общему), с записанной парой не совпадёт."""
    if os.name == "nt":
        команда = ["powershell", "-NoProfile", "-Command",
                   f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CreationDate"
                   ".ToString('o')"]
        r = subprocess.run(команда, capture_output=True, text=True, errors="replace",
                           creationflags=CREATE_NO_WINDOW)
    else:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                           capture_output=True, text=True, errors="replace")
    return r.stdout.strip()


def записать_pid(роль: str, pid: int) -> None:
    (ДОМ / f"clean_{роль}.pid").write_text(
        json.dumps({"pid": pid, "start": время_старта(pid)}), encoding="utf-8")


def прочитать_pid(роль: str) -> tuple[int, str] | None:
    try:
        запись = json.loads((ДОМ / f"clean_{роль}.pid").read_text(encoding="utf-8"))
        return int(запись["pid"]), str(запись["start"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def жив_индексатор(pid: int, старт: str) -> bool:
    """Жив ли ИМЕННО тот процесс индексатора, что запустил этот скрипт."""
    if pid <= 0 or not старт:
        return False
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, errors="replace")
    else:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                           capture_output=True, text=True, errors="replace")
    return "bsl-indexer" in r.stdout.lower() and время_старта(pid) == старт


def остановить_индекс() -> None:
    # Демон — штатно, через свой дом (POST /stop); общий демон при этом не
    # задевается: адрес берётся из daemon.json этого дома. Команда возвращается
    # раньше, чем демон закроет базу в копии, — без ожидания удаление копии
    # падает с «Invalid argument» (11.09.2026).
    if (ДОМ / "daemon.json").exists():
        subprocess.run([EXE, "daemon", "stop"], env=среда(), cwd=ДОМ,
                       capture_output=True, timeout=30)
        демон = прочитать_pid("daemon")
        for _ in range(20):
            if демон is None or not жив_индексатор(*демон):
                break
            time.sleep(0.5)
    for роль in ("serve", "daemon"):
        pid_файл = ДОМ / f"clean_{роль}.pid"
        if not pid_файл.exists():
            continue
        запись = прочитать_pid(роль)
        # Гасим только процесс, записанный этим скриптом: PID и момент запуска
        # обязаны совпасть, иначе процесс чужой — его не трогаем.
        if запись is not None and жив_индексатор(*запись):
            pid = запись[0]
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"[индекс] остановлен {роль}, PID {pid}")
        pid_файл.unlink()


def запустить(роль: str, аргументы: list[str]) -> subprocess.Popen:
    журнал = open(ДОМ / f"{роль}.log", "a", encoding="utf-8")
    параметры = dict(cwd=ДОМ, env=среда(), stdin=subprocess.DEVNULL,
                     stdout=журнал, stderr=subprocess.STDOUT)
    if os.name == "nt":
        параметры["creationflags"] = CREATE_NO_WINDOW
    p = subprocess.Popen([EXE, *аргументы], **параметры)
    записать_pid(роль, p.pid)
    return p


def запустить_индекс(копия: Path, язык: str) -> None:
    # Значения уходят в TOML: язык — только имя, путь — экранированной строкой.
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", язык):
        raise SystemExit(f"недопустимый --language: {язык!r}")
    ДОМ.mkdir(parents=True, exist_ok=True)
    путь = копия.as_posix()
    (ДОМ / "daemon.toml").write_text(
        "# Один репозиторий — чистая копия для исполнителя (clean_copy.py).\n"
        "[daemon]\nhttp_port = 0\n\n"
        f'[[paths]]\npath = {json.dumps(путь, ensure_ascii=False)}\nalias = "{ALIAS}"\nlanguage = "{язык}"\n',
        encoding="utf-8")
    (ДОМ / "serve.toml").write_text(
        "# Единственный репозиторий, локальный, никакой федерации.\n"
        '[me]\nip = "127.0.0.1"\n\n'
        f'[[paths]]\nalias = "{ALIAS}"\nip = "127.0.0.1"\nport = {PORT}\n',
        encoding="utf-8")
    # daemon.json прежнего запуска указал бы на мёртвый процесс.
    (ДОМ / "daemon.json").unlink(missing_ok=True)
    демон = запустить("daemon", ["daemon", "run"])
    serve = запустить("serve", [
        "serve", "--transport", "http", "--host", "127.0.0.1", "--port", str(PORT),
        "--config", str(ДОМ / "daemon.toml"), "--serve-config", str(ДОМ / "serve.toml")])
    начало = time.time()
    while time.time() - начало < 120:
        time.sleep(1)
        # `daemon run` запускает рабочий процесс отдельно и сам выходит с кодом 0
        # (проверено 11.09.2026); настоящий PID демон пишет в daemon.json.
        # Сбой — только ненулевой код запускающего процесса.
        for роль, p in (("daemon", демон), ("serve", serve)):
            if p.poll() not in (None, 0) or (роль == "serve" and p.poll() == 0):
                хвост = (ДОМ / f"{роль}.log").read_text(encoding="utf-8")[-1500:]
                остановить_индекс()
                raise SystemExit(f"{роль} завершился с кодом {p.returncode}:\n{хвост}")
        try:
            ответ = вызвать("list_files", {"repo": ALIAS, "limit": 1})
        except Exception:
            continue          # serve ещё не слушает порт
        if not any(f'\\"status\\":\\"{с}\\"' in ответ or f'"status":"{с}"' in ответ
                   for с in НЕ_ГОТОВ):
            pid_демона = json.loads((ДОМ / "daemon.json").read_text(encoding="utf-8"))["pid"]
            записать_pid("daemon", int(pid_демона))
            print(f"[индекс] готов за {time.time() - начало:.0f} с: демон PID {pid_демона}, "
                  f"serve PID {serve.pid}, порт {PORT}, алиас {ALIAS}")
            return
    остановить_индекс()
    raise SystemExit("индекс копии не готов за 120 с — см. журналы в " + str(ДОМ))


def вызвать(инструмент: str, аргументы: dict) -> str:
    """Вызов инструмента индекса по MCP; возвращает текст ответа целиком."""
    адрес = f"http://127.0.0.1:{PORT}/mcp"
    заголовки = {"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"}

    def post(тело: dict, сессия: str | None = None) -> tuple[str, str | None]:
        h = dict(заголовки)
        if сессия:
            h["Mcp-Session-Id"] = сессия
        req = urllib.request.Request(адрес, json.dumps(тело).encode(), h)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace"), r.headers.get("Mcp-Session-Id")

    _, сессия = post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "clean_copy", "version": "1"}}})
    post({"jsonrpc": "2.0", "method": "notifications/initialized"}, сессия)
    текст, _ = post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": инструмент, "arguments": аргументы}}, сессия)
    return текст


def чужие_метки() -> set[str]:
    """Пути и алиасы всех репозиториев общего индекса — их не должно быть в ответах."""
    метки: set[str] = set()
    for имя in ("daemon.toml", "serve.toml"):
        f = MAIN_HOME / имя
        if f.exists():
            t = f.read_text(encoding="utf-8", errors="replace")
            метки |= set(re.findall(r'^\s*(?:path|alias)\s*=\s*"([^"]+)"', t, re.M))
    метки.discard(ALIAS)
    return {м.replace("\\", "/").lower() for м in метки if len(м) > 3}


def проверить_изоляцию() -> None:
    чужие = чужие_метки()
    if not чужие:
        print(f"[изоляция] в {MAIN_HOME} нет конфигов общего индекса — сверять не с чем, "
              "проверка неполная (задайте CODE_INDEX_MAIN_HOME)")
    проверки = {
        "get_stats без repo": ("get_stats", {}),
        "health": ("health", {"repo": ALIAS}),
        "неизвестный алиас": ("get_stats", {"repo": "нет-такого-репо"}),
    }
    for название, (инструмент, арг) in проверки.items():
        ответ = вызвать(инструмент, арг)
        # Ответ — JSON с экранированными обратными косыми; сравниваем в одном виде.
        плоский = ответ.replace("\\\\", "/").replace("\\", "/").lower()
        найдено = sorted(м for м in чужие if м in плоский)
        if найдено:
            остановить_индекс()
            raise SystemExit(f"[изоляция] {название}: в ответе чужие репозитории {найдено}"
                             " — индекс остановлен")
        print(f"[изоляция] {название}: чужих путей нет")


# ── команды ──────────────────────────────────────────────────────────────────

def путь_копии(имя: str) -> Path:
    if (not имя or имя in (".", "..") or "/" in имя or "\\" in имя
            or ":" in имя or Path(имя).is_absolute()):
        raise SystemExit("имя копии должно быть одним непустым сегментом каталога")
    корень = КОПИИ.resolve()
    копия = (КОПИИ / имя).resolve()
    if корень not in копия.parents:
        raise SystemExit(f"копия {копия} должна находиться строго внутри {корень}")
    return копия


def общий_git_каталог(проект: Path) -> Path:
    проект = проект.resolve()
    значение = git("-C", str(проект), "rev-parse", "--git-common-dir", cwd=проект).strip()
    путь = Path(значение)
    if not путь.is_absolute():
        путь = проект / путь
    return путь.resolve()


def доверенный_git_dir(проект: Path, копия: Path) -> Path | None:
    каталог = общий_git_каталог(проект) / "worktrees"
    if not каталог.is_dir():
        return None
    git_файл = (копия / ".git").resolve()
    for запись in каталог.glob("*/gitdir"):
        if not запись.is_file():
            continue
        try:
            значение = Path(запись.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError):
            continue
        if not значение.is_absolute():
            значение = запись.parent / значение
        if значение.resolve() == git_файл:
            return запись.parent.resolve()
    return None


def проверить_git_файл(копия: Path, git_dir: Path) -> None:
    git_файл = копия / ".git"
    if not git_файл.is_file() or git_файл.is_symlink():
        raise SystemExit("копия не является рабочей копией этого проекта")
    try:
        строки = git_файл.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise SystemExit("копия не является рабочей копией этого проекта") from None
    if len(строки) != 1 or not строки[0].startswith("gitdir: "):
        raise SystemExit("копия не является рабочей копией этого проекта")
    значение = Path(строки[0][len("gitdir: "):])
    if not значение.is_absolute():
        значение = копия / значение
    if значение.resolve() != git_dir.resolve():
        raise SystemExit("файл .git копии подменён; перенос отменён")


def копия_в_списке_worktree(проект: Path, копия: Path) -> bool:
    данные = git_байты("-C", str(проект.resolve()), "worktree", "list", "--porcelain", "-z",
                       cwd=проект.resolve())
    for поле in данные.split(b"\0"):
        if поле.startswith(b"worktree "):
            путь = Path(os.fsdecode(поле[len(b"worktree "):])).resolve()
            if путь == копия:
                return True
    return False


def удалить_каталог(копия: Path) -> None:
    if os.name != "nt":
        shutil.rmtree(копия, onerror=lambda f, p, _: (os.chmod(p, 0o700), f(p)))
        return
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", wintypes.WORD), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", wintypes.LPVOID), ("lpszProgressTitle", wintypes.LPCWSTR)]

    операция = SHFILEOPSTRUCTW()
    операция.wFunc = 3  # FO_DELETE
    операция.pFrom = str(копия.resolve()) + "\0\0"
    операция.fFlags = 0x0040 | 0x0010 | 0x0004 | 0x0400
    код = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(операция))
    if код or операция.fAnyOperationsAborted:
        raise SystemExit(f"не удалось переместить {копия} в Корзину (код {код})")

def prepare(проект: Path, имя: str, язык: str) -> None:
    копия = путь_копии(имя)
    if копия.exists():
        raise SystemExit(f"копия {копия} уже есть — сначала remove")
    остановить_индекс()
    КОПИИ.mkdir(parents=True, exist_ok=True)
    # Хуки проекта при создании копии не выполняются — как и при переносе.
    with tempfile.TemporaryDirectory(prefix="clean-copy-hooks-") as пустые_hooks:
        git("-c", f"core.hooksPath={пустые_hooks}", "worktree", "add", "--detach",
            str(копия), "HEAD", cwd=проект)
    запустить_индекс(копия, язык)
    проверить_изоляцию()
    print(json.dumps({"work_dir": копия.as_posix(), "repo": ALIAS, "port": PORT},
                     ensure_ascii=False))
    print(f"--code-index-url http://127.0.0.1:{PORT}/mcp")


def apply(проект: Path, имя: str) -> None:
    копия = путь_копии(имя)
    git_dir = доверенный_git_dir(проект, копия)
    if git_dir is None:
        raise SystemExit("копия не является рабочей копией этого проекта")
    проверить_git_файл(копия, git_dir)
    # Индекс копии лежит внутри неё и в перенос попасть не должен. Исключение
    # «:!.code-index» в add не годится: если каталог уже в .gitignore проекта,
    # git отвечает ошибкой (11.09.2026). Поэтому добавить всё и снять индекс.
    # Фильтры clean/smudge из .gitattributes копии чужих команд не исполняют:
    # .gitattributes лишь называет фильтр, а команду задают конфиги git — проекта,
    # глобальный пользователя и переменные GIT_CONFIG_* процесса этого скрипта.
    # Всё это вне копии и задаётся тем, кто запускает скрипт, а не агентом — при
    # корнях файлового доступа агента, ограниченных копией (шаг 4 в README).
    # Системный конфиг отключён переменной ниже.
    окружение = dict(os.environ, GIT_CONFIG_NOSYSTEM="1")
    with tempfile.TemporaryDirectory(prefix="clean-copy-hooks-") as пустые_hooks:
        безопасные = ("-c", "core.fsmonitor=false", "-c", f"core.hooksPath={пустые_hooks}",
                      f"--git-dir={git_dir}", f"--work-tree={копия}")
        git(*безопасные, "add", "-A", cwd=копия, env=окружение)
        git(*безопасные, "rm", "--cached", "-r", "-q", "--ignore-unmatch", "--",
            ".code-index", cwd=копия, env=окружение)
        заплатка = git_байты(*безопасные, "diff", "--cached", "--binary", "HEAD",
                             cwd=копия, env=окружение)
    if not заплатка.strip():
        print("[перенос] изменений в копии нет")
        return
    файл = КОПИИ / f"{имя}.patch"
    файл.write_bytes(заплатка)
    print(git("apply", "--stat", str(файл), cwd=проект), end="")
    git("apply", "--whitespace=nowarn", str(файл), cwd=проект)
    print(f"[перенос] изменения в {проект}, не зафиксированы; заплатка — {файл}")


def remove(проект: Path, имя: str) -> None:
    копия = путь_копии(имя)
    была_рабочей_копией = False
    if копия.exists():
        была_рабочей_копией = (доверенный_git_dir(проект, копия) is not None
                               or копия_в_списке_worktree(проект, копия))
        if not была_рабочей_копией:
            raise SystemExit("копия не является рабочей копией этого проекта")
    остановить_индекс()
    if была_рабочей_копией and копия.exists():
        r = subprocess.run(["git", "worktree", "remove", "--force", str(копия)], cwd=проект,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        # Прерванное удаление снимает копию с учёта git, а каталог оставляет;
        # тогда git его уже не знает — каталог наш, удаляем сами.
        if r.returncode != 0 and копия.exists():
            print(f"[копия] git: {r.stderr.strip()} — удаляю каталог сам")
            удалить_каталог(копия)
    # Учётная запись ЭТОЙ копии в .git/worktrees, если git её не снял. Общий
    # `git worktree prune` не годится: он снимает и чужие рабочие копии проекта,
    # чей каталог сейчас недоступен.
    git_dir = доверенный_git_dir(проект, копия)
    if git_dir is not None and git_dir.parent == общий_git_каталог(проект) / "worktrees":
        shutil.rmtree(git_dir, ignore_errors=True)
    print(f"[копия] {копия} удалена")


def main() -> None:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument("команда", choices=("prepare", "apply", "remove"))
    разбор.add_argument("проект", type=Path, help="корень git-репозитория проекта")
    разбор.add_argument("имя", help="имя копии — каталог в AGENT_WORK_DIR")
    разбор.add_argument("--language", default="python",
                        help="язык индекса копии (как в daemon.toml)")
    a = разбор.parse_args()
    if a.команда != "apply" and not EXE:
        raise SystemExit("не найден индексатор: задайте CODE_INDEX_EXE "
                         "или добавьте bsl-indexer в PATH")
    if a.команда == "prepare":
        prepare(a.проект, a.имя, a.language)
    elif a.команда == "apply":
        apply(a.проект, a.имя)
    else:
        remove(a.проект, a.имя)


if __name__ == "__main__":
    main()
