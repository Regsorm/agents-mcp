"""Чистая копия проекта для исполнителя на внешней модели и отдельный индекс по ней.

Исполнитель на внешней модели не должен видеть
того, что лежит в проекте вне контроля версий: .env с доступами, словари,
отчёты прогонов с настоящими значениями. И не должен ходить в общий индекс кода:
тот знает все репозитории машины и перечисляет их с путями в get_stats,
health и в ошибке «неизвестный алиас» — а всё это уходит внешней модели.

Поэтому:

* ``prepare`` — git worktree проекта в <AGENT_WORK_DIR>/<имя> на новой ветке
  ``agent/<имя>`` от указанного коммита (в копии только файлы под контролем
  версий), отдельные демон и ``bsl-indexer serve`` со своим CODE_INDEX_HOME в
  <AGENT_WORK_DIR>/_indexes/<имя> на 127.0.0.1:<порт> с единственным
  репозиторием под указанным алиасом — демон следит за копией, так что правки
  агента видны в индексе сразу; затем проверка, что в ответах индекса нет чужих
  путей. Порт (можно ``--port auto``) и алиас обязательны и у каждой копии свои, поэтому копии разных
  задач живут одновременно, каждая со своим индексом. Напоследок каталог копии
  дописывается секцией ``[[local]]`` в конфиг хука ``code-index-guard``: обычное
  чтение копии (Read, Grep, cat/grep/ls в оболочке) хук отклоняет и отправляет в
  индекс копии. В общий ``daemon.toml`` индексатора при этом не попадает ничего;
* ``commit`` — фиксация правок копии на её ветке ``agent/<имя>`` (с ветки работу
  потом сливают в main — это вне этого скрипта); необязательная зона ``--files``
  ограничивает, какие пути вообще можно зафиксировать;
* ``apply`` — перенос изменений копии (с новыми файлами) в рабочий каталог
  проекта через git apply, без фиксации: тесты и фиксация — в самом проекте;
* ``list`` — сведения о копиях: по строке JSON на каждую;
* ``remove`` — остановка индекса, удаление копии, её дома индекса и её секции в
  конфиге хука; ветка ``agent/<имя>`` остаётся — на ней зафиксированная работа.
  ``remove <проект> --all`` снимает все копии проекта.

Копии, созданные прежней версией скрипта (общий каталог
<AGENT_WORK_DIR>/_index и постоянные 8037 и ``work``), эта версия не видит и не
снимает. Снять такую копию вручную::

    CODE_INDEX_HOME=<AGENT_WORK_DIR>/_index bsl-indexer daemon stop
    # затем остановить процесс serve по PID из <AGENT_WORK_DIR>/_index/clean_serve.pid
    git worktree remove --force <копия>

Сведения о копии — <AGENT_WORK_DIR>/_indexes/<имя>/copy.json: имя, проект, путь
копии, ветка, коммит-начало, порт, алиас, язык и зона files.

Настройка переменными окружения:

* ``CODE_INDEX_EXE`` — путь к индексатору ``bsl-indexer`` (code-index); не
  задан — ищется в PATH;
* ``CODE_INDEX_MAIN_HOME`` — каталог конфигов общего индекса (daemon.toml,
  serve.toml) для проверки изоляции; не задан — каталог индексатора;
* ``AGENT_WORK_DIR`` — каталог копий; не задан — ``C:/Temp/agent-work``;
* ``CODE_INDEX_GUARD_CONFIG`` — конфиг хука ``code-index-guard``, куда пишется
  секция копии; не задан — ``~/.claude/hooks/code-index-guard.toml``.

Запуск::

    python scripts/clean_copy.py prepare C:/Projects/my-app задача1 --port auto --alias work --language python --files "src/*"
    python scripts/clean_copy.py commit  C:/Projects/my-app задача1 --message "Правка по задаче"
    python scripts/clean_copy.py list
    python scripts/clean_copy.py apply   C:/Projects/my-app задача1
    python scripts/clean_copy.py remove  C:/Projects/my-app задача1
    python scripts/clean_copy.py remove  C:/Projects/my-app --all
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import signal
import socket
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
# Конфиг хука code-index-guard: в каких каталогах он отклоняет обычное чтение
# (Read, Grep, cat/grep/ls в оболочке) в пользу инструментов индекса. Каталог
# копии дописывается туда секцией [[local]] — в общий daemon.toml индексатора
# при этом не попадает ничего, чужие секции конфига не трогаются.
КОНФИГ_ХУКА = Path(os.environ.get("CODE_INDEX_GUARD_CONFIG")
                   or Path.home() / ".claude" / "hooks" / "code-index-guard.toml")
МЕТКА_НАЧАЛА = "# >>> clean_copy"
МЕТКА_КОНЦА = "# <<< clean_copy"
# Статусы, которыми serve отвечает, пока индекс не готов (TRANSIENT_STATUSES
# в исходниках code-index).
НЕ_ГОТОВ = ("not_started", "indexing", "error", "daemon_offline", "unknown_repo")
CREATE_NO_WINDOW = 0x08000000
ПОПЫТОК_ПОРТА = 3


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


def безопасный_git(копия: Path, git_dir: Path, пустые_hooks: str) -> tuple:
    """Параметры git, при которых команды работают ровно с этой копией и не
    исполняют чужих хуков: свой каталог учёта, свой рабочий каталог, пустой
    hooksPath и выключенный fsmonitor."""
    return ("-c", "core.fsmonitor=false", "-c", f"core.hooksPath={пустые_hooks}",
            f"--git-dir={git_dir}", f"--work-tree={копия}")


# ── экземпляр индекса ────────────────────────────────────────────────────────

def каталог_домов() -> Path:
    """Каталог домов индекса: у каждой копии свой дом (см. `дом`). Старый общий
    каталог КОПИИ/_index эта версия не читает и не трогает НИКОГДА: там может
    работать копия прежней версии скрипта."""
    return КОПИИ / "_indexes"


def дом(имя: str) -> Path:
    """Дом индекса одной копии: <AGENT_WORK_DIR>/_indexes/<имя>."""
    return каталог_домов() / имя


def среда(дом_копии: Path) -> dict:
    return dict(os.environ, CODE_INDEX_HOME=str(дом_копии))


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


def записать_pid(дом_копии: Path, роль: str, pid: int) -> None:
    (дом_копии / f"clean_{роль}.pid").write_text(
        json.dumps({"pid": pid, "start": время_старта(pid)}), encoding="utf-8")


def прочитать_pid(дом_копии: Path, роль: str) -> tuple[int, str] | None:
    try:
        запись = json.loads((дом_копии / f"clean_{роль}.pid").read_text(encoding="utf-8"))
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


def роль_жива(дом_копии: Path, роль: str) -> bool:
    запись = прочитать_pid(дом_копии, роль)
    return запись is not None and жив_индексатор(*запись)


def остановить_индекс(дом_копии: Path) -> None:
    # Демон — штатно, через свой дом (POST /stop); общий демон при этом не
    # задевается: адрес берётся из daemon.json этого дома. Команда возвращается
    # раньше, чем демон закроет базу в копии, — без ожидания удаление копии
    # падает с «Invalid argument» (11.09.2026).
    if (дом_копии / "daemon.json").exists():
        subprocess.run([EXE, "daemon", "stop"], env=среда(дом_копии), cwd=дом_копии,
                       capture_output=True, timeout=30)
        демон = прочитать_pid(дом_копии, "daemon")
        for _ in range(20):
            if демон is None or not жив_индексатор(*демон):
                break
            time.sleep(0.5)
    for роль in ("serve", "daemon"):
        pid_файл = дом_копии / f"clean_{роль}.pid"
        if not pid_файл.exists():
            continue
        запись = прочитать_pid(дом_копии, роль)
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


def запустить(дом_копии: Path, роль: str, аргументы: list[str]) -> subprocess.Popen:
    журнал = open(дом_копии / f"{роль}.log", "a", encoding="utf-8")
    параметры = dict(cwd=дом_копии, env=среда(дом_копии), stdin=subprocess.DEVNULL,
                     stdout=журнал, stderr=subprocess.STDOUT)
    if os.name == "nt":
        параметры["creationflags"] = CREATE_NO_WINDOW
    p = subprocess.Popen([EXE, *аргументы], **параметры)
    записать_pid(дом_копии, роль, p.pid)
    return p


def запустить_индекс(копия: Path, дом_копии: Path, язык: str, порт: int, алиас: str) -> None:
    # Значения уходят в TOML: язык — только имя, путь — экранированной строкой.
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", язык):
        raise SystemExit(f"недопустимый --language: {язык!r}")
    дом_копии.mkdir(parents=True, exist_ok=True)
    путь = копия.as_posix()
    (дом_копии / "daemon.toml").write_text(
        "# Один репозиторий — чистая копия для исполнителя (clean_copy.py).\n"
        "[daemon]\nhttp_port = 0\n\n"
        f'[[paths]]\npath = {json.dumps(путь, ensure_ascii=False)}\nalias = "{алиас}"\nlanguage = "{язык}"\n',
        encoding="utf-8")
    (дом_копии / "serve.toml").write_text(
        "# Единственный репозиторий, локальный, никакой федерации.\n"
        '[me]\nip = "127.0.0.1"\n\n'
        f'[[paths]]\nalias = "{алиас}"\nip = "127.0.0.1"\nport = {порт}\n',
        encoding="utf-8")
    # daemon.json прежнего запуска указал бы на мёртвый процесс.
    (дом_копии / "daemon.json").unlink(missing_ok=True)
    демон = запустить(дом_копии, "daemon", ["daemon", "run"])
    serve = запустить(дом_копии, "serve", [
        "serve", "--transport", "http", "--host", "127.0.0.1", "--port", str(порт),
        "--config", str(дом_копии / "daemon.toml"),
        "--serve-config", str(дом_копии / "serve.toml")])
    начало = time.time()
    while time.time() - начало < 120:
        time.sleep(1)
        # `daemon run` запускает рабочий процесс отдельно и сам выходит с кодом 0
        # (проверено 11.09.2026); настоящий PID демон пишет в daemon.json.
        # Сбой — только ненулевой код запускающего процесса.
        for роль, p in (("daemon", демон), ("serve", serve)):
            if p.poll() not in (None, 0) or (роль == "serve" and p.poll() == 0):
                хвост = (дом_копии / f"{роль}.log").read_text(encoding="utf-8")[-1500:]
                остановить_индекс(дом_копии)
                raise SystemExit(f"{роль} завершился с кодом {p.returncode}:\n{хвост}")
        try:
            ответ = вызвать(порт, "list_files", {"repo": алиас, "limit": 1})
        except Exception:
            continue          # serve ещё не слушает порт
        if not any(f'\\"status\\":\\"{с}\\"' in ответ or f'"status":"{с}"' in ответ
                   for с in НЕ_ГОТОВ):
            pid_демона = json.loads((дом_копии / "daemon.json").read_text(encoding="utf-8"))["pid"]
            записать_pid(дом_копии, "daemon", int(pid_демона))
            print(f"[индекс] готов за {time.time() - начало:.0f} с: демон PID {pid_демона}, "
                  f"serve PID {serve.pid}, порт {порт}, алиас {алиас}")
            return
    остановить_индекс(дом_копии)
    raise SystemExit("индекс копии не готов за 120 с — см. журналы в " + str(дом_копии))


def вызвать(порт: int, инструмент: str, аргументы: dict) -> str:
    """Вызов инструмента индекса по MCP; возвращает текст ответа целиком."""
    адрес = f"http://127.0.0.1:{порт}/mcp"
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


def чужие_метки(алиас: str) -> set[str]:
    """Пути и алиасы всех репозиториев общего индекса — их не должно быть в ответах."""
    метки: set[str] = set()
    for имя in ("daemon.toml", "serve.toml"):
        f = MAIN_HOME / имя
        if f.exists():
            t = f.read_text(encoding="utf-8", errors="replace")
            метки |= set(re.findall(r'^\s*(?:path|alias)\s*=\s*"([^"]+)"', t, re.M))
    метки.discard(алиас)
    return {м.replace("\\", "/").lower() for м in метки if len(м) > 3}


def проверить_изоляцию(дом_копии: Path, порт: int, алиас: str) -> None:
    чужие = чужие_метки(алиас)
    if not чужие:
        print(f"[изоляция] в {MAIN_HOME} нет конфигов общего индекса — сверять не с чем, "
              "проверка неполная (задайте CODE_INDEX_MAIN_HOME)")
    проверки = {
        "get_stats без repo": ("get_stats", {}),
        "health": ("health", {"repo": алиас}),
        "неизвестный алиас": ("get_stats", {"repo": "нет-такого-репо"}),
    }
    for название, (инструмент, арг) in проверки.items():
        ответ = вызвать(порт, инструмент, арг)
        # Ответ — JSON с экранированными обратными косыми; сравниваем в одном виде.
        плоский = ответ.replace("\\\\", "/").replace("\\", "/").lower()
        найдено = sorted(м for м in чужие if м in плоский)
        if найдено:
            остановить_индекс(дом_копии)
            raise SystemExit(f"[изоляция] {название}: в ответе чужие репозитории {найдено}"
                             " — индекс остановлен")
        print(f"[изоляция] {название}: чужих путей нет")


def секция_охраны(имя: str, копия: Path, алиас: str) -> str:
    """Секция [[local]] конфига хука для одной копии, между своими метками."""
    путь = копия.as_posix()
    if '"' in путь or "\\" in путь:
        raise SystemExit(f"путь копии {путь!r} не записать в конфиг хука")
    return (f"{МЕТКА_НАЧАЛА} {имя}\n"
            "[[local]]\n"
            f'path = "{путь}"\n'
            f'alias = "{алиас}"\n'
            f"{МЕТКА_КОНЦА} {имя}\n")


def без_секции(текст: str, имя: str) -> str:
    """Тот же конфиг без секции этой копии; чужие секции остаются как есть.
    Уходит и пустая строка перед секцией — её добавляет правка_конфига_хука,
    иначе prepare → remove оставлял бы файл на строку длиннее."""
    образец = re.compile(
        rf"(?:(?<=\n)\r?\n)?^{re.escape(МЕТКА_НАЧАЛА)} {re.escape(имя)}\r?$.*?"
        rf"^{re.escape(МЕТКА_КОНЦА)} {re.escape(имя)}\r?$\n?",
        re.MULTILINE | re.DOTALL)
    return образец.sub("", текст)


def правка_конфига_хука(имя: str, секция: str | None) -> bool:
    """Заменяет секцию этой копии в конфиге хука; секция None — удаляет её.
    Правка идёт под замком: копии готовятся параллельно, и без него одна
    правка затирала бы другую. False — менять было нечего."""
    замок = КОНФИГ_ХУКА.with_name(КОНФИГ_ХУКА.name + ".lock")
    КОНФИГ_ХУКА.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        try:
            дескриптор = os.open(замок, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.1)
            continue
        os.close(дескриптор)
        break
    else:
        raise SystemExit(f"конфиг хука занят: замок {замок} держит другая правка")
    try:
        # Концы строк файла сохраняются как были (newline=""): текстовый режим
        # Windows превращал бы LF в CRLF при каждой правке.
        try:
            with КОНФИГ_ХУКА.open(encoding="utf-8", newline="") as файл:
                текст = файл.read()
        except FileNotFoundError:
            if секция is None:
                return False
            текст = ("# Конфиг хука code-index-guard.\n"
                     "# Секции ниже добавляет clean_copy.py — по одной на чистую копию.\n")
        перевод = "\r\n" if "\r\n" in текст else "\n"
        новый = без_секции(текст, имя)
        if секция is not None:
            if новый and not новый.endswith("\n"):
                новый += перевод
            новый += перевод + секция.replace("\n", перевод)
        if новый == текст:
            return False
        временный = КОНФИГ_ХУКА.with_name(КОНФИГ_ХУКА.name + ".tmp")
        временный.write_text(новый, encoding="utf-8", newline="")
        os.replace(временный, КОНФИГ_ХУКА)
        return True
    finally:
        замок.unlink(missing_ok=True)


def под_охрану(имя: str, копия: Path, алиас: str) -> None:
    """Каталог копии — под запрет обычного чтения: хук отклонит Read, Grep и
    cat/grep/ls по нему и отправит в индекс самой копии."""
    try:
        правка_конфига_хука(имя, секция_охраны(имя, копия, алиас))
    except OSError as ошибка:
        raise SystemExit(f"[охрана] конфиг хука {КОНФИГ_ХУКА} не записан: {ошибка}; "
                         "снимите копию командой remove") from None
    print(f"[охрана] обычное чтение {копия} запрещено ({КОНФИГ_ХУКА})")


def снять_с_охраны(имя: str) -> None:
    """Снятие запрета вместе с копией. Ошибка здесь снятию копии не мешает:
    каталога уже нет, и оставшаяся секция безвредна — о ней сообщаем."""
    try:
        if правка_конфига_хука(имя, None):
            print(f"[охрана] запрет чтения снят ({КОНФИГ_ХУКА})")
    except (OSError, SystemExit) as ошибка:
        print(f"[охрана] секция копии в {КОНФИГ_ХУКА} не убрана: {ошибка}")


def порт_занят(порт: int) -> bool:
    """Занят ли порт на петлевом адресе: подключиться удалось — занят."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as сокет:
        сокет.settimeout(1)
        try:
            сокет.connect(("127.0.0.1", порт))
        except OSError:
            return False
    return True


def проверить_порт(порт: int | None) -> int:
    if порт is None or not 1024 <= порт <= 65535:
        raise SystemExit(f"недопустимый --port {порт!r}: нужно целое 1024..65535")
    return порт


def порт_от_системы() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as сокет:
        сокет.bind(("127.0.0.1", 0))
        return сокет.getsockname()[1]


def занятые_порты_копий() -> set[int]:
    порты = set()
    for файл in каталог_домов().glob("*/copy.json"):
        try:
            порт = прочитать_copy(файл.parent).get("port")
            if isinstance(порт, int):
                порты.add(порт)
        except (OSError, ValueError, UnicodeError):
            pass
    return порты


def свободный_порт() -> int:
    for _ in range(20):
        порт = порт_от_системы()
        if порт not in занятые_порты_копий():
            return порт
    raise SystemExit("не удалось выбрать свободный порт за 20 попыток")


def проверить_алиас(алиас: str | None) -> str:
    if not алиас or not re.fullmatch(r"[A-Za-z0-9_-]+", алиас):
        raise SystemExit(f"недопустимый --alias {алиас!r}: только буквы, цифры, _ и -")
    return алиас


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


def ветка_существует(проект: Path, ветка: str) -> bool:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{ветка}"],
                       cwd=проект, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode == 0


def имя_ветки_годно(проект: Path, ветка: str) -> bool:
    r = subprocess.run(["git", "check-ref-format", "--branch", ветка], cwd=проект,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0


def разобрать_файлы(значение: str | None) -> list[str] | None:
    """--files: пути через запятую, пустые элементы отброшены; не задан — None."""
    if значение is None:
        return None
    return [часть.strip() for часть in значение.split(",") if часть.strip()]


def прочитать_copy(дом_копии: Path) -> dict:
    данные = json.loads((дом_копии / "copy.json").read_text(encoding="utf-8"))
    if not isinstance(данные, dict):
        raise ValueError("copy.json — не объект JSON")
    return данные


def prepare(проект: Path, имя: str, язык: str, порт: int | None, алиас: str,
            база: str = "HEAD", файлы: list[str] | None = None) -> None:
    копия = путь_копии(имя)
    дом_копии = дом(имя)
    if копия.exists():
        raise SystemExit(f"копия {копия} уже есть — сначала remove")
    if дом_копии.exists():
        raise SystemExit(f"дом индекса {дом_копии} уже есть — сначала remove")
    авто = порт is None
    if авто:
        порт = свободный_порт()
    else:
        проверить_порт(порт)
    проверить_алиас(алиас)
    if not авто and порт_занят(порт):
        raise SystemExit(f"порт {порт} занят — возьмите другой --port")
    ветка = f"agent/{имя}"
    if not имя_ветки_годно(проект, ветка):
        raise SystemExit(f"имя копии не годится для ветки {ветка!r}")
    if ветка_существует(проект, ветка):
        raise SystemExit(f"ветка {ветка} уже есть — сначала снимите её или возьмите другое имя")
    sha = git("rev-parse", "--verify", f"{база}^{{commit}}", cwd=проект).strip()
    КОПИИ.mkdir(parents=True, exist_ok=True)
    # Хуки проекта при создании копии не выполняются — как и при переносе.
    with tempfile.TemporaryDirectory(prefix="clean-copy-hooks-") as пустые_hooks:
        git("-c", f"core.hooksPath={пустые_hooks}", "worktree", "add", "-b", ветка,
            str(копия), sha, cwd=проект)
    # Сведения о копии пишутся до запуска индекса: list и remove должны видеть
    # и копию с упавшим индексом.
    дом_копии.mkdir(parents=True, exist_ok=True)
    сведения = {
        "name": имя,
        "project": проект.resolve().as_posix(),
        "copy": копия.as_posix(),
        "branch": ветка,
        "base": sha,
        "port": порт,
        "alias": алиас,
        "language": язык,
        "files": файлы,
    }
    for попытка in range(ПОПЫТОК_ПОРТА):
        (дом_копии / "copy.json").write_text(
            json.dumps(сведения, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            запустить_индекс(копия, дом_копии, язык, порт, алиас)
            break
        except SystemExit:
            if not авто or попытка == ПОПЫТОК_ПОРТА - 1 or not порт_занят(порт):
                raise
            новый_порт = свободный_порт()
            print(f"[индекс] порт {порт} перехвачен — беру {новый_порт}")
            порт = новый_порт
            сведения["port"] = порт
    проверить_изоляцию(дом_копии, порт, алиас)
    под_охрану(имя, копия, алиас)
    адрес = f"http://127.0.0.1:{порт}/mcp"
    print(json.dumps({"work_dir": копия.as_posix(), "repo": алиас, "port": порт,
                      "branch": ветка, "base": sha, "code_index_url": адрес},
                     ensure_ascii=False))
    print(f"--code-index-url {адрес}")


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
        безопасные = безопасный_git(копия, git_dir, пустые_hooks)
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


def commit(проект: Path, имя: str, сообщение: str) -> None:
    копия = путь_копии(имя)
    дом_копии = дом(имя)
    git_dir = доверенный_git_dir(проект, копия)
    if git_dir is None:
        raise SystemExit("копия не является рабочей копией этого проекта")
    проверить_git_файл(копия, git_dir)
    try:
        сведения = прочитать_copy(дом_копии)
    except (OSError, ValueError, UnicodeError):
        raise SystemExit(f"не читаются сведения о копии {дом_копии / 'copy.json'}"
                         " — снимите копию через remove и создайте заново") from None
    ветка = f"agent/{имя}"
    if сведения.get("branch") != ветка:
        raise SystemExit(f"в {дом_копии / 'copy.json'} указана чужая ветка"
                         f" вместо {ветка}")
    зона = сведения.get("files")
    if зона is not None and not isinstance(зона, list):
        raise SystemExit(f"в {дом_копии / 'copy.json'} поле files не список")
    # Безопасные параметры git — те же, что при переносе: чужие хуки не
    # исполняются, индекс копии в коммит не попадает.
    окружение = dict(os.environ, GIT_CONFIG_NOSYSTEM="1")
    with tempfile.TemporaryDirectory(prefix="clean-copy-hooks-") as пустые_hooks:
        безопасные = безопасный_git(копия, git_dir, пустые_hooks)
        текущая_ветка = git(*безопасные, "rev-parse", "--abbrev-ref", "HEAD",
                             cwd=копия, env=окружение).strip()
        if текущая_ветка != ветка:
            raise SystemExit(f"копия сейчас на ветке {текущая_ветка}, "
                             f"ожидалась {ветка}; фиксация отменена")
        git(*безопасные, "add", "-A", cwd=копия, env=окружение)
        git(*безопасные, "rm", "--cached", "-r", "-q", "--ignore-unmatch", "--",
            ".code-index", cwd=копия, env=окружение)
        изменённые = [os.fsdecode(поле) for поле in
                      git_байты(*безопасные, "diff", "--cached", "--name-only", "-z",
                                "HEAD", cwd=копия, env=окружение).split(b"\0") if поле]
        # Предупреждение — до проверки «изменений нет»: если агент создал только
        # скрытые файлы, без него фиксация молча сообщала бы, что работы нет.
        скрытые = [файл for файл in
                   (os.fsdecode(поле).rstrip("/") for поле in
                    git_байты(*безопасные, "ls-files", "--others", "--ignored",
                              "--exclude-standard", "--directory", "-z",
                              cwd=копия, env=окружение).split(b"\0") if поле)
                   if файл != ".code-index"]
        if скрытые:
            print("[фиксация] внимание: не попали в коммит (скрыты .gitignore "
                  "или .git/info/exclude):")
            for файл in скрытые[:20]:
                print("  " + файл)
            if len(скрытые) > 20:
                print(f"  и ещё {len(скрытые) - 20}")
        if not изменённые:
            print("[фиксация] изменений в копии нет")
            return
        if зона is not None:
            вне = [путь for путь in изменённые
                   if not any(fnmatch.fnmatchcase(путь, шаблон) for шаблон in зона)]
            if вне:
                # Снять индексацию, файлы в копии не трогать.
                git(*безопасные, "reset", "-q", cwd=копия, env=окружение)
                raise SystemExit("[фиксация] вне зоны --files: " + ", ".join(вне))
        git(*безопасные, "commit", "-q", "-m", сообщение, cwd=копия, env=окружение)
        sha = git(*безопасные, "rev-parse", "--short", "HEAD",
                  cwd=копия, env=окружение).strip()
    print(f"[фиксация] {sha} на ветке {ветка}: {len(изменённые)} файлов")


def список() -> None:
    for файл in sorted(каталог_домов().glob("*/copy.json")):
        try:
            сведения = прочитать_copy(файл.parent)
        except (OSError, ValueError, UnicodeError) as ошибка:
            print(json.dumps({"name": файл.parent.name, "error": str(ошибка)},
                             ensure_ascii=False))
            continue
        запись = dict(сведения)
        копия = сведения.get("copy")
        запись["copy_exists"] = bool(isinstance(копия, str) and копия
                                     and Path(копия).is_dir())
        запись["daemon_alive"] = роль_жива(файл.parent, "daemon")
        запись["serve_alive"] = роль_жива(файл.parent, "serve")
        print(json.dumps(запись, ensure_ascii=False))


def remove(проект: Path, имя: str) -> None:
    копия = путь_копии(имя)
    дом_копии = дом(имя)
    была_рабочей_копией = False
    if копия.exists():
        была_рабочей_копией = (доверенный_git_dir(проект, копия) is not None
                               or копия_в_списке_worktree(проект, копия))
        if not была_рабочей_копией:
            raise SystemExit("копия не является рабочей копией этого проекта")
    остановить_индекс(дом_копии)
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
    снять_с_охраны(имя)
    # Ветку agent/<имя> не трогаем: на ней зафиксированная работа.
    if дом_копии.exists():
        try:
            shutil.rmtree(дом_копии, ignore_errors=False)
        except OSError as ошибка:
            raise SystemExit(f"[индекс] дом {дом_копии} не удалён: {ошибка}") from None
        else:
            print(f"[индекс] дом {дом_копии} удалён")


def remove_all(проект: Path) -> None:
    цель = проект.resolve()
    ошибки: list[str] = []
    for файл in sorted(каталог_домов().glob("*/copy.json")):
        try:
            сведения = прочитать_copy(файл.parent)
        except (OSError, ValueError, UnicodeError) as ошибка:
            ошибки.append(f"{файл.parent.name}: {ошибка}")
            continue
        записанный = сведения.get("project")
        if not isinstance(записанный, str) or not записанный:
            ошибки.append(f"{файл.parent.name}: в copy.json нет поля project")
            continue
        try:
            совпадает = Path(записанный).resolve() == цель
        except OSError:
            совпадает = False
        if not совпадает:
            continue
        # Имя копии задаёт каталог дома; поле в JSON не должно
        # позволять снять другую копию.
        имя = файл.parent.name
        try:
            remove(проект, имя)
        except SystemExit as ошибка:
            ошибки.append(f"{имя}: {ошибка}")
        except Exception as ошибка:
            ошибки.append(f"{имя}: {ошибка}")
    if ошибки:
        raise SystemExit("не сняты копии:\n  " + "\n  ".join(ошибки))


def main() -> None:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument("команда", choices=("prepare", "apply", "commit", "list", "remove"))
    разбор.add_argument("проект", type=Path, nargs="?", help="корень git-репозитория проекта")
    разбор.add_argument("имя", nargs="?", help="имя копии — каталог в AGENT_WORK_DIR")
    разбор.add_argument("--language", default="python",
                        help="язык индекса копии (как в daemon.toml)")
    разбор.add_argument("--port", type=str, help="порт индекса копии, целое 1024..65535 или auto")
    разбор.add_argument("--alias", help="алиас репозитория копии в индексе")
    разбор.add_argument("--base", default="HEAD",
                        help="коммит или ветка проекта — начало ветки agent/<имя>")
    разбор.add_argument("--files", help="зона правок: пути от корня проекта через запятую, "
                                        "допускаются шаблоны (fnmatch)")
    разбор.add_argument("--message", help="текст фиксации для commit")
    разбор.add_argument("--all", action="store_true", help="remove: снять все копии проекта")
    a = разбор.parse_args()
    if a.команда == "prepare":
        if a.проект is None or not a.имя:
            raise SystemExit("для prepare нужны <проект> и <имя>")
        if a.port is None or not a.alias:
            raise SystemExit("для prepare обязательны --port и --alias")
    elif a.команда == "commit":
        if a.проект is None or not a.имя:
            raise SystemExit("для commit нужны <проект> и <имя>")
        if not a.message:
            raise SystemExit("для commit обязателен --message с текстом фиксации")
    elif a.команда == "apply":
        if a.проект is None or not a.имя:
            raise SystemExit("для apply нужны <проект> и <имя>")
    elif a.команда == "list":
        if a.проект is not None or a.имя is not None:
            raise SystemExit("для list не нужны <проект> и <имя>")
    elif a.команда == "remove":
        if a.проект is None:
            raise SystemExit("для remove нужен <проект>")
        if bool(a.имя) == bool(a.all):
            raise SystemExit("для remove нужно ровно одно: <имя> или --all")
    if a.команда in ("prepare", "remove") and not EXE:
        raise SystemExit("не найден индексатор: задайте CODE_INDEX_EXE "
                         "или добавьте bsl-indexer в PATH")
    if a.команда == "prepare":
        if a.port == "auto":
            порт = None
        else:
            try:
                порт = int(a.port)
            except ValueError:
                raise SystemExit(f"недопустимый --port {a.port!r}")
            проверить_порт(порт)
        prepare(a.проект, a.имя, a.language, порт, a.alias, a.base,
                разобрать_файлы(a.files))
    elif a.команда == "commit":
        commit(a.проект, a.имя, a.message)
    elif a.команда == "apply":
        apply(a.проект, a.имя)
    elif a.команда == "list":
        список()
    elif a.all:
        remove_all(a.проект)
    else:
        remove(a.проект, a.имя)


if __name__ == "__main__":
    main()
