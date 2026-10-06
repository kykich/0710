# -*- coding: utf-8 -*-
"""
Блок подключения ЛОКАЛЬНОЙ модели (llama.cpp / Ollama / LM Studio).

Модуль решает три задачи (по требованиям):

1. СКАНИРОВАНИЕ папки. Пользователь указывает КОРЕНЬ (напр.
   ``E:\\Models\\llama``). Модуль рекурсивно ищет:
     * исполняемый файл сервера: ``llama-server.exe`` (Windows) /
       ``llama-server`` (*nix), а также ``ollama.exe`` / ``ollama``;
     * файлы моделей (``*.gguf``);
     * известные менеджеры (Ollama, LM Studio) — по косвенным признакам.
   Путь валидируется: сервер запускается ТОЛЬКО из указанной пользователем
   папки (без поиска по всей файловой системе).

2. АВТОПОДБОР ПАРАМЕТРОВ ЗАПУСКА. По найденной модели и объёму свободной
   видеопамяти (``nvidia-smi``, если есть) рассчитывается число слоёв на
   GPU (``-ngl``). Возвращается Готовый набор параметров (модель, host,
   port, ctx, ngl, threads, flash-attn, jinja, api-key) — все значения
   доступны для РЕДАКТИРОВАНИЯ в интерфейсе.

3. ЗАПУСК СЕРВЕРА. Поднимает ``llama-server`` подпроцессом с выбранными
   параметрами, ждёт готовности (опрос ``/v1/models``), отдаёт URL
   ``http://<host>:<port>/v1`` (OpenAI-совместимый). Остановка — ВРУЧНУЮ.

Модуль кроссплатформенный: разделители путей, имена исполняемых файлов и
запуск процесса различаются для Windows и Linux/macOS.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

from . import config

__all__ = [
    "LocalServerManager",
    "scan_root",
    "probe_vram_mb",
    "suggest_params",
    "build_command",
    "server_url",
]


# ---------------------------------------------------------------------------
# Платформенные помощники
# ---------------------------------------------------------------------------
def _is_windows():
    return os.name == "nt"


def _server_exe_name():
    """Имя исполняемого файла llama.cpp-сервера для текущей ОС."""
    return (config.LOCAL_SERVER_EXE_WIN if _is_windows()
            else config.LOCAL_SERVER_EXE_NIX)


def _find_executable(root, names):
    """Ищет исполняемый файл из names рекурсивно в root.

    Возвращает первый найденный абсолютный путь или None. Ищет по имени
    файла (регистр для Windows — не важен). НЕ спускается вглубь без
    надобности: обходит дерево, но останавливается на первом совпадении.
    """
    if not root or not os.path.isdir(root):
        return None
    wanted = {n.lower() for n in names}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower() in wanted:
                return os.path.join(dirpath, f)
    return None


def _find_in_path(name):
    """Ищет исполняемый файл в PATH (кроссплатформенно)."""
    return shutil.which(name) or shutil.which(os.path.splitext(name)[0])


# ---------------------------------------------------------------------------
# Сканирование корня
# ---------------------------------------------------------------------------
def scan_root(root):
    """Сканирует КОРЕНЬ и возвращает найденное (сервер, модели, менеджеры).

    root — путь к папке (строка). Возвращает dict:
        ok        — найден ли хоть один сервер/модель;
        root      — нормализованный путь (или исходный, если нет папки);
        exists    — существует ли папка;
        servers   — список {kind, path, name} найденных серверов
                    (kind: "llama.cpp", "ollama");
        models    — список {path, name, size_mb} файлов *.gguf;
        managers  — список строк-признаков установленных менеджеров
                    ("Ollama", "LM Studio");
        errors    — список сообщений об ошибках сканирования.
    """
    root = str(root or "").strip().strip('"')
    result = {"ok": False, "root": root, "exists": False,
              "servers": [], "models": [], "managers": [], "errors": []}
    if not root:
        result["errors"].append("путь не указан")
        return result
    if not os.path.isdir(root):
        result["errors"].append("папка не найдена: %s" % root)
        return result
    result["exists"] = True

    # 1) Сервер llama.cpp: ищем .exe/*nix-бинарник в самой папке и внутри.
    exe_name = _server_exe_name()
    llama_path = _find_executable(root, [exe_name])
    if not llama_path:
        # На *nix бинарник без расширения; на Windows — с .exe.
        llama_path = _find_executable(root, ["llama-server"])
    if llama_path and os.path.isfile(llama_path):
        result["servers"].append({
            "kind": "llama.cpp",
            "path": llama_path,
            "name": os.path.basename(llama_path),
        })

    # 2) Ollama-сервер (ollama.exe / ollama) — как альтернатива.
    ollama = (_find_executable(root, ["ollama.exe", "ollama"])
              or _find_in_path("ollama"))
    if ollama:
        result["servers"].append({
            "kind": "ollama",
            "path": ollama,
            "name": os.path.basename(ollama),
        })

    # 3) Модели *.gguf рекурсивно.
    ext = str(getattr(config, "LOCAL_MODEL_EXT", ".gguf") or ".gguf").lower()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                p = os.path.join(dirpath, f)
                try:
                    size_mb = int(os.path.getsize(p) / (1024 * 1024))
                except OSError:
                    size_mb = 0
                result["models"].append({
                    "path": p, "name": f, "size_mb": size_mb,
                })
    result["models"].sort(key=lambda m: m["name"].lower())

    # 4) Признаки менеджеров: папка LM Studio / каталог моделей Ollama.
    low = root.lower()
    if "lm" in low and "studio" in low:
        result["managers"].append("LM Studio")
    if os.path.isdir(os.path.join(root, "blobs")) or \
            os.path.isdir(os.path.join(root, ".ollama")):
        result["managers"].append("Ollama")

    result["ok"] = bool(result["servers"] or result["models"])
    return result


# ---------------------------------------------------------------------------
# Видеопамять (для автоподбора -ngl)
# ---------------------------------------------------------------------------
def probe_vram_mb():
    """Свободная видеопамять GPU (МБ) через nvidia-smi, либо None.

    Возвращает суммарную свободную память по всем NVIDIA-GPU в МБ. Если
    nvidia-smi нет/недоступен — None (тогда -ngl подбирается без учёта
    видеопамяти, по умолчанию «все слои»).
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10)
        total = 0
        for line in out.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line.isdigit():
                total += int(line)
        return total or None
    except Exception:
        return None


def suggest_params(model_path, vram_free_mb=None):
    """Подбирает параметры запуска под найденную модель.

    model_path — путь к .gguf. Возвращает dict с предлагаемыми значениями:
        model, host, port, ctx, ngl, threads, flash_attn, jinja, api_key,
        model_mb, vram_free_mb, ngl_auto (признак автоподбора).

    Логика -ngl: если известна свободная видеопамять и размер модели —
    прикидываем, сколько слоёв выгрузить на GPU. Для простоты: если модель
    помещается в память с запасом — предлагаем все слои (``-ngl 99``);
    иначе — часть слоёв пропорционально доступной памяти. Точное число
    слоёв модели неизвестно без разбора метаданных, поэтому используем
    консервативную оценку по размеру файла.
    """
    model_mb = 0
    try:
        model_mb = int(os.path.getsize(model_path) / (1024 * 1024))
    except OSError:
        model_mb = 0
    if vram_free_mb is None:
        vram_free_mb = probe_vram_mb()

    ngl = int(getattr(config, "LOCAL_NGL", 0) or 0)
    ngl_auto = False
    if ngl == 0:                                  # 0 → автоподбор
        ngl_auto = True
        reserve = int(getattr(config, "LOCAL_VRAM_RESERVE_MB", 512) or 512)
        if vram_free_mb and model_mb:
            usable = max(0, vram_free_mb - reserve)
            if usable >= model_mb:
                # Модель помещается с запасом — все слои на GPU.
                ngl = 99
            else:
                # Частичная выгрузка: доля слоёв ~ доля памяти.
                frac = usable / float(model_mb)
                # Оценка «типичного» числа слоёв: консервативно 40.
                ngl = max(1, int(40 * frac))
        else:
            # Видеопамять неизвестна — все слои (пользователь поправит).
            ngl = 99

    threads = int(getattr(config, "LOCAL_THREADS", 0) or 0)
    if threads <= 0:
        # Число физических ядер (делённое logical на 2 — грубая, но безопасная
        # оценка; пользователь может переопределить).
        logical = os.cpu_count() or 8
        threads = max(1, logical // 2)

    return {
        "model": model_path,
        "host": config.LOCAL_HOST,
        "port": int(config.LOCAL_PORT),
        "ctx": int(config.LOCAL_CTX),
        "ngl": int(ngl),
        "threads": int(threads),
        "flash_attn": config.LOCAL_FLASH_ATTN,
        "jinja": bool(config.LOCAL_JINJA),
        "api_key": config.LOCAL_API_KEY,
        "model_mb": model_mb,
        "vram_free_mb": vram_free_mb,
        "ngl_auto": ngl_auto,
    }


# ---------------------------------------------------------------------------
# Формирование команды и URL
# ---------------------------------------------------------------------------
def build_command(server_path, params):
    """Собирает командную строку запуска llama-server по параметрам.

    server_path — путь к llama-server(.exe). params — dict (см. suggest_params).
    Возвращает список аргументов (для subprocess). Неизвестные/пустые
    значения пропускаются, api_key добавляется только если задан.
    """
    cmd = [server_path]
    model = str(params.get("model") or "").strip()
    if model:
        cmd += ["-m", model, "--alias", str(params.get("alias")
                                            or config.LOCAL_LABEL)]
    host = str(params.get("host") or config.LOCAL_HOST)
    cmd += ["--host", host]
    cmd += ["--port", str(int(params.get("port") or config.LOCAL_PORT))]
    try:
        cmd += ["-c", str(int(params.get("ctx") or config.LOCAL_CTX))]
    except (TypeError, ValueError):
        cmd += ["-c", str(config.LOCAL_CTX)]
    try:
        cmd += ["-ngl", str(int(params.get("ngl")))]
    except (TypeError, ValueError):
        pass
    try:
        th = int(params.get("threads") or 0)
        if th > 0:
            cmd += ["-t", str(th)]
    except (TypeError, ValueError):
        pass
    fa = str(params.get("flash_attn") or "").strip().lower()
    if fa in ("on", "off", "auto"):
        cmd += ["-fa", fa]
    if params.get("jinja"):
        cmd += ["--jinja"]
    key = str(params.get("api_key") or "").strip()
    if key:
        cmd += ["--api-key", key]
    return cmd


def server_url(host, port):
    """Базовый URL OpenAI-совместимого API локального сервера (…/v1)."""
    return "http://%s:%d/v1" % (host or config.LOCAL_HOST,
                                int(port or config.LOCAL_PORT))


# ---------------------------------------------------------------------------
# Менеджер процесса
# ---------------------------------------------------------------------------
class LocalServerManager:
    """Запуск/остановка локального LLM-сервера и его статус.

    Потокобезопасен (запуск/остановка — под замком). Держит ОДИН процесс
    (llama-server). Остановка — вручную (метод stop()).
    """

    def __init__(self, settings_file=None):
        self.settings_file = settings_file or config.LOCAL_SETTINGS_FILE
        self._proc = None
        self._lock = threading.RLock()
        self._last_log = ""            # последние строки вывода процесса
        self._last_error = None

    # ---- настройки на диске ----
    def load_settings(self):
        """Читает сохранённые настройки блока (или значения по умолчанию)."""
        default = {
            "enabled": bool(config.LOCAL_ENABLED),
            "root": config.LOCAL_ROOT,
            "server_path": "",
            "params": {},
        }
        try:
            with open(self.settings_file, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                default.update(data)
        except Exception:
            pass
        return default

    def save_settings(self, settings):
        """Сохраняет настройки блока в файл (best-effort)."""
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:                       # noqa: BLE001
            self._last_error = str(exc)
            return False

    # ---- процесс ----
    def _proc_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def is_running(self):
        """Запущен ли сейчас процесс локального сервера."""
        with self._lock:
            return self._proc_alive()

    def start(self, server_path, params):
        """Запускает llama-server и ждёт готовности.

        server_path — путь к llama-server(.exe); params — dict параметров.
        Возвращает dict:
            {ok, url, pid, error, log, model}
        ready — после успешного опроса /v1/models; при ошибке ok=False.
        Процесс запускается ТОЛЬКО по явному вызову (кнопка «Запустить»).
        """
        with self._lock:
            if self._proc_alive():
                return {"ok": True, "url": server_url(params.get("host"),
                                                      params.get("port")),
                        "pid": self._proc.pid, "already": True,
                        "error": None, "model": params.get("model")}
            if not server_path or not os.path.isfile(server_path):
                return {"ok": False, "error": "сервер не найден: %s"
                                              % server_path}
            model = str(params.get("model") or "").strip()
            if not model or not os.path.isfile(model):
                return {"ok": False, "error": "модель не найдена: %s" % model}

            cmd = build_command(server_path, params)
            creationflags = 0
            if _is_windows():
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            cwd = os.path.dirname(server_path) or None
            try:
                self._proc = subprocess.Popen(
                    cmd, cwd=cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            except Exception as exc:                   # noqa: BLE001
                self._last_error = str(exc)
                self._proc = None
                return {"ok": False, "error": "не удалось запустить сервер: %s"
                                              % exc}

            # Ждём готовности: опрашиваем /v1/models.
            url = server_url(params.get("host"), params.get("port"))
            deadline = time.time() + int(getattr(config, "LOCAL_START_TIMEOUT",
                                                 180) or 180)
            ready = False
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    # Процесс завершился — читаем вывод для диагностики.
                    out = self._read_process_output()
                    return {"ok": False, "pid": None, "url": url,
                            "error": "процесс сервера завершился при запуске",
                            "log": out, "model": model}
                if self._probe_ready(url):
                    ready = True
                    break
                time.sleep(1.0)

            if not ready:
                self.stop()
                return {"ok": False, "pid": None, "url": url,
                        "error": "сервер не ответил за %d с"
                                 % int(getattr(config, "LOCAL_START_TIMEOUT",
                                               180) or 180),
                        "model": model}
            return {"ok": True, "pid": self._proc.pid, "url": url,
                    "error": None, "model": model}

    @staticmethod
    def _probe_ready(url):
        """Проверяет готовность сервера: GET <url>/models (OpenAI-совместимо)."""
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models",
                                         method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _read_process_output(self, max_chars=2000):
        """Читает (не блокируя) накопленный вывод процесса для диагностики."""
        if not self._proc or not self._proc.stdout:
            return ""
        try:
            data = self._proc.stdout.read() or b""
            text = data.decode("utf-8", "replace")
            self._last_log = text[-max_chars:]
            return self._last_log
        except Exception:
            return self._last_log

    def stop(self):
        """Останавливает процесс сервера (вручную). Возвращает True/False."""
        with self._lock:
            if self._proc is None:
                return False
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=5)
            except Exception as exc:                   # noqa: BLE001
                self._last_error = str(exc)
            finally:
                self._proc = None
            return True

    def status(self, host=None, port=None):
        """Текущий статус: запущен ли, URL, PID, ответ сервера (models)."""
        running = self.is_running()
        st = {"running": running, "pid": (self._proc.pid if running else None),
              "url": None, "ready": False, "models": [], "error": None}
        settings = self.load_settings()
        params = settings.get("params") or {}
        h = host or params.get("host") or config.LOCAL_HOST
        p = port or params.get("port") or config.LOCAL_PORT
        st["url"] = server_url(h, p)
        if running:
            st["ready"] = self._probe_ready(st["url"])
            if st["ready"]:
                st["models"] = self._list_server_models(st["url"])
        return st

    @staticmethod
    def _list_server_models(url):
        """Список моделей от локального сервера (GET /v1/models -> data[].id)."""
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models",
                                         method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data") or []
            return [str(it.get("id")) for it in items
                    if isinstance(it, dict) and it.get("id")]
        except Exception:
            return []
